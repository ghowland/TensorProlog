// ============================================================
// vlp_builtin.zig
// Builtin executor — host dispatches, GPU or host executes.
// 448 builtins: 404 pure (GPU-eligible), 44 operational (host-only).
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const gpu = @import("vlp_gpu_params.zig");
const kb_mod = @import("vlp_kb_store.zig");

// ============================================================
// Builtin categories
// ============================================================

pub const BuiltinCategory = enum(i32) {
    // Pure — GPU-eligible
    text_ops = 0,
    collections = 1,
    sets = 2,
    mappings = 3,
    closed_arithmetic = 4,
    comparison = 5,
    rounding = 6,
    integer_bit_ops = 7,
    linear_algebra = 8,
    statistics = 9,
    active_arithmetic = 10,
    structure_ops = 11,
    number_theory = 12,
    polynomial = 13,
    finite_field = 14,
    discrete_calculus = 15,
    // Operational — host-only, grant-gated
    op_filesystem = 16,
    op_compile = 17,
    op_execute = 18,
    op_lint = 19,
    op_network = 20,
    op_process = 21,
};

// ============================================================
// IOSE declaration — Input/Output/SideEffects/Errors
// ============================================================

pub const IoSe = struct {
    builtin_id: i32,
    category: BuiltinCategory,
    name: [64]u8,
    name_len: i32,
    input_count: i32,
    input_types: [8]types.TermType,
    output_type: types.TermType,
    side_effects: bool,
    grant_class: i8, // -1 = none required
    max_input_elements: i32, // -1 = unbounded
    bounded: bool, // termination guaranteed
    deterministic: bool, // same input → same output

    pub fn requiresGrant(self: IoSe) bool {
        return self.grant_class >= 0;
    }

    pub fn grantRequired(self: IoSe) ?types.GrantClass {
        if (self.grant_class < 0) return null;
        return @enumFromInt(self.grant_class);
    }

    pub fn isPure(self: IoSe) bool {
        return !self.side_effects and self.deterministic;
    }
};

// ============================================================
// Builtin arguments and results
// ============================================================

pub const BuiltinArgs = struct {
    input_kb_id: i32,
    input_slot_ids: []const i32,
    output_kb_id: i32,
    output_slot_id: i32,
    extra_params: []const i32,
    input_array_length: i32, // for array builtins
};

pub const BuiltinResult = struct {
    status: types.Status,
    output_kb_id: i32,
    output_slot_id: i32,
    output_count: i32, // number of output elements

    pub fn ok(kb_id: i32, slot_id: i32, count: i32) BuiltinResult {
        return .{
            .status = types.Status.ok(),
            .output_kb_id = kb_id,
            .output_slot_id = slot_id,
            .output_count = count,
        };
    }

    pub fn err(status: types.Status) BuiltinResult {
        return .{
            .status = status,
            .output_kb_id = -1,
            .output_slot_id = -1,
            .output_count = 0,
        };
    }
};

// ============================================================
// GPU pipeline mapping — which pipeline handles which builtins
// ============================================================

const PipelineMapping = struct {
    pipeline: gpu.PipelineId,
    op_code: i32,
};

fn builtinToPipeline(builtin_id: i32) ?PipelineMapping {
    // Map builtin_id ranges to GPU pipeline + op_code
    // 0-34:   unary ops     → builtin_unary
    // 35-74:  binary ops    → builtin_binary
    // 75-99:  reductions    → builtin_reduction
    // 100-114: sort/search  → builtin_sort
    // 115-144: linear algebra → builtin_matmul
    // 145-164: statistics   → builtin_reduction (different op_codes)
    // 165-194: comparison   → builtin_binary
    // 195-209: confidence   → builtin_confidence_combine / chain
    // 210-403: other pure   → various
    // 404-447: operational  → no GPU pipeline (host-only)
    if (builtin_id < 0 or builtin_id >= 448) return null;
    if (builtin_id >= 404) return null; // operational

    if (builtin_id < 35) return .{ .pipeline = .builtin_unary, .op_code = builtin_id };
    if (builtin_id < 75) return .{ .pipeline = .builtin_binary, .op_code = builtin_id - 35 };
    if (builtin_id < 100) return .{ .pipeline = .builtin_reduction, .op_code = builtin_id - 75 };
    if (builtin_id < 115) return .{ .pipeline = .builtin_sort, .op_code = builtin_id - 100 };
    if (builtin_id < 145) return .{ .pipeline = .builtin_matmul, .op_code = builtin_id - 115 };
    // Everything else: unary as catch-all for now
    return .{ .pipeline = .builtin_unary, .op_code = 0 };
}

// ============================================================
// Builtin Executor
// ============================================================

pub const BuiltinExecutor = struct {
    bridge: *bridge_mod.Bridge,
    kb_store: *kb_mod.KbStore,
    allocator: std.mem.Allocator,
    iose_table: [448]IoSe,
    initialized: bool,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(bridge: *bridge_mod.Bridge, kb_store: *kb_mod.KbStore, allocator: std.mem.Allocator) BuiltinExecutor {
    var executor = BuiltinExecutor{
        .bridge = bridge,
        .kb_store = kb_store,
        .allocator = allocator,
        .iose_table = undefined,
        .initialized = false,
    };

    // Initialize IOSE table with defaults
    for (&executor.iose_table, 0..) |*entry, i| {
        entry.* = std.mem.zeroes(IoSe);
        entry.builtin_id = @intCast(i);
        entry.bounded = true;
        entry.deterministic = true;
        entry.side_effects = false;
        entry.grant_class = -1;
        entry.output_type = .vdr;
        entry.max_input_elements = -1;

        // Mark operational builtins
        if (i >= 404) {
            entry.side_effects = true;
            entry.deterministic = false;
            entry.category = categoryForOperational(@intCast(i));
            entry.grant_class = grantForOperational(@intCast(i));
        }
    }

    executor.initialized = true;
    return executor;
}

pub fn deinit(self: *BuiltinExecutor) void {
    self.initialized = false;
}

// ============================================================
// IOSE introspection
// ============================================================

pub fn getIoSe(self: *BuiltinExecutor, builtin_id: i32) ?IoSe {
    if (builtin_id < 0 or builtin_id >= 448) return null;
    return self.iose_table[@intCast(builtin_id)];
}

pub fn isOperational(builtin_id: i32) bool {
    return builtin_id >= 404 and builtin_id < 448;
}

pub fn requiredGrant(self: *BuiltinExecutor, builtin_id: i32) ?types.GrantClass {
    if (builtin_id < 0 or builtin_id >= 448) return null;
    return self.iose_table[@intCast(builtin_id)].grantRequired();
}

// ============================================================
// Validation
// ============================================================

pub fn validateIoSe(self: *BuiltinExecutor, builtin_id: i32, args: *const BuiltinArgs) types.Status {
    if (builtin_id < 0 or builtin_id >= 448) return types.Status.err(.arithmetic, .overflow, builtin_id);

    const iose = &self.iose_table[@intCast(builtin_id)];

    // Check input count
    if (@as(i32, @intCast(args.input_slot_ids.len)) < iose.input_count) {
        return types.Status.err(.arithmetic, .overflow, iose.input_count);
    }

    // Check input element limit
    if (iose.max_input_elements >= 0 and args.input_array_length > iose.max_input_elements) {
        return types.Status.err(.arithmetic, .overflow, args.input_array_length);
    }

    // Check output KB exists and is writable
    if (self.kb_store.getKb(args.output_kb_id)) |kb| {
        if (kb.isFrozen()) return types.Status.err(.kb, .kb_frozen, args.output_kb_id);
    } else {
        return types.Status.err(.kb, .kb_not_found, args.output_kb_id);
    }

    return types.Status.ok();
}

// ============================================================
// Dispatch — routes to GPU or host
// ============================================================

pub fn dispatch(self: *BuiltinExecutor, builtin_id: i32, args: *const BuiltinArgs) BuiltinResult {
    // Validate first
    const val_status = self.validateIoSe(builtin_id, args);
    if (val_status.isErr()) return BuiltinResult.err(val_status);

    // Operational builtins — always host
    if (isOperational(builtin_id)) {
        return self.dispatchOperational(builtin_id, args);
    }

    // Pure builtins — GPU if array is large enough
    if (builtinToPipeline(builtin_id)) |mapping| {
        if (self.bridge.shouldUseGpu(.builtin_array, args.input_array_length)) {
            return self.dispatchGpu(mapping, args);
        }
    }

    // Host fallback for small arrays or unmapped builtins
    return self.dispatchHost(builtin_id, args);
}

fn dispatchGpu(self: *BuiltinExecutor, mapping: PipelineMapping, args: *const BuiltinArgs) BuiltinResult {
    const n = args.input_array_length;

    // Load input data from KB to scratch_a
    const slot_ids = self.allocator.alloc(i32, @intCast(args.input_slot_ids.len)) catch
        return BuiltinResult.err(types.Status.err(.device, .device_out_of_memory, 0));
    defer self.allocator.free(slot_ids);
    @memcpy(slot_ids, args.input_slot_ids);

    // Read input facts
    const facts = self.allocator.alloc(types.Fact, @intCast(args.input_slot_ids.len)) catch
        return BuiltinResult.err(types.Status.err(.device, .device_out_of_memory, 0));
    defer self.allocator.free(facts);
    _ = self.kb_store.factReadBatch(args.input_kb_id, slot_ids, facts);

    // Extract Q16 values and upload to scratch_a
    var values = self.allocator.alloc(i32, @intCast(n)) catch
        return BuiltinResult.err(types.Status.err(.device, .device_out_of_memory, 0));
    defer self.allocator.free(values);
    for (facts, 0..) |f, i| {
        if (i >= values.len) break;
        values[i] = f.value.v;
    }
    const val_bytes: []const u8 = @as([*]const u8, @ptrCast(values.ptr))[0 .. values.len * 4];
    _ = self.bridge.uploadToBuffer(.scratch_a, 0, val_bytes);

    // Dispatch based on pipeline type
    _ = self.bridge.resetResultCounts();
    var status: types.Status = undefined;

    switch (mapping.pipeline) {
        .builtin_unary => {
            var params = gpu.BuiltinUnaryParams{
                .n_elements = n,
                .op_code = mapping.op_code,
                .input_offset = 0,
                .output_offset = 0,
            };
            status = self.bridge.dispatch(&.{
                .pipeline = .builtin_unary,
                .group_count_x = @divTrunc(n + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
                .group_count_y = 1,
                .group_count_z = 1,
                .params_ptr = @ptrCast(&params),
                .params_size = @sizeOf(gpu.BuiltinUnaryParams),
            });
        },
        .builtin_binary => {
            var params = gpu.BuiltinBinaryParams{
                .n_elements = n,
                .op_code = mapping.op_code,
                .input_a_offset = 0,
                .input_b_offset = n * 4, // second input follows first
                .output_offset = 0,
            };
            status = self.bridge.dispatch(&.{
                .pipeline = .builtin_binary,
                .group_count_x = @divTrunc(n + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
                .group_count_y = 1,
                .group_count_z = 1,
                .params_ptr = @ptrCast(&params),
                .params_size = @sizeOf(gpu.BuiltinBinaryParams),
            });
        },
        .builtin_reduction => {
            var params = gpu.BuiltinReductionParams{
                .n_elements = n,
                .op_code = mapping.op_code,
                .input_offset = 0,
            };
            status = self.bridge.dispatch(&.{
                .pipeline = .builtin_reduction,
                .group_count_x = 1, // single workgroup for reduction
                .group_count_y = 1,
                .group_count_z = 1,
                .params_ptr = @ptrCast(&params),
                .params_size = @sizeOf(gpu.BuiltinReductionParams),
            });
        },
        .builtin_sort => {
            var params = gpu.BuiltinSortParams{
                .n_elements = n,
                .ascending = if (args.extra_params.len > 0) args.extra_params[0] else 1,
                .input_offset = 0,
                .output_offset = 0,
            };
            status = self.bridge.dispatch(&.{
                .pipeline = .builtin_sort,
                .group_count_x = 1,
                .group_count_y = 1,
                .group_count_z = 1,
                .params_ptr = @ptrCast(&params),
                .params_size = @sizeOf(gpu.BuiltinSortParams),
            });
        },
        .builtin_matmul => {
            const m = if (args.extra_params.len > 0) args.extra_params[0] else n;
            const k = if (args.extra_params.len > 1) args.extra_params[1] else 1;
            const nn = if (args.extra_params.len > 2) args.extra_params[2] else 1;
            var params = gpu.BuiltinMatmulParams{
                .m = m,
                .n = nn,
                .k = k,
                .a_offset = 0,
                .b_offset = m * k * 4,
                .c_offset = 0,
            };
            status = self.bridge.dispatch(&.{
                .pipeline = .builtin_matmul,
                .group_count_x = @divTrunc(m * nn + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
                .group_count_y = 1,
                .group_count_z = 1,
                .params_ptr = @ptrCast(&params),
                .params_size = @sizeOf(gpu.BuiltinMatmulParams),
            });
        },
        else => {
            status = types.Status.err(.arithmetic, .overflow, mapping.op_code);
        },
    }

    if (status.isErr()) return BuiltinResult.err(status);

    // Download result from scratch_b, write to output KB
    const result_values = self.allocator.alloc(i32, @intCast(n)) catch
        return BuiltinResult.err(types.Status.err(.device, .device_out_of_memory, 0));
    defer self.allocator.free(result_values);
    const res_bytes: []u8 = @as([*]u8, @ptrCast(result_values.ptr))[0 .. result_values.len * 4];
    _ = self.bridge.downloadFromBuffer(.scratch_b, 0, res_bytes);

    // Write result as fact
    const result_fact = types.Fact{
        .tag = .value,
        .value = types.Q16.fromParts(result_values[0], 0),
        .provenance = types.Provenance.direct(.vdr_computation, args.output_kb_id, args.output_slot_id, kb_mod.currentTimestamp()),
    };
    _ = self.kb_store.factWrite(args.output_kb_id, args.output_slot_id, &result_fact);

    return BuiltinResult.ok(args.output_kb_id, args.output_slot_id, n);
}

fn dispatchHost(self: *BuiltinExecutor, builtin_id: i32, args: *const BuiltinArgs) BuiltinResult {
    // Host-side execution for small arrays or unmapped builtins
    // Read input, compute, write output
    if (args.input_slot_ids.len == 0) return BuiltinResult.err(types.Status.err(.arithmetic, .overflow, 0));

    const fact_a = self.kb_store.factRead(args.input_kb_id, args.input_slot_ids[0]) orelse
        return BuiltinResult.err(types.Status.err(.kb, .slot_empty, args.input_slot_ids[0]));

    var result_value = fact_a.value;

    // Dispatch by builtin_id for common operations
    if (builtin_id < 35) {
        // Unary: apply to single value
        result_value = hostUnary(builtin_id, fact_a.value);
    } else if (builtin_id < 75 and args.input_slot_ids.len >= 2) {
        // Binary: two inputs
        const fact_b = self.kb_store.factRead(args.input_kb_id, args.input_slot_ids[1]) orelse
            return BuiltinResult.err(types.Status.err(.kb, .slot_empty, args.input_slot_ids[1]));
        result_value = hostBinary(builtin_id - 35, fact_a.value, fact_b.value);
    }

    const result_fact = types.Fact{
        .tag = .value,
        .value = result_value,
        .provenance = types.Provenance.direct(.vdr_computation, args.output_kb_id, args.output_slot_id, kb_mod.currentTimestamp()),
    };
    _ = self.kb_store.factWrite(args.output_kb_id, args.output_slot_id, &result_fact);

    return BuiltinResult.ok(args.output_kb_id, args.output_slot_id, 1);
}

fn dispatchOperational(self: *BuiltinExecutor, builtin_id: i32, args: *const BuiltinArgs) BuiltinResult {
    // Operational builtins — host-only, side effects
    // Grant already checked by command processor before reaching here.
    // Each operational builtin category has its own execution path.
    _ = self;
    // _ = builtin_id;
    _ = args;
    // Stub — implementations for filesystem, compile, execute, lint,
    // network, process will be added when operational layer is built.
    return BuiltinResult.err(types.Status.err(.system, .init_failed, builtin_id));
}

// ============================================================
// Host-side arithmetic — mirrors GPU kernels for small inputs
// ============================================================

fn hostUnary(op: i32, a: types.Q16) types.Q16 {
    return switch (op) {
        0 => types.Q16.fromParts(if (a.v < 0) -a.v else a.v, a.r0), // abs
        1 => types.Q16.fromParts(-a.v, -a.r0), // negate
        2 => types.Q16.fromParts(if (a.v > 0) types.Q16.D else if (a.v < 0) -types.Q16.D else 0, 0), // sign
        10 => types.Q16.mul(a, a), // square
        11 => types.Q16.fromParts(a.v * 2, a.r0 * 2), // double
        12 => types.Q16.fromParts(@divTrunc(a.v, 2), a.r0), // halve
        else => a,
    };
}

fn hostBinary(op: i32, a: types.Q16, b: types.Q16) types.Q16 {
    return switch (op) {
        0 => types.Q16.add(a, b),
        1 => types.Q16.sub(a, b),
        2 => types.Q16.mul(a, b),
        3 => types.Q16.div(a, b),
        5 => if (a.v < b.v) a else b, // min
        6 => if (a.v > b.v) a else b, // max
        else => a,
    };
}

// ============================================================
// Category / grant mapping for operational builtins
// ============================================================

fn categoryForOperational(id: i32) BuiltinCategory {
    if (id < 412) return .op_filesystem;
    if (id < 420) return .op_compile;
    if (id < 428) return .op_execute;
    if (id < 436) return .op_lint;
    if (id < 444) return .op_network;
    return .op_process;
}

fn grantForOperational(id: i32) i8 {
    if (id < 412) return @intFromEnum(types.GrantClass.filesystem);
    if (id < 420) return @intFromEnum(types.GrantClass.compile);
    if (id < 428) return @intFromEnum(types.GrantClass.execute);
    if (id < 436) return @intFromEnum(types.GrantClass.lint);
    if (id < 444) return @intFromEnum(types.GrantClass.network);
    return @intFromEnum(types.GrantClass.process);
}
