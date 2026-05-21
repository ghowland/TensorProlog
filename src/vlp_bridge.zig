// ============================================================
// vlp_bridge.zig — CUDA Driver API backend.
// One kernel module. One function. 15 device buffers.
// OpCode in params selects operation.
// Sole GPU interface — no other module touches CUDA.
// ============================================================

const std = @import("std");
const shared = @import("vlp_gpu_shared");
const types = @import("vlp_types");
const gpu_params = @import("vlp_gpu_params");
const mem = @import("vlp_device_memory");

// ============================================================
// CUDA Driver API types
// ============================================================

pub const CUdevice = i32;
pub const CUcontext = ?*anyopaque;
pub const CUmodule = ?*anyopaque;
pub const CUfunction = ?*anyopaque;
pub const CUdeviceptr = u64; // device pointer — 64-bit unsigned
pub const CUstream = ?*anyopaque;
pub const CUresult = i32;

// Success code
const CUDA_SUCCESS: CUresult = 0;

// ============================================================
// CUDA Driver API extern declarations
// ============================================================

extern "cuda" fn cuInit(flags: u32) CUresult;
extern "cuda" fn cuDeviceGet(device: *CUdevice, ordinal: i32) CUresult;
extern "cuda" fn cuDeviceGetCount(count: *i32) CUresult;
extern "cuda" fn cuDeviceGetName(name: [*]u8, len: i32, dev: CUdevice) CUresult;
extern "cuda" fn cuDeviceTotalMem_v2(bytes: *usize, dev: CUdevice) CUresult;
extern "cuda" fn cuDeviceGetAttribute(pi: *i32, attrib: i32, dev: CUdevice) CUresult;
extern "cuda" fn cuCtxCreate_v2(pctx: *CUcontext, flags: u32, dev: CUdevice) CUresult;
extern "cuda" fn cuCtxDestroy_v2(ctx: CUcontext) CUresult;
extern "cuda" fn cuCtxSynchronize() CUresult;
extern "cuda" fn cuModuleLoadData(module: *CUmodule, image: [*]const u8) CUresult;
extern "cuda" fn cuModuleUnload(module: CUmodule) CUresult;
extern "cuda" fn cuModuleGetFunction(hfunc: *CUfunction, hmod: CUmodule, name: [*:0]const u8) CUresult;
extern "cuda" fn cuMemAlloc_v2(dptr: *CUdeviceptr, bytesize: usize) CUresult;
extern "cuda" fn cuMemFree_v2(dptr: CUdeviceptr) CUresult;
extern "cuda" fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: [*]const u8, byteCount: usize) CUresult;
extern "cuda" fn cuMemcpyDtoH_v2(dst: [*]u8, src: CUdeviceptr, byteCount: usize) CUresult;
extern "cuda" fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, byteCount: usize) CUresult;
extern "cuda" fn cuMemsetD32_v2(dptr: CUdeviceptr, ui: u32, n: usize) CUresult;
extern "cuda" fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: u32,
    gridDimY: u32,
    gridDimZ: u32,
    blockDimX: u32,
    blockDimY: u32,
    blockDimZ: u32,
    sharedMemBytes: u32,
    hStream: CUstream,
    kernelParams: ?[*]?*anyopaque,
    extra: ?[*]?*anyopaque,
) CUresult;

// CUDA device attributes we query
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 1;
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: i32 = 5;
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: i32 = 6;
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: i32 = 7;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: i32 = 2;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: i32 = 3;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: i32 = 4;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

// ============================================================
// Embedded PTX kernel — baked in at compile time
// ============================================================

const kernel_ptx = @embedFile("vlp_kernel_ptx");

// ============================================================
// Configuration
// ============================================================

pub const BridgeConfig = struct {
    sizing: mem.SizingConfig,
    enable_validation: bool = false,
    preferred_device_index: i32 = -1,
    force_host_visible_memory: bool = false,
};

// ============================================================
// Dispatch request — what the host passes per GPU call
// ============================================================

pub const DispatchRequest = struct {
    params: gpu_params.ParamsBuffer,
    group_count_x: i32,
    group_count_y: i32 = 1,
    group_count_z: i32 = 1,
};

// ============================================================
// Operation type — for shouldUseGpu decision
// ============================================================

pub const OperationType = enum(i32) {
    llm_forward = 0,
    fact_scan = 1,
    unification = 2,
    rule_match = 3,
    builtin_array = 4,
    text_grammar = 5,
    access_check = 6,
    sampling = 7,
};

const GPU_THRESHOLD_FACT_SCAN: i32 = 256;
const GPU_THRESHOLD_UNIFICATION: i32 = 32;
const GPU_THRESHOLD_RULE_MATCH: i32 = 64;
const GPU_THRESHOLD_BUILTIN: i32 = 512;
const GPU_THRESHOLD_SAMPLING_VOCAB: i32 = 256 * 1024;

// ============================================================
// Buffer target — names for the device allocations
// ============================================================

pub const BufferTarget = enum(i32) {
    // Set 0 — Model
    embedding_table = 0,
    layer_weights = 1,
    lm_head = 2,
    ln_params = 3,
    // Set 1 — KB Data
    kb_store = 4,
    fact_store = 5,
    rule_store = 6,
    term_store = 7,
    live_state = 8,
    // Set 2 — Scratch
    scratch_a = 9,
    scratch_b = 10,
    kv_cache = 11,
    // Set 3 — Control
    params = 12,
    status = 13,
    result_counts = 14,
};

const BUFFER_COUNT = 15;

// ============================================================
// Device properties — queried at init, cached
// ============================================================

pub const DeviceProperties = struct {
    device_name: [256]u8 = [_]u8{0} ** 256,
    device_name_len: i32 = 0,
    max_compute_shared_memory: i32 = 0,
    max_compute_workgroup_invocations: i32 = 0,
    max_compute_workgroup_count: [3]i32 = .{ 0, 0, 0 },
    max_compute_workgroup_size: [3]i32 = .{ 0, 0, 0 },
    max_storage_buffer_range: i64 = 0,
    max_uniform_buffer_range: i32 = 0,
    host_visible_memory_available: bool = false,
    host_coherent_memory_available: bool = false,
    total_device_memory: i64 = 0,
    compute_queue_family: u32 = 0,
    compute_queue_count: u32 = 0,
    supports_int64: bool = false,
};

// ============================================================
// Params buffer — i32 array uploaded to device before each dispatch
// ============================================================

pub const ParamsBuffer = [shared.PARAMS_INTS]i32;

// ============================================================
// Bridge — the single GPU interface
// ============================================================

pub const Bridge = struct {
    allocator: std.mem.Allocator,

    // CUDA handles
    device: CUdevice = 0,
    context: CUcontext = null,
    module: CUmodule = null,
    kernel_func: CUfunction = null,

    // Device info
    properties: DeviceProperties = .{},

    // Device memory — one allocation per logical buffer
    device_ptrs: [BUFFER_COUNT]CUdeviceptr = .{0} ** BUFFER_COUNT,
    buffer_sizes: [BUFFER_COUNT]i64 = .{0} ** BUFFER_COUNT,

    // Host-side params staging — always host memory, copied to device before dispatch
    params_host: ParamsBuffer = .{0} ** shared.PARAMS_INTS,

    // Host-side readback buffers for status and result_counts
    // Small, frequently read — cached host-side after each dispatch
    status_cache: [256]i32 = .{0} ** 256,
    result_counts_cache: [shared.MAX_RESULT_SLOTS]i32 = .{0} ** shared.MAX_RESULT_SLOTS,

    // Layout
    layout: mem.DeviceMemoryLayout = undefined,

    // State
    config: BridgeConfig = undefined,
    initialized: bool = false,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(allocator: std.mem.Allocator, config: *const BridgeConfig) Bridge {
    var bridge = Bridge{ .allocator = allocator, .config = config.* };

    // 1. Initialize CUDA driver
    if (cuInit(0) != CUDA_SUCCESS) return bridge;

    // 2. Select device
    var device_count: i32 = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS) return bridge;
    if (device_count == 0) return bridge;

    const ordinal = if (config.preferred_device_index >= 0 and config.preferred_device_index < device_count)
        config.preferred_device_index
    else
        0;

    if (cuDeviceGet(&bridge.device, ordinal) != CUDA_SUCCESS) return bridge;

    // 3. Query device properties
    queryDeviceProperties(&bridge);

    // 4. Create context
    if (cuCtxCreate_v2(&bridge.context, 0, bridge.device) != CUDA_SUCCESS) return bridge;

    // 5. Load PTX module
    if (cuModuleLoadData(&bridge.module, kernel_ptx.ptr) != CUDA_SUCCESS) {
        cuCtxDestroy_v2(bridge.context);
        return bridge;
    }

    // 6. Get kernel function handle
    // Zig mangles the name — look for the mangled form.
    // If this fails, try common mangled patterns.
    if (cuModuleGetFunction(&bridge.kernel_func, bridge.module, "vlp_kernel_$_main") != CUDA_SUCCESS) {
        _ = cuModuleUnload(bridge.module);
        _ = cuCtxDestroy_v2(bridge.context);
        return bridge;
    }

    // 7. Compute memory layout
    bridge.layout = mem.computeLayout(&config.sizing);

    // 8. Allocate device buffers
    if (!allocateBuffers(&bridge)) {
        freeBuffers(&bridge);
        cuModuleUnload(bridge.module);
        cuCtxDestroy_v2(bridge.context);
        return bridge;
    }

    bridge.initialized = true;
    return bridge;
}

pub fn deinit(self: *Bridge) void {
    if (!self.initialized) return;

    _ = cuCtxSynchronize();
    freeBuffers(self);

    if (self.module != null) {
        _ = cuModuleUnload(self.module);
        self.module = null;
    }
    if (self.context != null) {
        _ = cuCtxDestroy_v2(self.context);
        self.context = null;
    }

    self.kernel_func = null;
    self.initialized = false;
}

// ============================================================
// Buffer allocation
// ============================================================

fn allocateBuffers(bridge: *Bridge) bool {
    const cap = mem.computeCapacity(&bridge.config.sizing);

    // Map BufferTarget to byte sizes
    const sizes = [BUFFER_COUNT]i64{
        cap.model_bytes / 4, // embedding — approximate split
        cap.model_bytes / 2, // layer_weights — bulk of model
        cap.model_bytes / 4, // lm_head
        1 * 1024 * 1024, // ln_params — small
        cap.kb_store_bytes,
        cap.fact_store_bytes,
        cap.rule_store_bytes,
        cap.term_store_bytes,
        cap.live_state_bytes,
        cap.scratch_bytes / 2, // scratch_a
        cap.scratch_bytes / 2, // scratch_b
        cap.kv_cache_bytes,
        shared.PARAMS_INTS * 4, // params uniform
        shared.MAX_STATUS_ENTRIES * 4, // status
        shared.MAX_RESULT_SLOTS * 4, // result_counts
    };

    for (0..BUFFER_COUNT) |i| {
        const sz: usize = @intCast(@max(sizes[i], 4)); // minimum 4 bytes
        bridge.buffer_sizes[i] = @intCast(sz);
        if (cuMemAlloc_v2(&bridge.device_ptrs[i], sz) != CUDA_SUCCESS) {
            // Free already-allocated buffers
            for (0..i) |j| {
                _ = cuMemFree_v2(bridge.device_ptrs[j]);
                bridge.device_ptrs[j] = 0;
            }
            return false;
        }
        // Zero-initialize
        _ = cuMemsetD32_v2(bridge.device_ptrs[i], 0, sz / 4);
    }

    return true;
}

fn freeBuffers(bridge: *Bridge) void {
    for (0..BUFFER_COUNT) |i| {
        if (bridge.device_ptrs[i] != 0) {
            _ = cuMemFree_v2(bridge.device_ptrs[i]);
            bridge.device_ptrs[i] = 0;
        }
    }
}

// ============================================================
// Dispatch — the core GPU call
// ============================================================

pub fn dispatch(self: *Bridge, request: *const DispatchRequest) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);

    // 1. Upload params to device params buffer
    @memcpy(&self.params_host, &request.params);
    const params_status = self.uploadToBuffer(.params, 0, std.mem.asBytes(&self.params_host));
    if (params_status.isErr()) return params_status;

    // 2. Build kernel argument array
    // The kernel expects 15 pointer arguments — one per buffer.
    // CUDA passes kernel args as array of pointers-to-args.
    var arg_ptrs: [BUFFER_COUNT]CUdeviceptr = undefined;
    for (0..BUFFER_COUNT) |i| {
        arg_ptrs[i] = self.device_ptrs[i];
    }

    // Array of pointers to each argument
    var kernel_args: [BUFFER_COUNT]?*anyopaque = undefined;
    for (0..BUFFER_COUNT) |i| {
        kernel_args[i] = @ptrCast(&arg_ptrs[i]);
    }

    // 3. Launch kernel
    const grid_x: u32 = @intCast(@max(request.group_count_x, 1));
    const grid_y: u32 = @intCast(@max(request.group_count_y, 1));
    const grid_z: u32 = @intCast(@max(request.group_count_z, 1));

    // Shared memory: enough for s_i32 + s_i64 + s_idx
    // 256 * (4 + 8 + 4) = 4096 bytes
    const shared_mem: u32 = shared.MAX_WORKGROUP * (4 + 8 + 4);

    const launch_result = cuLaunchKernel(
        self.kernel_func,
        grid_x,
        grid_y,
        grid_z,
        @intCast(shared.MAX_WORKGROUP), // blockDimX = 256
        1, // blockDimY
        1, // blockDimZ
        shared_mem,
        null, // default stream
        &kernel_args,
        null,
    );

    if (launch_result != CUDA_SUCCESS) {
        return types.Status.err(.device, .dispatch_failed, launch_result);
    }

    // 4. Synchronize — wait for kernel to complete
    const sync_result = cuCtxSynchronize();
    if (sync_result != CUDA_SUCCESS) {
        return types.Status.err(.device, .dispatch_failed, sync_result);
    }

    // 5. Read back status buffer (first 256 entries) for error checking
    _ = cuMemcpyDtoH_v2(
        @ptrCast(&self.status_cache),
        self.device_ptrs[@intFromEnum(BufferTarget.status)],
        256 * 4,
    );

    // 6. Read back result_counts
    _ = cuMemcpyDtoH_v2(
        @ptrCast(&self.result_counts_cache),
        self.device_ptrs[@intFromEnum(BufferTarget.result_counts)],
        shared.MAX_RESULT_SLOTS * 4,
    );

    // 7. Check status for kernel-reported errors
    const first_error = checkStatusCache(self, request);
    if (first_error != 0) {
        return types.Status.err(.device, .dispatch_failed, first_error);
    }

    return types.Status.ok();
}

/// Record multiple dispatches sequentially, synchronize once at end.
pub fn dispatchSequence(self: *Bridge, requests: []const DispatchRequest) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);
    if (requests.len == 0) return types.Status.ok();

    for (requests) |*req| {
        // Upload params
        @memcpy(&self.params_host, &req.params);
        _ = cuMemcpyHtoD_v2(
            self.device_ptrs[@intFromEnum(BufferTarget.params)],
            @ptrCast(&self.params_host),
            shared.PARAMS_INTS * 4,
        );

        // Build args
        var arg_ptrs: [BUFFER_COUNT]CUdeviceptr = undefined;
        for (0..BUFFER_COUNT) |i| {
            arg_ptrs[i] = self.device_ptrs[i];
        }
        var kernel_args: [BUFFER_COUNT]?*anyopaque = undefined;
        for (0..BUFFER_COUNT) |i| {
            kernel_args[i] = @ptrCast(&arg_ptrs[i]);
        }

        const shared_mem: u32 = shared.MAX_WORKGROUP * (4 + 8 + 4);

        _ = cuLaunchKernel(
            self.kernel_func,
            @intCast(@max(req.group_count_x, 1)),
            @intCast(@max(req.group_count_y, 1)),
            @intCast(@max(req.group_count_z, 1)),
            @intCast(shared.MAX_WORKGROUP),
            1,
            1,
            shared_mem,
            null,
            &kernel_args,
            null,
        );
    }

    // Single sync at end of sequence
    const sync_result = cuCtxSynchronize();
    if (sync_result != CUDA_SUCCESS) {
        return types.Status.err(.device, .dispatch_failed, sync_result);
    }

    // Read back status and result_counts after final dispatch
    _ = cuMemcpyDtoH_v2(
        @ptrCast(&self.status_cache),
        self.device_ptrs[@intFromEnum(BufferTarget.status)],
        256 * 4,
    );
    _ = cuMemcpyDtoH_v2(
        @ptrCast(&self.result_counts_cache),
        self.device_ptrs[@intFromEnum(BufferTarget.result_counts)],
        shared.MAX_RESULT_SLOTS * 4,
    );

    return types.Status.ok();
}

// ============================================================
// Buffer data transfer
// ============================================================

pub fn uploadToBuffer(self: *Bridge, target: BufferTarget, offset: i64, data: []const u8) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);
    if (data.len == 0) return types.Status.ok();

    const idx = @intFromEnum(target);
    const dst = self.device_ptrs[idx] + @as(u64, @intCast(offset));

    const result = cuMemcpyHtoD_v2(dst, data.ptr, data.len);
    if (result != CUDA_SUCCESS) {
        return types.Status.err(.device, .dispatch_failed, result);
    }
    return types.Status.ok();
}

pub fn downloadFromBuffer(self: *Bridge, source: BufferTarget, offset: i64, dest: []u8) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);
    if (dest.len == 0) return types.Status.ok();

    const idx = @intFromEnum(source);
    const src = self.device_ptrs[idx] + @as(u64, @intCast(offset));

    const result = cuMemcpyDtoH_v2(dest.ptr, src, dest.len);
    if (result != CUDA_SUCCESS) {
        return types.Status.err(.device, .dispatch_failed, result);
    }
    return types.Status.ok();
}

pub fn copyBufferToBuffer(self: *Bridge, src: BufferTarget, src_offset: i64, dst: BufferTarget, dst_offset: i64, size: i64) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);
    if (size <= 0) return types.Status.ok();

    const src_ptr = self.device_ptrs[@intFromEnum(src)] + @as(u64, @intCast(src_offset));
    const dst_ptr = self.device_ptrs[@intFromEnum(dst)] + @as(u64, @intCast(dst_offset));

    const result = cuMemcpyDtoD_v2(dst_ptr, src_ptr, @intCast(size));
    if (result != CUDA_SUCCESS) {
        return types.Status.err(.device, .dispatch_failed, result);
    }
    return types.Status.ok();
}

pub fn fillBuffer(self: *Bridge, target: BufferTarget, offset: i64, size: i64, value: u32) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);
    if (size <= 0) return types.Status.ok();

    const dst = self.device_ptrs[@intFromEnum(target)] + @as(u64, @intCast(offset));
    const n_ints: usize = @intCast(@divTrunc(size, 4));

    const result = cuMemsetD32_v2(dst, value, n_ints);
    if (result != CUDA_SUCCESS) {
        return types.Status.err(.device, .dispatch_failed, result);
    }
    return types.Status.ok();
}

// ============================================================
// Status / result readback
// ============================================================

pub fn readStatus(self: *Bridge, invocation_index: i32) i32 {
    if (invocation_index < 0 or invocation_index >= 256) return 0;
    return self.status_cache[@intCast(invocation_index)];
}

pub fn readResultCount(self: *Bridge, slot: i32) i32 {
    if (slot < 0 or slot >= shared.MAX_RESULT_SLOTS) return 0;
    return self.result_counts_cache[@intCast(slot)];
}

pub fn resetStatusBuffer(self: *Bridge) types.Status {
    return self.fillBuffer(.status, 0, self.buffer_sizes[@intFromEnum(BufferTarget.status)], 0);
}

pub fn resetResultCounts(self: *Bridge) types.Status {
    @memset(&self.result_counts_cache, 0);
    return self.fillBuffer(.result_counts, 0, self.buffer_sizes[@intFromEnum(BufferTarget.result_counts)], 0);
}

fn checkStatusCache(self: *Bridge, request: *const DispatchRequest) i32 {
    const n = request.group_count_x * request.group_count_y * request.group_count_z;
    const check_limit: usize = @intCast(@min(n, 256));
    for (self.status_cache[0..check_limit]) |s| {
        if (s != 0) return s;
    }
    return 0;
}

// ============================================================
// GPU vs host decision
// ============================================================

pub fn shouldUseGpu(self: *Bridge, op: OperationType, data_size: i32) bool {
    if (!self.initialized) return false;
    return switch (op) {
        .llm_forward => true,
        .fact_scan => data_size > GPU_THRESHOLD_FACT_SCAN,
        .unification => data_size > GPU_THRESHOLD_UNIFICATION,
        .rule_match => data_size > GPU_THRESHOLD_RULE_MATCH,
        .builtin_array => data_size > GPU_THRESHOLD_BUILTIN,
        .text_grammar => false,
        .access_check => false,
        .sampling => data_size > GPU_THRESHOLD_SAMPLING_VOCAB,
    };
}

// ============================================================
// Convenience: typed upload/download
// ============================================================

pub fn uploadInts(self: *Bridge, target: BufferTarget, offset_ints: i32, data: []const i32) types.Status {
    const bytes = std.mem.sliceAsBytes(data);
    return self.uploadToBuffer(target, @as(i64, offset_ints) * 4, bytes);
}

pub fn downloadInts(self: *Bridge, target: BufferTarget, offset_ints: i32, out: []i32) types.Status {
    const bytes = std.mem.sliceAsBytes(out);
    return self.downloadFromBuffer(target, @as(i64, offset_ints) * 4, bytes);
}

pub fn uploadFact(self: *Bridge, fact_index: i32, fact: *const types.Fact) types.Status {
    const ints = fact.toInts();
    return self.uploadInts(.fact_store, fact_index * shared.FACT_INTS, &ints);
}

pub fn downloadFact(self: *Bridge, fact_index: i32) ?types.Fact {
    var ints: [shared.FACT_INTS]i32 = undefined;
    const status = self.downloadInts(.fact_store, fact_index * shared.FACT_INTS, &ints);
    if (status.isErr()) return null;
    return types.Fact.fromInts(ints);
}

pub fn uploadTerm(self: *Bridge, term_index: i32, term: *const types.Term) types.Status {
    const ints = term.toInts();
    return self.uploadInts(.term_store, term_index * shared.TERM_INTS, &ints);
}

pub fn downloadTerm(self: *Bridge, term_index: i32) ?types.Term {
    var ints: [shared.TERM_INTS]i32 = undefined;
    const status = self.downloadInts(.term_store, term_index * shared.TERM_INTS, &ints);
    if (status.isErr()) return null;
    return types.Term.fromInts(ints);
}

pub fn uploadRule(self: *Bridge, rule_index: i32, rule: *const types.Rule) types.Status {
    const ints = rule.toInts();
    return self.uploadInts(.rule_store, rule_index * shared.RULE_INTS, &ints);
}

pub fn downloadRule(self: *Bridge, rule_index: i32) ?types.Rule {
    var ints: [shared.RULE_INTS]i32 = undefined;
    const status = self.downloadInts(.rule_store, rule_index * shared.RULE_INTS, &ints);
    if (status.isErr()) return null;
    return types.Rule.fromInts(ints);
}

pub fn uploadKb(self: *Bridge, kb_id: i32, data: []const i32) types.Status {
    if (data.len != shared.KB_STRUCT_INTS) return types.Status.err(.kb, .kb_not_found, kb_id);
    return self.uploadInts(.kb_store, kb_id * shared.KB_STRUCT_INTS, data);
}

pub fn downloadKb(self: *Bridge, kb_id: i32, out: []i32) types.Status {
    if (out.len != shared.KB_STRUCT_INTS) return types.Status.err(.kb, .kb_not_found, kb_id);
    return self.downloadInts(.kb_store, kb_id * shared.KB_STRUCT_INTS, out);
}

// ============================================================
// Convenience: dispatch helpers
// ============================================================

pub fn workgroups(n: i32) i32 {
    return @divTrunc(n + shared.MAX_WORKGROUP - 1, shared.MAX_WORKGROUP);
}

pub fn dispatch1D(self: *Bridge, p: ParamsBuffer, n_elements: i32) types.Status {
    return self.dispatch(&.{
        .params = p,
        .group_count_x = workgroups(n_elements),
    });
}

pub fn dispatch2D(self: *Bridge, p: ParamsBuffer, x: i32, y: i32) types.Status {
    return self.dispatch(&.{
        .params = p,
        .group_count_x = x,
        .group_count_y = y,
    });
}

pub fn dispatchClean(self: *Bridge, request: *const DispatchRequest) types.Status {
    var status = self.resetStatusBuffer();
    if (status.isErr()) return status;
    status = self.resetResultCounts();
    if (status.isErr()) return status;
    return self.dispatch(request);
}

// ============================================================
// Scratch slice read — typed download from scratch buffers
// ============================================================

pub fn readScratchSlice(self: *Bridge, comptime T: type, target: BufferTarget, offset_elements: i32, count: i32, out: []T) types.Status {
    const byte_offset = @as(i64, offset_elements) * @sizeOf(T);
    const byte_count = @as(usize, @intCast(count)) * @sizeOf(T);
    const dest: []u8 = @as([*]u8, @ptrCast(out.ptr))[0..byte_count];
    return self.downloadFromBuffer(target, byte_offset, dest);
}

// ============================================================
// Device queries
// ============================================================

pub fn deviceName(self: *Bridge) []const u8 {
    return self.properties.device_name[0..@intCast(self.properties.device_name_len)];
}

pub fn totalDeviceMemory(self: *Bridge) i64 {
    return self.properties.total_device_memory;
}

pub fn maxWorkgroupSize(self: *Bridge) i32 {
    return self.properties.max_compute_workgroup_invocations;
}

pub fn sharedMemorySize(self: *Bridge) i32 {
    return self.properties.max_compute_shared_memory;
}

pub fn supportsInt64(self: *Bridge) bool {
    return self.properties.supports_int64;
}

pub fn bufferSize(self: *Bridge, target: BufferTarget) i64 {
    return self.buffer_sizes[@intFromEnum(target)];
}

// ============================================================
// Descriptor set updates — no-ops in CUDA backend
// Kept for interface compatibility with host modules that call them.
// ============================================================

pub fn updateModelDescriptors(self: *Bridge) types.Status {
    _ = self;
    return types.Status.ok();
}

pub fn updateKbDescriptors(self: *Bridge, session_kb_offset: i64, session_fact_offset: i64) types.Status {
    _ = self;
    _ = session_kb_offset;
    _ = session_fact_offset;
    return types.Status.ok();
}

pub fn updateScratchDescriptors(self: *Bridge) types.Status {
    _ = self;
    return types.Status.ok();
}

pub fn updateControlDescriptors(self: *Bridge) types.Status {
    _ = self;
    return types.Status.ok();
}

// ============================================================
// Device property query
// ============================================================

fn queryDeviceProperties(bridge: *Bridge) void {
    // Device name
    _ = cuDeviceGetName(&bridge.properties.device_name, 256, bridge.device);
    bridge.properties.device_name_len = blk: {
        for (bridge.properties.device_name, 0..) |c, i| {
            if (c == 0) break :blk @intCast(i);
        }
        break :blk 256;
    };

    // Total memory
    var total_mem: usize = 0;
    _ = cuDeviceTotalMem_v2(&total_mem, bridge.device);
    bridge.properties.total_device_memory = @intCast(total_mem);

    // Shared memory per block
    var shared_mem: i32 = 0;
    _ = cuDeviceGetAttribute(&shared_mem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, bridge.device);
    bridge.properties.max_compute_shared_memory = shared_mem;

    // Max threads per block
    var max_threads: i32 = 0;
    _ = cuDeviceGetAttribute(&max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, bridge.device);
    bridge.properties.max_compute_workgroup_invocations = max_threads;

    // Max grid dimensions
    _ = cuDeviceGetAttribute(&bridge.properties.max_compute_workgroup_count[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, bridge.device);
    _ = cuDeviceGetAttribute(&bridge.properties.max_compute_workgroup_count[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, bridge.device);
    _ = cuDeviceGetAttribute(&bridge.properties.max_compute_workgroup_count[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, bridge.device);

    // Max block dimensions
    _ = cuDeviceGetAttribute(&bridge.properties.max_compute_workgroup_size[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, bridge.device);
    _ = cuDeviceGetAttribute(&bridge.properties.max_compute_workgroup_size[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, bridge.device);
    _ = cuDeviceGetAttribute(&bridge.properties.max_compute_workgroup_size[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, bridge.device);

    // Compute capability (int64 supported on all modern GPUs, sm_50+)
    var cc_major: i32 = 0;
    _ = cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, bridge.device);
    bridge.properties.supports_int64 = cc_major >= 5;

    // CUDA always has "host visible" memory via cuMemcpyHtoD/DtoH
    bridge.properties.host_visible_memory_available = true;
    bridge.properties.host_coherent_memory_available = false;
}
