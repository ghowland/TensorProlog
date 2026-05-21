// ============================================================
// vlp_bridge.zig — REWRITTEN for single-kernel architecture.
// One pipeline. One shader module. One descriptor set layout.
// OpCode in params uniform selects operation.
// Sole Vulkan interface — no other module touches Vulkan.
// ============================================================

const std = @import("std");
const shared = @import("vlp_gpu_shared");
const types = @import("vlp_types");
const gpu_params = @import("vlp_gpu_params");
const mem = @import("vlp_device_memory");

// ============================================================
// Vulkan handle types — opaque until vulkan-zig is wired in.
// Replace with vk.* types when integrating vulkan-zig.
// ============================================================

pub const VkInstance = ?*anyopaque;
pub const VkPhysicalDevice = ?*anyopaque;
pub const VkDevice = ?*anyopaque;
pub const VkQueue = ?*anyopaque;
pub const VkCommandPool = ?*anyopaque;
pub const VkCommandBuffer = ?*anyopaque;
pub const VkPipeline = ?*anyopaque;
pub const VkPipelineLayout = ?*anyopaque;
pub const VkDescriptorPool = ?*anyopaque;
pub const VkDescriptorSetLayout = ?*anyopaque;
pub const VkDescriptorSet = ?*anyopaque;
pub const VkBuffer = ?*anyopaque;
pub const VkDeviceMemory = ?*anyopaque;
pub const VkFence = ?*anyopaque;
pub const VkShaderModule = ?*anyopaque;

// ============================================================
// Embedded SPIR-V kernel — baked in at compile time
// ============================================================

const kernel_spv align(@alignOf(u32)) = @embedFile("vlp_kernel_spv").*;

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
// Buffer target — names for the storage buffers
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
// Bridge — the single GPU interface
// ============================================================

pub const Bridge = struct {
    allocator: std.mem.Allocator,

    // Vulkan core handles
    instance: VkInstance = null,
    physical_device: VkPhysicalDevice = null,
    device: VkDevice = null,
    compute_queue: VkQueue = null,
    command_pool: VkCommandPool = null,
    dispatch_cmd: VkCommandBuffer = null,
    dispatch_fence: VkFence = null,

    // Device info
    properties: DeviceProperties = .{},

    // THE pipeline — single kernel, single layout
    shader_module: VkShaderModule = null,
    pipeline_layout: VkPipelineLayout = null,
    pipeline: VkPipeline = null,

    // Descriptor infrastructure
    descriptor_pool: VkDescriptorPool = null,
    set_layouts: [4]VkDescriptorSetLayout = .{ null, null, null, null },
    active_sets: [4]VkDescriptorSet = .{ null, null, null, null },

    // Storage buffers — one per logical buffer
    buffers: [15]VkBuffer = .{null} ** 15,
    buffer_sizes: [15]i64 = .{0} ** 15,

    // Device memory — grouped by update frequency
    model_memory: VkDeviceMemory = null,
    kb_data_memory: VkDeviceMemory = null,
    scratch_memory: VkDeviceMemory = null,
    control_memory: VkDeviceMemory = null,

    // Host-mapped pointers (null if not host-visible)
    mapped: [15]?[*]u8 = .{null} ** 15,

    // Staging buffers for non-host-visible transfers
    staging_upload: VkBuffer = null,
    staging_upload_memory: VkDeviceMemory = null,
    staging_upload_mapped: ?[*]u8 = null,
    staging_upload_size: i64 = 0,
    staging_download: VkBuffer = null,
    staging_download_memory: VkDeviceMemory = null,
    staging_download_mapped: ?[*]u8 = null,
    staging_download_size: i64 = 0,

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

    // Implementation sequence:
    //
    // 1. Create Vulkan instance
    //    - Enable validation layers if config.enable_validation
    //    - API version 1.2 (required for PhysicalStorageBuffer, Int64)
    //
    // 2. Select physical device
    //    - If config.preferred_device_index >= 0: use that index
    //    - Else: pick first device with compute queue family
    //    - Query and cache DeviceProperties
    //    - Verify supports_int64 (mandatory for Q16 multiply accumulators)
    //
    // 3. Create logical device + compute queue
    //    - Enable Int64 feature
    //    - One compute queue from compute queue family
    //
    // 4. Create command pool + pre-allocate one command buffer
    //
    // 5. Compute memory layout
    bridge.layout = mem.computeLayout(&config.sizing);
    //
    // 6. Create storage buffers (15 total)
    //    - Set 0: embedding, layer_weights, lm_head, ln_params
    //    - Set 1: kb_store, fact_store, rule_store, term_store, live_state
    //    - Set 2: scratch_a, scratch_b, kv_cache
    //    - Set 3: params (uniform), status, result_counts
    //    Buffer sizes from layout. Params buffer = 256 bytes (uniform).
    //
    // 7. Allocate device memory, bind buffers
    //    - Model: device-local (read-only after load)
    //    - KB data: device-local or host-visible (for mapped access)
    //    - Scratch: device-local
    //    - Control: host-visible + host-coherent (for direct param/status access)
    //
    // 8. Attempt persistent mapping of host-visible buffers
    //    - Params, status, result_counts: always mapped (host-visible)
    //    - KB/fact/rule/term/text: mapped if host-visible, else use staging
    //    - Scratch: typically not mapped (device-local)
    //
    // 9. Allocate staging buffers if needed
    //    - Upload staging: 16 MB host-visible
    //    - Download staging: 16 MB host-visible
    //
    // 10. Create shader module from embedded SPIR-V
    //     const spv_ptr: [*]const u32 = @ptrCast(&kernel_spv);
    //     const spv_size: usize = kernel_spv.len;
    //     createShaderModule(spv_ptr, spv_size)
    //
    // 11. Create descriptor set layouts (4 sets)
    //     Set 0: 4 storage buffer bindings (read-only)
    //     Set 1: 5 storage buffer bindings (read-write)
    //     Set 2: 3 storage buffer bindings (read-write)
    //     Set 3: 1 uniform buffer + 2 storage buffer bindings
    //
    // 12. Create pipeline layout from the 4 set layouts
    //
    // 13. Create THE compute pipeline
    //     One shader module, entry point "main", one pipeline.
    //
    // 14. Create descriptor pool, allocate 4 descriptor sets
    //
    // 15. Write initial descriptor sets (bind buffers to bindings)
    //
    // 16. Create fence (unsignaled)

    bridge.initialized = true;
    return bridge;
}

pub fn deinit(self: *Bridge) void {
    // Destroy in reverse order:
    // fence, descriptor pool, pipeline, pipeline layout,
    // shader module, staging buffers, storage buffers,
    // device memory, command pool, device, instance
    self.initialized = false;
}

// ============================================================
// Dispatch — the core GPU call
// ============================================================

pub fn dispatch(self: *Bridge, request: *const DispatchRequest) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);

    // 1. Upload params to uniform buffer
    const param_bytes = std.mem.asBytes(&request.params);
    const up_status = self.uploadToBuffer(.params, 0, param_bytes);
    if (up_status.isErr()) return up_status;

    // 2. Begin command buffer (reset + begin)
    //    vkResetCommandBuffer(dispatch_cmd, 0)
    //    vkBeginCommandBuffer(dispatch_cmd, {.flags = ONE_TIME_SUBMIT})

    // 3. Bind pipeline
    //    vkCmdBindPipeline(dispatch_cmd, COMPUTE, pipeline)

    // 4. Bind descriptor sets
    //    vkCmdBindDescriptorSets(dispatch_cmd, COMPUTE, pipeline_layout,
    //        0, 4, active_sets, 0, null)

    // 5. Dispatch
    //    vkCmdDispatch(dispatch_cmd, group_count_x, group_count_y, group_count_z)

    // 6. Pipeline barrier: compute write → host read
    //    memory barrier: SHADER_WRITE → HOST_READ
    //    stage: COMPUTE → HOST

    // 7. End command buffer
    //    vkEndCommandBuffer(dispatch_cmd)

    // 8. Reset fence
    //    vkResetFences(device, 1, &dispatch_fence)

    // 9. Submit
    //    vkQueueSubmit(compute_queue, 1, &submit_info, dispatch_fence)

    // 10. Wait
    //     vkWaitForFences(device, 1, &dispatch_fence, TRUE, UINT64_MAX)

    // 11. Check status buffer for kernel-reported errors
    const first_error = self.checkStatusBuffer(request);
    if (first_error != 0) {
        return types.Status.err(.device, .dispatch_failed, first_error);
    }

    return types.Status.ok();
}

/// Record multiple dispatches into one command buffer, one submit, one fence.
/// Used for LLM forward pass (12+ dispatches per layer).
pub fn dispatchSequence(self: *Bridge, requests: []const DispatchRequest) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);
    if (requests.len == 0) return types.Status.ok();

    // 1. Reset + begin command buffer

    // 2. For each request:
    for (requests) |*req| {
        // a. Upload params
        const param_bytes = std.mem.asBytes(&req.params);
        _ = self.writeToMapped(.params, 0, param_bytes);

        // b. Bind pipeline (same every time, but Vulkan requires it
        //    after descriptor set changes if params buffer content changed)
        //    vkCmdBindPipeline(dispatch_cmd, COMPUTE, pipeline)

        // c. Bind descriptor sets
        //    vkCmdBindDescriptorSets(...)

        // d. Dispatch
        //    vkCmdDispatch(dispatch_cmd, req.group_count_x, y, z)

        // e. Pipeline barrier: compute write → compute read
        //    (between dependent dispatches within the sequence)
        //    memory barrier: SHADER_WRITE → SHADER_READ
        //    stage: COMPUTE → COMPUTE
    }

    // 3. Final barrier: compute write → host read
    // 4. End command buffer
    // 5. Reset fence, submit, wait

    return types.Status.ok();
}

// ============================================================
// Buffer data transfer
// ============================================================

pub fn uploadToBuffer(self: *Bridge, target: BufferTarget, offset: i64, data: []const u8) types.Status {
    const idx = @intFromEnum(target);

    // Fast path: mapped memory
    if (self.mapped[idx]) |ptr| {
        _ = ptr;
        return self.writeToMapped(target, offset, data);
    }

    // Slow path: staging buffer
    if (self.staging_upload_mapped == null) return types.Status.err(.device, .dispatch_failed, -1);
    if (data.len > @as(usize, @intCast(self.staging_upload_size))) return types.Status.err(.device, .device_out_of_memory, -2);

    // Copy to staging
    const staging_ptr = self.staging_upload_mapped.?;
    @memcpy(staging_ptr[0..data.len], data);

    // Record copy command: staging → target buffer at offset
    // vkCmdCopyBuffer(dispatch_cmd, staging_upload, buffers[idx],
    //     1, &VkBufferCopy{ .srcOffset=0, .dstOffset=offset, .size=data.len })
    // Submit + fence

    return types.Status.ok();
}

pub fn downloadFromBuffer(self: *Bridge, source: BufferTarget, offset: i64, dest: []u8) types.Status {
    const idx = @intFromEnum(source);

    // Fast path: mapped memory
    if (self.mapped[idx]) |ptr| {
        const src = ptr[@intCast(offset)..@intCast(offset + @as(i64, @intCast(dest.len)))];
        @memcpy(dest, src);
        return types.Status.ok();
    }

    // Slow path: staging buffer
    if (self.staging_download_mapped == null) return types.Status.err(.device, .dispatch_failed, -1);
    if (dest.len > @as(usize, @intCast(self.staging_download_size))) return types.Status.err(.device, .device_out_of_memory, -2);

    // Record copy command: source buffer at offset → staging
    // vkCmdCopyBuffer(dispatch_cmd, buffers[idx], staging_download,
    //     1, &VkBufferCopy{ .srcOffset=offset, .dstOffset=0, .size=dest.len })
    // Submit + fence

    // Copy from staging to dest
    const staging_ptr = self.staging_download_mapped.?;
    @memcpy(dest, staging_ptr[0..dest.len]);

    return types.Status.ok();
}

pub fn copyBufferToBuffer(self: *Bridge, src: BufferTarget, src_offset: i64, dst: BufferTarget, dst_offset: i64, size: i64) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);
    _ = src;
    _ = src_offset;
    _ = dst;
    _ = dst_offset;
    _ = size;
    // Record vkCmdCopyBuffer between the two buffers
    // Submit + fence
    return types.Status.ok();
}

pub fn fillBuffer(self: *Bridge, target: BufferTarget, offset: i64, size: i64, value: u32) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);
    _ = target;
    _ = offset;
    _ = size;
    _ = value;
    // vkCmdFillBuffer(dispatch_cmd, buffers[idx], offset, size, value)
    // Submit + fence
    return types.Status.ok();
}

// ============================================================
// Mapped pointer access
// ============================================================

fn writeToMapped(self: *Bridge, target: BufferTarget, offset: i64, data: []const u8) types.Status {
    const idx = @intFromEnum(target);
    const ptr = self.mapped[idx] orelse return types.Status.err(.device, .dispatch_failed, idx);
    const dst = ptr[@intCast(offset)..@intCast(offset + @as(i64, @intCast(data.len)))];
    @memcpy(dst, data);
    return types.Status.ok();
}

pub fn isMapped(self: *Bridge, target: BufferTarget) bool {
    return self.mapped[@intFromEnum(target)] != null;
}

// ============================================================
// Status / result readback
// ============================================================

pub fn readStatus(self: *Bridge, invocation_index: i32) i32 {
    if (self.mapped[@intFromEnum(BufferTarget.status)]) |ptr| {
        const i32_ptr: [*]const i32 = @ptrCast(@alignCast(ptr));
        return i32_ptr[@intCast(invocation_index)];
    }
    var val: i32 = 0;
    _ = self.downloadFromBuffer(.status, @as(i64, invocation_index) * 4, std.mem.asBytes(&val));
    return val;
}

pub fn readResultCount(self: *Bridge, slot: i32) i32 {
    if (self.mapped[@intFromEnum(BufferTarget.result_counts)]) |ptr| {
        const i32_ptr: [*]const i32 = @ptrCast(@alignCast(ptr));
        return i32_ptr[@intCast(slot)];
    }
    var val: i32 = 0;
    _ = self.downloadFromBuffer(.result_counts, @as(i64, slot) * 4, std.mem.asBytes(&val));
    return val;
}

pub fn resetStatusBuffer(self: *Bridge) types.Status {
    return self.fillBuffer(.status, 0, self.buffer_sizes[@intFromEnum(BufferTarget.status)], 0);
}

pub fn resetResultCounts(self: *Bridge) types.Status {
    return self.fillBuffer(.result_counts, 0, self.buffer_sizes[@intFromEnum(BufferTarget.result_counts)], 0);
}

fn checkStatusBuffer(self: *Bridge, request: *const DispatchRequest) i32 {
    // Scan first N entries for non-zero (N = dispatch count)
    const n: i32 = request.group_count_x * request.group_count_y * request.group_count_z;
    // Only check first few entries — full scan too expensive for large dispatches
    const check_limit = @min(n, 256);
    var i: i32 = 0;
    while (i < check_limit) : (i += 1) {
        const s = self.readStatus(i);
        if (s != 0) return s;
    }
    return 0;
}

// ============================================================
// Descriptor set updates
// ============================================================

pub fn updateModelDescriptors(self: *Bridge) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);
    // Write Set 0 descriptor bindings:
    //   binding 0 → buffers[embedding_table]
    //   binding 1 → buffers[layer_weights]
    //   binding 2 → buffers[lm_head]
    //   binding 3 → buffers[ln_params]
    // vkUpdateDescriptorSets with 4 VkWriteDescriptorSet entries
    return types.Status.ok();
}

pub fn updateKbDescriptors(self: *Bridge, session_kb_offset: i64, session_fact_offset: i64) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);
    _ = session_kb_offset;
    _ = session_fact_offset;
    // Write Set 1 descriptor bindings:
    //   binding 0 → buffers[kb_store] offset=session_kb_offset
    //   binding 1 → buffers[fact_store] offset=session_fact_offset
    //   binding 2 → buffers[rule_store]
    //   binding 3 → buffers[term_store]
    //   binding 4 → buffers[live_state]
    // This rebinds for the active session's region within the global buffers.
    return types.Status.ok();
}

pub fn updateScratchDescriptors(self: *Bridge) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);
    // Write Set 2 descriptor bindings:
    //   binding 0 → buffers[scratch_a]
    //   binding 1 → buffers[scratch_b]
    //   binding 2 → buffers[kv_cache]
    return types.Status.ok();
}

pub fn updateControlDescriptors(self: *Bridge) types.Status {
    if (!self.initialized) return types.Status.err(.device, .dispatch_failed, 0);
    // Write Set 3 descriptor bindings:
    //   binding 0 → buffers[params] (uniform buffer)
    //   binding 1 → buffers[status]
    //   binding 2 → buffers[result_counts]
    return types.Status.ok();
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

/// Upload a Fact (as 10 i32s) to fact_store at fact index
pub fn uploadFact(self: *Bridge, fact_index: i32, fact: *const types.Fact) types.Status {
    const ints = fact.toInts();
    return self.uploadInts(.fact_store, fact_index * shared.FACT_INTS, &ints);
}

/// Download a Fact from fact_store at fact index
pub fn downloadFact(self: *Bridge, fact_index: i32) ?types.Fact {
    var ints: [10]i32 = undefined;
    const status = self.downloadInts(.fact_store, fact_index * shared.FACT_INTS, &ints);
    if (status.isErr()) return null;
    return types.Fact.fromInts(ints);
}

/// Upload a Term (as 6 i32s) to term_store at term index
pub fn uploadTerm(self: *Bridge, term_index: i32, term: *const types.Term) types.Status {
    const ints = term.toInts();
    return self.uploadInts(.term_store, term_index * shared.TERM_INTS, &ints);
}

/// Download a Term from term_store at term index
pub fn downloadTerm(self: *Bridge, term_index: i32) ?types.Term {
    var ints: [6]i32 = undefined;
    const status = self.downloadInts(.term_store, term_index * shared.TERM_INTS, &ints);
    if (status.isErr()) return null;
    return types.Term.fromInts(ints);
}

/// Upload a Rule (as 12 i32s) to rule_store at rule index
pub fn uploadRule(self: *Bridge, rule_index: i32, rule: *const types.Rule) types.Status {
    const ints = rule.toInts();
    return self.uploadInts(.rule_store, rule_index * shared.RULE_INTS, &ints);
}

/// Download a Rule from rule_store at rule index
pub fn downloadRule(self: *Bridge, rule_index: i32) ?types.Rule {
    var ints: [12]i32 = undefined;
    const status = self.downloadInts(.rule_store, rule_index * shared.RULE_INTS, &ints);
    if (status.isErr()) return null;
    return types.Rule.fromInts(ints);
}

/// Upload a Kb struct (as KB_STRUCT_INTS i32s) to kb_store
pub fn uploadKb(self: *Bridge, kb_id: i32, data: []const i32) types.Status {
    if (data.len != shared.KB_STRUCT_INTS) return types.Status.err(.kb, .kb_not_found, kb_id);
    return self.uploadInts(.kb_store, kb_id * shared.KB_STRUCT_INTS, data);
}

/// Download a Kb struct from kb_store
pub fn downloadKb(self: *Bridge, kb_id: i32, out: []i32) types.Status {
    if (out.len != shared.KB_STRUCT_INTS) return types.Status.err(.kb, .kb_not_found, kb_id);
    return self.downloadInts(.kb_store, kb_id * shared.KB_STRUCT_INTS, out);
}

// ============================================================
// Convenience: dispatch helpers
// ============================================================

/// Compute workgroup count for N elements at workgroup size 256
pub fn workgroups(n: i32) i32 {
    return @divTrunc(n + shared.MAX_WORKGROUP - 1, shared.MAX_WORKGROUP);
}

/// Dispatch with auto-computed workgroup count for 1D
pub fn dispatch1D(self: *Bridge, params: gpu_params.ParamsBuffer, n_elements: i32) types.Status {
    return self.dispatch(&.{
        .params = params,
        .group_count_x = workgroups(n_elements),
    });
}

/// Dispatch with explicit 2D workgroup count
pub fn dispatch2D(self: *Bridge, params: gpu_params.ParamsBuffer, x: i32, y: i32) types.Status {
    return self.dispatch(&.{
        .params = params,
        .group_count_x = x,
        .group_count_y = y,
    });
}

/// Reset status + result counts, then dispatch
pub fn dispatchClean(self: *Bridge, request: *const DispatchRequest) types.Status {
    var status = self.resetStatusBuffer();
    if (status.isErr()) return status;
    status = self.resetResultCounts();
    if (status.isErr()) return status;
    return self.dispatch(request);
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
