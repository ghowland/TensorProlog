// ============================================================
// vlp_bridge.zig
// Host-device bridge. Vulkan resource management and dispatch.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const mem = @import("vlp_device_memory.zig");
const gpu = @import("vlp_gpu_params.zig");

// ============================================================
// Vulkan handle types — opaque pointers matching vulkan-zig
// If using vulkan-zig, replace these with vk.* types.
// Kept opaque here so vlp_bridge compiles without vulkan dep
// during early development / host-only testing.
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
// Bridge configuration
// ============================================================

pub const BridgeConfig = struct {
    sizing: mem.SizingConfig,
    shader_dir: []const u8,
    enable_validation: bool,
    preferred_device_index: i32, // -1 for auto
    force_host_visible_memory: bool, // for integrated GPUs / testing
};

// ============================================================
// Dispatch configuration — passed per kernel launch
// ============================================================

pub const DispatchConfig = struct {
    pipeline: gpu.PipelineId,
    group_count_x: i32,
    group_count_y: i32,
    group_count_z: i32,
    params_ptr: *const anyopaque,
    params_size: i32,
};

// ============================================================
// Operation type — for should_use_gpu decision
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

// GPU dispatch thresholds — integer element counts
const GPU_THRESHOLD_FACT_SCAN: i32 = 256;
const GPU_THRESHOLD_UNIFICATION: i32 = 32;
const GPU_THRESHOLD_RULE_MATCH: i32 = 64;
const GPU_THRESHOLD_BUILTIN: i32 = 512;
const GPU_THRESHOLD_SAMPLING_VOCAB: i32 = 256 * 1024;

// ============================================================
// Device properties — queried at init, cached
// ============================================================

pub const DeviceProperties = struct {
    device_name: [256]u8,
    device_name_len: i32,
    max_compute_shared_memory: i32,
    max_compute_workgroup_invocations: i32,
    max_compute_workgroup_count: [3]i32,
    max_compute_workgroup_size: [3]i32,
    max_storage_buffer_range: i64,
    max_uniform_buffer_range: i32,
    host_visible_memory_available: bool,
    host_coherent_memory_available: bool,
    total_device_memory: i64,
    compute_queue_family: u32,
    compute_queue_count: u32,
};

// ============================================================
// Bridge struct
// ============================================================

pub const Bridge = struct {
    allocator: std.mem.Allocator,

    // Vulkan handles
    instance: VkInstance,
    physical_device: VkPhysicalDevice,
    device: VkDevice,
    compute_queue: VkQueue,
    command_pool: VkCommandPool,

    // Device info
    properties: DeviceProperties,

    // Pipelines — one per kernel type
    pipelines: [gpu.PipelineId.count]VkPipeline,
    pipeline_layouts: [gpu.PipelineId.count]VkPipelineLayout,
    shader_modules: [gpu.PipelineId.count]VkShaderModule,

    // Descriptor infrastructure
    descriptor_pool: VkDescriptorPool,
    set_layouts: [4]VkDescriptorSetLayout,
    // Active descriptor sets — updated per session / per dispatch
    active_sets: [4]VkDescriptorSet,

    // Storage buffers
    model_weights_buffer: VkBuffer,
    kb_store_buffer: VkBuffer,
    fact_store_buffer: VkBuffer,
    rule_store_buffer: VkBuffer,
    term_store_buffer: VkBuffer,
    text_store_buffer: VkBuffer,
    grammar_store_buffer: VkBuffer,
    live_state_buffer: VkBuffer,
    scratch_a_buffer: VkBuffer,
    scratch_b_buffer: VkBuffer,
    kv_cache_buffer: VkBuffer,
    status_buffer: VkBuffer,
    result_counts_buffer: VkBuffer,
    params_buffer: VkBuffer,

    // Device memory objects
    model_memory: VkDeviceMemory,
    kb_data_memory: VkDeviceMemory,
    scratch_memory: VkDeviceMemory,
    control_memory: VkDeviceMemory,

    // Host-mapped pointers (null if not host-visible)
    kb_store_mapped: ?[*]u8,
    fact_store_mapped: ?[*]u8,
    rule_store_mapped: ?[*]u8,
    term_store_mapped: ?[*]u8,
    text_store_mapped: ?[*]u8,
    scratch_a_mapped: ?[*]u8,
    scratch_b_mapped: ?[*]u8,
    status_mapped: ?[*]i32,
    result_counts_mapped: ?[*]i32,
    params_mapped: ?[*]u8,

    // Staging buffers for non-host-visible transfers
    staging_upload_buffer: VkBuffer,
    staging_upload_memory: VkDeviceMemory,
    staging_upload_mapped: ?[*]u8,
    staging_upload_size: i64,
    staging_download_buffer: VkBuffer,
    staging_download_memory: VkDeviceMemory,
    staging_download_mapped: ?[*]u8,
    staging_download_size: i64,

    // Layout
    layout: mem.DeviceMemoryLayout,

    // Synchronization
    dispatch_fence: VkFence,
    // Pre-allocated command buffers for reuse
    dispatch_cmd: VkCommandBuffer,

    // Configuration
    config: BridgeConfig,
    initialized: bool,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(allocator: std.mem.Allocator, config: *const BridgeConfig) Bridge {
    // Implementation will:
    // 1. Create Vulkan instance (+ validation if enabled)
    // 2. Select physical device (by index or auto: pick first with compute queue)
    // 3. Query device properties, cache in DeviceProperties
    // 4. Create logical device + compute queue
    // 5. Create command pool + pre-allocate command buffer
    // 6. Compute memory layout from config.sizing
    // 7. Allocate all storage buffers + device memory
    // 8. Attempt persistent mapping for host-visible buffers
    // 9. Allocate staging buffers if not all memory is host-visible
    // 10. Load shader modules from config.shader_dir
    // 11. Create descriptor set layouts, pool, allocate sets
    // 12. Create all compute pipelines
    // 13. Create fence
    _ = allocator;
    _ = config;
    return std.mem.zeroes(Bridge);
}

pub fn deinit(self: *Bridge) void {
    // Destroy in reverse order:
    // fence, pipelines, pipeline layouts, shader modules,
    // descriptor pool, descriptor set layouts,
    // staging buffers, storage buffers, device memory,
    // command pool, device, instance
    self.initialized = false;
}

// ============================================================
// GPU Dispatch
// ============================================================

pub fn dispatch(self: *Bridge, config: *const DispatchConfig) types.Status {
    // 1. Write params to params_buffer (mapped or staged)
    // 2. Reset status_buffer and result_counts for this dispatch
    // 3. Begin command buffer
    // 4. Bind pipeline[config.pipeline]
    // 5. Bind active_sets[0..3]
    // 6. vkCmdDispatch(group_count_x, group_count_y, group_count_z)
    // 7. Pipeline barrier: compute write → host read
    // 8. End command buffer
    // 9. Submit to compute_queue with dispatch_fence
    // 10. Wait on dispatch_fence
    // 11. Check status_buffer for kernel-reported errors
    _ = self;
    _ = config;
    return types.Status.ok();
}

pub fn dispatchAsync(self: *Bridge, config: *const DispatchConfig) VkFence {
    // Same as dispatch steps 1-9, but returns fence instead of waiting.
    // Caller must call waitFence before reading results.
    _ = self;
    _ = config;
    return null;
}

pub fn dispatchSequence(self: *Bridge, configs: []const DispatchConfig) types.Status {
    // Records all dispatches into a single command buffer with
    // pipeline barriers between each. Single submit. Single fence wait.
    // Used for LLM forward pass (12+ kernels per layer).
    // Avoids per-dispatch submit overhead.
    _ = self;
    _ = configs;
    return types.Status.ok();
}

pub fn waitFence(self: *Bridge, fence: VkFence, timeout_ns: u64) types.Status {
    _ = self;
    _ = fence;
    _ = timeout_ns;
    return types.Status.ok();
}

// ============================================================
// Buffer Data Transfer
// ============================================================

pub fn uploadToBuffer(self: *Bridge, target: BufferTarget, offset: i64, data: []const u8) types.Status {
    // If target buffer is mapped: memcpy directly
    // Else: copy to staging_upload, record buffer copy cmd, submit, fence
    _ = self;
    _ = target;
    _ = offset;
    _ = data;
    return types.Status.ok();
}

pub fn downloadFromBuffer(self: *Bridge, source: BufferTarget, offset: i64, dest: []u8) types.Status {
    // If source buffer is mapped: memcpy directly
    // Else: record buffer copy to staging_download, submit, fence, memcpy
    _ = self;
    _ = source;
    _ = offset;
    _ = dest;
    return types.Status.ok();
}

pub fn copyBufferToBuffer(self: *Bridge, src: BufferTarget, src_offset: i64, dst: BufferTarget, dst_offset: i64, size: i64) types.Status {
    // Record vkCmdCopyBuffer, submit, fence
    _ = self;
    _ = src;
    _ = src_offset;
    _ = dst;
    _ = dst_offset;
    _ = size;
    return types.Status.ok();
}

pub fn fillBuffer(self: *Bridge, target: BufferTarget, offset: i64, size: i64, value: u32) types.Status {
    // Record vkCmdFillBuffer, submit, fence
    _ = self;
    _ = target;
    _ = offset;
    _ = size;
    _ = value;
    return types.Status.ok();
}

pub const BufferTarget = enum(i32) {
    model_weights = 0,
    kb_store = 1,
    fact_store = 2,
    rule_store = 3,
    term_store = 4,
    text_store = 5,
    grammar_store = 6,
    live_state = 7,
    scratch_a = 8,
    scratch_b = 9,
    kv_cache = 10,
    status = 11,
    result_counts = 12,
    params = 13,
};

// ============================================================
// Mapped pointer access — fast path for host-visible memory
// ============================================================

pub fn getMappedPtr(self: *Bridge, target: BufferTarget) ?[*]u8 {
    return switch (target) {
        .kb_store => self.kb_store_mapped,
        .fact_store => self.fact_store_mapped,
        .rule_store => self.rule_store_mapped,
        .term_store => self.term_store_mapped,
        .text_store => self.text_store_mapped,
        .scratch_a => self.scratch_a_mapped,
        .scratch_b => self.scratch_b_mapped,
        .params => self.params_mapped,
        else => null,
    };
}

pub fn isMapped(self: *Bridge, target: BufferTarget) bool {
    return self.getMappedPtr(target) != null;
}

// ============================================================
// Status / Result Readback
// ============================================================

pub fn readStatus(self: *Bridge, invocation_index: i32) i32 {
    if (self.status_mapped) |mapped| {
        return mapped[@intCast(invocation_index)];
    }
    // Fallback: download from buffer
    var val: i32 = 0;
    const bytes: *[4]u8 = @ptrCast(&val);
    _ = self.downloadFromBuffer(.status, @as(i64, invocation_index) * 4, bytes);
    return val;
}

pub fn readResultCount(self: *Bridge, slot: i32) i32 {
    if (self.result_counts_mapped) |mapped| {
        return mapped[@intCast(slot)];
    }
    var val: i32 = 0;
    const bytes: *[4]u8 = @ptrCast(&val);
    _ = self.downloadFromBuffer(.result_counts, @as(i64, slot) * 4, bytes);
    return val;
}

pub fn resetStatusBuffer(self: *Bridge) types.Status {
    return self.fillBuffer(.status, 0, self.layout.status_buffer_size, 0);
}

pub fn resetResultCounts(self: *Bridge) types.Status {
    return self.fillBuffer(.result_counts, 0, self.layout.result_counts_size, 0);
}

// ============================================================
// Descriptor Set Updates
// ============================================================

pub fn updateModelDescriptors(self: *Bridge) types.Status {
    // Bind model buffers to Set 0 (done once at init)
    _ = self;
    return types.Status.ok();
}

pub fn updateKbDescriptors(self: *Bridge, session_kb_offset: i64, session_fact_offset: i64) types.Status {
    // Update Set 1 bindings to point to session's region of KB/fact stores
    // Called when switching active session
    _ = self;
    _ = session_kb_offset;
    _ = session_fact_offset;
    return types.Status.ok();
}

pub fn updateScratchDescriptors(self: *Bridge) types.Status {
    // Update Set 2 bindings (scratch_a, scratch_b, kv_cache)
    // Called before dispatches that use different scratch regions
    _ = self;
    return types.Status.ok();
}

pub fn updateControlDescriptors(self: *Bridge) types.Status {
    // Update Set 3 bindings (params, status, result_counts)
    // Usually stable — only needed if buffer reallocation occurs
    _ = self;
    return types.Status.ok();
}

// ============================================================
// GPU vs Host Decision
// ============================================================

pub fn shouldUseGpu(self: *Bridge, op: OperationType, data_size: i32) bool {
    if (!self.initialized) return false;
    return switch (op) {
        .llm_forward => true,
        .fact_scan => data_size > GPU_THRESHOLD_FACT_SCAN,
        .unification => data_size > GPU_THRESHOLD_UNIFICATION,
        .rule_match => data_size > GPU_THRESHOLD_RULE_MATCH,
        .builtin_array => data_size > GPU_THRESHOLD_BUILTIN,
        .text_grammar => false, // always host
        .access_check => false, // always host
        .sampling => data_size > GPU_THRESHOLD_SAMPLING_VOCAB,
    };
}

// ============================================================
// Shared memory tier detection
// ============================================================

pub fn sharedMemoryTier(self: *Bridge) gpu.SharedMemoryTier {
    const sm = self.properties.max_compute_shared_memory;
    if (sm >= gpu.SHARED_MEM_H100) return .h100;
    if (sm >= gpu.SHARED_MEM_EXTENDED) return .extended;
    return .baseline;
}

// ============================================================
// Convenience: write typed struct to params buffer
// ============================================================

pub fn writeParams(self: *Bridge, comptime T: type, params: *const T) types.Status {
    const bytes: []const u8 = @as([*]const u8, @ptrCast(params))[0..@sizeOf(T)];
    return self.uploadToBuffer(.params, 0, bytes);
}

// ============================================================
// Convenience: read typed struct array from scratch buffer
// ============================================================

pub fn readScratchSlice(self: *Bridge, comptime T: type, target: BufferTarget, offset: i64, count: i32, out: []T) types.Status {
    const byte_size = @as(i64, count) * @sizeOf(T);
    const dest: []u8 = @as([*]u8, @ptrCast(out.ptr))[0..@intCast(byte_size)];
    return self.downloadFromBuffer(target, offset, dest);
}

// ============================================================
// Device query helpers
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
