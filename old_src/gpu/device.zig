// ============================================================
// src/gpu/device.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;

pub const DeviceType = enum(i8) {
    cpu_fallback = 0,
    gpu_int8 = 1,
    gpu_int8_fru = 2,
    gpu_native_qiu = 3,
};

pub const DeviceProps = struct {
    device_id: i32,
    device_type: DeviceType,
    n_compute_units: i32,
    max_q_basis: i32,
    has_fru: bool,
    kb_cache_size_bytes: i32,
    max_shared_mem_per_block: i32,
    max_registers_per_block: i32,
    max_concurrent_sessions: i32,
    global_mem_bytes: i64,
    mem_bandwidth_bytes_sec: i64,
    clock_rate_hz: i32,
    warp_size: i32,
    name: [64]u8,
    name_len: i32,
};

pub fn defaultDeviceProps() DeviceProps {
    return .{
        .device_id = 0,
        .device_type = .cpu_fallback,
        .n_compute_units = 1,
        .max_q_basis = 335,
        .has_fru = false,
        .kb_cache_size_bytes = 65536,
        .max_shared_mem_per_block = 228 * 1024,
        .max_registers_per_block = 65536,
        .max_concurrent_sessions = 10000,
        .global_mem_bytes = 0,
        .mem_bandwidth_bytes_sec = 0,
        .clock_rate_hz = 0,
        .warp_size = 32,
        .name = undefined,
        .name_len = 0,
    };
}

pub const DeviceState = struct {
    initialized: bool,
    device_count: i32,
    current_device: i32,
    props: [8]DeviceProps,
};

var global_device_state = DeviceState{
    .initialized = false,
    .device_count = 0,
    .current_device = -1,
    .props = undefined,
};

pub fn deviceInit(flags: u32) VlpStatus {
    _ = flags;
    if (global_device_state.initialized) return .ok;

    global_device_state.device_count = 1;
    global_device_state.current_device = 0;

    var props = defaultDeviceProps();
    props.device_id = 0;
    props.device_type = .cpu_fallback;
    props.n_compute_units = @intCast(std.Thread.getCpuCount() catch 4);
    props.global_mem_bytes = 0;

    const cpu_name = "CPU Fallback (Integer ALU)";
    @memcpy(props.name[0..cpu_name.len], cpu_name);
    props.name_len = @intCast(cpu_name.len);

    global_device_state.props[0] = props;
    global_device_state.initialized = true;

    return .ok;
}

pub fn deviceGetCount() struct { count: i32, status: VlpStatus } {
    if (!global_device_state.initialized) return .{ .count = 0, .status = .err_not_initialized };
    return .{ .count = global_device_state.device_count, .status = .ok };
}

pub fn deviceGetProps(device_id: i32) ?*const DeviceProps {
    if (!global_device_state.initialized) return null;
    if (device_id < 0 or device_id >= global_device_state.device_count) return null;
    return &global_device_state.props[@intCast(device_id)];
}

pub fn deviceSetCurrent(device_id: i32) VlpStatus {
    if (!global_device_state.initialized) return .err_not_initialized;
    if (device_id < 0 or device_id >= global_device_state.device_count) return .err_invalid_device;
    global_device_state.current_device = device_id;
    return .ok;
}

pub fn deviceGetCurrent() struct { device_id: i32, status: VlpStatus } {
    if (!global_device_state.initialized) return .{ .device_id = -1, .status = .err_not_initialized };
    return .{ .device_id = global_device_state.current_device, .status = .ok };
}

pub fn deviceSynchronize() VlpStatus {
    if (!global_device_state.initialized) return .err_not_initialized;
    return .ok;
}

pub fn deviceReset() VlpStatus {
    if (!global_device_state.initialized) return .err_not_initialized;
    return .ok;
}

pub fn isInitialized() bool {
    return global_device_state.initialized;
}

pub fn getErrorString(status: VlpStatus) []const u8 {
    return switch (status) {
        .ok => "OK",
        .err_not_initialized => "ERR_NOT_INITIALIZED",
        .err_invalid_device => "ERR_INVALID_DEVICE",
        .err_out_of_memory => "ERR_OUT_OF_MEMORY",
        .err_invalid_qbasis => "ERR_INVALID_QBASIS",
        .err_qbasis_mismatch => "ERR_QBASIS_MISMATCH",
        .err_kb_not_found => "ERR_KB_NOT_FOUND",
        .err_kb_access_denied => "ERR_KB_ACCESS_DENIED",
        .err_kb_full => "ERR_KB_FULL",
        .err_slot_out_of_range => "ERR_SLOT_OUT_OF_RANGE",
        .err_grant_denied => "ERR_GRANT_DENIED",
        .err_session_limit => "ERR_SESSION_LIMIT",
        .err_stream_busy => "ERR_STREAM_BUSY",
        .err_invalid_kernel => "ERR_INVALID_KERNEL",
        .err_launch_failure => "ERR_LAUNCH_FAILURE",
        .err_prolog_depth_exceeded => "ERR_PROLOG_DEPTH_EXCEEDED",
        .err_remainder_overflow => "ERR_REMAINDER_OVERFLOW",
        .err_fru_not_available => "ERR_FRU_NOT_AVAILABLE",
        .err_grammar_invalid => "ERR_GRAMMAR_INVALID",
        .err_primitive_bounds => "ERR_PRIMITIVE_BOUNDS",
        .err_snapshot_failed => "ERR_SNAPSHOT_FAILED",
        .err_clone_failed => "ERR_CLONE_FAILED",
        .err_determinism_violation => "ERR_DETERMINISM_VIOLATION",
        .err_command_parse => "ERR_COMMAND_PARSE",
    };
}
