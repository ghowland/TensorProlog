// ============================================================
// src/ops/process.zig
// ============================================================

pub const ProcessHandle = struct {
    pid: i32,
    active: bool,
};

pub fn procStart(command: []const u8) ProcessHandle {
    _ = command;
    return .{ .pid = -1, .active = false };
}

pub fn procKill(handle: *ProcessHandle) VlpStatus {
    if (!handle.active) return .ok;
    handle.active = false;
    return .ok;
}

pub fn procStatus(handle: *const ProcessHandle) struct { running: bool, exit_code: i32 } {
    return .{ .running = handle.active, .exit_code = if (handle.active) -1 else 0 };
}
