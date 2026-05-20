// ============================================================
// src/runner/types.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;

pub const VlpRunnerType = enum(i8) {
    poller = 0,
    processor = 1,
    internal = 2,
    batch = 3,
};

pub const VlpRunnerState = enum(i8) {
    stopped = 0,
    running = 1,
    err = 2,
    recycling = 3,
};

pub const VlpRunnerAction = enum(i8) {
    run_cycle = 0,
    recycle = 1,
    stop = 2,
    kill = 3,
};

pub const VlpRunner = struct {
    id: i32,
    runner_type: VlpRunnerType,
    state: VlpRunnerState,
    session_id: i32,

    interval_ms: i32,
    max_turns_before_recycle: i32,
    max_consecutive_errors: i32,

    iterations_completed: i64,
    errors_consecutive: i32,
    errors_total: i64,
    last_iteration_ms: i32,
    last_iteration_timestamp: i32,

    recycle_count: i32,
    last_recycle_timestamp: i32,

    notification_kb_id: i32,
    log_kb_id: i32,
    compact_rules_kb_id: i32,
    task_queue_kb_id: i32,
    result_queue_kb_id: i32,

    max_concurrent_batch: i32,
};

pub const RunnerTask = struct {
    runner_id: i32,
    action: VlpRunnerAction,
};

pub const PollerConfig = struct {
    session_id: i32,
    interval_ms: i32,
    max_consecutive_errors: i32,
    notification_kb_id: i32,
    log_kb_id: i32,
};

pub const ProcessorConfig = struct {
    session_id: i32,
    max_turns_before_recycle: i32,
    max_consecutive_errors: i32,
    compact_rules_kb_id: i32,
    log_kb_id: i32,
};

pub const InternalConfig = struct {
    session_id: i32,
    interval_ms: i32,
    max_consecutive_errors: i32,
    log_kb_id: i32,
};

pub const BatchConfig = struct {
    session_id: i32,
    task_queue_kb_id: i32,
    result_queue_kb_id: i32,
    max_concurrent: i32,
    max_consecutive_errors: i32,
    log_kb_id: i32,
};

pub const RunnerStatus = struct {
    id: i32,
    runner_type: VlpRunnerType,
    state: VlpRunnerState,
    iterations_completed: i64,
    errors_consecutive: i32,
    errors_total: i64,
    last_iteration_ms: i32,
    recycle_count: i32,
};

pub fn defaultRunner() VlpRunner {
    return .{
        .id = -1,
        .runner_type = .poller,
        .state = .stopped,
        .session_id = -1,
        .interval_ms = 60000,
        .max_turns_before_recycle = 200,
        .max_consecutive_errors = 5,
        .iterations_completed = 0,
        .errors_consecutive = 0,
        .errors_total = 0,
        .last_iteration_ms = 0,
        .last_iteration_timestamp = 0,
        .recycle_count = 0,
        .last_recycle_timestamp = 0,
        .notification_kb_id = -1,
        .log_kb_id = -1,
        .compact_rules_kb_id = -1,
        .task_queue_kb_id = -1,
        .result_queue_kb_id = -1,
        .max_concurrent_batch = 4,
    };
}

pub fn getStatus(runner: *const VlpRunner) RunnerStatus {
    return .{
        .id = runner.id,
        .runner_type = runner.runner_type,
        .state = runner.state,
        .iterations_completed = runner.iterations_completed,
        .errors_consecutive = runner.errors_consecutive,
        .errors_total = runner.errors_total,
        .last_iteration_ms = runner.last_iteration_ms,
        .recycle_count = runner.recycle_count,
    };
}
