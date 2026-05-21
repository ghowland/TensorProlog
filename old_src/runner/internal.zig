// ============================================================
// src/runner/internal.zig
// ============================================================

const InternalConfig = runner_types.InternalConfig;

pub const ComputeFn = *const fn (*KBStore, i32) VlpStatus;

pub const InternalState = struct {
    runner_id: i32,
    store: *KBStore,
    compute_fn: ?ComputeFn,
    session_kb_root: i32,
};

pub fn createInternal(config: InternalConfig, store: *KBStore, table: *RunnerTable) ?i32 {
    const id = table.allocate() orelse return null;
    var runner = table.get(id) orelse return null;
    runner.runner_type = .internal;
    runner.session_id = config.session_id;
    runner.interval_ms = config.interval_ms;
    runner.max_consecutive_errors = config.max_consecutive_errors;
    runner.log_kb_id = config.log_kb_id;
    runner.state = .stopped;
    _ = store;
    return id;
}

pub fn internalIteration(runner: *VlpRunner, store: *KBStore, compute_fn: ?ComputeFn) VlpStatus {
    if (compute_fn) |f| {
        return f(store, runner.session_id);
    }
    return .ok;
}

pub fn internalLoop(runner: *VlpRunner, store: *KBStore, compute_fn: ?ComputeFn, running_flag: *std.atomic.Value(i32)) void {
    while (running_flag.load(.seq_cst) != 0 and runner.state == .running) {
        const start_ms = std.time.milliTimestamp();

        const status = internalIteration(runner, store, compute_fn);

        runner.iterations_completed += 1;
        runner.last_iteration_timestamp = timestampNow();

        const end_ms = std.time.milliTimestamp();
        runner.last_iteration_ms = @intCast(end_ms - start_ms);

        if (status != .ok) {
            runner.errors_consecutive += 1;
            runner.errors_total += 1;
            if (runner.errors_consecutive >= runner.max_consecutive_errors) {
                runner.state = .err;
                break;
            }
        } else {
            runner.errors_consecutive = 0;
        }

        const elapsed_ms = end_ms - start_ms;
        const sleep_ms = @max(@as(i64, runner.interval_ms) - elapsed_ms, 0);
        if (sleep_ms > 0) {
            std.time.sleep(@intCast(sleep_ms * std.time.ns_per_ms));
        }
    }
}

pub fn defaultComputeRollingAverage(store: *KBStore, session_id: i32) VlpStatus {
    _ = session_id;
    _ = store;
    return .ok;
}

pub fn defaultComputeTrendDetection(store: *KBStore, session_id: i32) VlpStatus {
    _ = session_id;
    _ = store;
    return .ok;
}

pub fn defaultComputeCoverageGap(store: *KBStore, session_id: i32) VlpStatus {
    _ = session_id;
    _ = store;
    return .ok;
}
