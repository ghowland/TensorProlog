// ============================================================
// src/runner/runner_ops.zig
// ============================================================

const RunnerManager = @import("runner_manager.zig").RunnerManager;

pub fn createProcessorRunner(mgr: *RunnerManager, config: ProcessorConfig, store: *KBStore) ?i32 {
    return createProcessor(config, store, &mgr.table);
}

pub fn createInternalRunner(mgr: *RunnerManager, config: InternalConfig, store: *KBStore) ?i32 {
    return internal_mod.createInternal(config, store, &mgr.table);
}

pub fn createBatchRunner(mgr: *RunnerManager, config: BatchConfig, store: *KBStore) ?i32 {
    return batch_mod.createBatch(config, store, &mgr.table);
}

pub fn startProcessorRunner(mgr: *RunnerManager, id: i32) VlpStatus {
    const runner = mgr.table.get(id) orelse return .err_kb_not_found;
    if (runner.runner_type != .processor) return .err_invalid_qbasis;
    if (runner.state == .running) return .ok;
    runner.state = .running;
    _ = mgr.pool.submit(.{ .runner_id = id, .action = .run_cycle });
    return .ok;
}

pub fn startInternalRunner(mgr: *RunnerManager, id: i32) VlpStatus {
    const runner = mgr.table.get(id) orelse return .err_kb_not_found;
    if (runner.runner_type != .internal) return .err_invalid_qbasis;
    if (runner.state == .running) return .ok;
    runner.state = .running;
    _ = mgr.pool.submit(.{ .runner_id = id, .action = .run_cycle });
    return .ok;
}

pub fn startBatchRunner(mgr: *RunnerManager, id: i32) VlpStatus {
    const runner = mgr.table.get(id) orelse return .err_kb_not_found;
    if (runner.runner_type != .batch) return .err_invalid_qbasis;
    if (runner.state == .running) return .ok;
    runner.state = .running;
    _ = mgr.pool.submit(.{ .runner_id = id, .action = .run_cycle });
    return .ok;
}

pub fn recycleProcessor(mgr: *RunnerManager, id: i32, store: *KBStore) RecycleResult {
    const runner = mgr.table.get(id) orelse return .{
        .status = .err_kb_not_found,
        .old_session_id = -1,
        .new_session_id = -1,
        .snapshot_size = 0,
    };
    return processorRecycle(runner, store);
}

const internal_mod = @import("internal.zig");
const batch_mod = @import("batch.zig");
