// ============================================================
// src/runner/batch.zig
// ============================================================

const BatchConfig = runner_types.BatchConfig;
const primitives_queue = @import("../primitives/queue.zig");

pub const BatchClone = struct {
    session_id: i32,
    task: VlpFact,
    completed: bool,
    result_status: VlpStatus,
    output_buf: [4096]u8,
    output_len: i32,
};

pub const MAX_BATCH_CONCURRENT: usize = 16;

pub fn createBatch(config: BatchConfig, store: *KBStore, table: *RunnerTable) ?i32 {
    const id = table.allocate() orelse return null;
    var runner = table.get(id) orelse return null;
    runner.runner_type = .batch;
    runner.session_id = config.session_id;
    runner.max_consecutive_errors = config.max_consecutive_errors;
    runner.task_queue_kb_id = config.task_queue_kb_id;
    runner.result_queue_kb_id = config.result_queue_kb_id;
    runner.max_concurrent_batch = config.max_concurrent;
    runner.log_kb_id = config.log_kb_id;
    runner.state = .stopped;
    _ = store;
    return id;
}

pub fn batchPopTask(store: *KBStore, queue_kb_id: i32) ?VlpFact {
    if (queue_kb_id < 0) return null;
    const kb = store.getKB(queue_kb_id) orelse return null;
    if (kb.facts_count <= 0) return null;

    const fact = fact_mod.factQuery(store, queue_kb_id, 0);
    if (fact) |f| {
        if (f.tag == .empty) return null;
        _ = fact_mod.factRetract(store, queue_kb_id, 0);
        return f;
    }
    return null;
}

pub fn batchWriteResult(store: *KBStore, result_kb_id: i32, task_result: *const VlpFact) void {
    if (result_kb_id < 0) return;
    const kb = store.getKB(result_kb_id) orelse return;
    if (kb.facts_count >= kb.facts_capacity) return;
    _ = fact_mod.factAssert(store, result_kb_id, kb.facts_count, task_result);
}

pub fn batchProcessTask(store: *KBStore, clone: *BatchClone, runner: *VlpRunner) void {
    _ = runner;

    if (clone.task.tag == .empty) {
        clone.result_status = .ok;
        clone.completed = true;
        return;
    }

    const result_fact = VlpFact{
        .tag = .value,
        .value = clone.task.value,
        .provenance = .{
            .source_type = .prolog_derivation,
            .source_kb_id = -1,
            .source_slot_id = 0,
            .confidence = .{ .v = Q16.D, .r0 = 0 },
            .timestamp = timestampNow(),
            .derivation_rule_id = -1,
        },
    };

    if (clone.session_id >= 0) {
        const kb = store.getKB(clone.session_id);
        if (kb) |k| {
            if (k.facts_count < k.facts_capacity) {
                _ = fact_mod.factAssert(store, clone.session_id, k.facts_count, &result_fact);
            }
        }
    }

    clone.result_status = .ok;
    clone.completed = true;
}

pub fn batchIteration(runner: *VlpRunner, store: *KBStore, clones: []BatchClone, active_count: *i32) VlpStatus {
    var w: usize = 0;
    const ac: usize = @intCast(active_count.*);
    for (0..ac) |i| {
        if (clones[i].completed) {
            if (runner.result_queue_kb_id >= 0) {
                const result_fact = VlpFact{
                    .tag = .value,
                    .value = clones[i].task.value,
                    .provenance = .{
                        .source_type = .prolog_derivation,
                        .source_kb_id = runner.result_queue_kb_id,
                        .source_slot_id = 0,
                        .confidence = .{ .v = Q16.D, .r0 = 0 },
                        .timestamp = timestampNow(),
                        .derivation_rule_id = -1,
                    },
                };
                batchWriteResult(store, runner.result_queue_kb_id, &result_fact);
            }
        } else {
            if (w != i) clones[w] = clones[i];
            w += 1;
        }
    }
    active_count.* = @intCast(w);

    const max_c: usize = @intCast(@min(runner.max_concurrent_batch, @as(i32, MAX_BATCH_CONCURRENT)));
    const ac_now: usize = @intCast(active_count.*);
    while (ac_now + (active_count.* - @as(i32, @intCast(ac_now))) < @as(i32, @intCast(max_c))) {
        const maybe_task = batchPopTask(store, runner.task_queue_kb_id);
        if (maybe_task) |task| {
            const idx: usize = @intCast(active_count.*);
            if (idx >= clones.len) break;
            clones[idx] = .{
                .session_id = runner.session_id,
                .task = task,
                .completed = false,
                .result_status = .ok,
                .output_buf = undefined,
                .output_len = 0,
            };
            active_count.* += 1;
        } else {
            break;
        }
    }

    const new_ac: usize = @intCast(active_count.*);
    for (0..new_ac) |i| {
        if (!clones[i].completed) {
            batchProcessTask(store, &clones[i], runner);
        }
    }

    return .ok;
}

pub fn batchLoop(runner: *VlpRunner, store: *KBStore, running_flag: *std.atomic.Value(i32)) void {
    var clones: [MAX_BATCH_CONCURRENT]BatchClone = undefined;
    var active_count: i32 = 0;

    for (&clones) |*c| {
        c.completed = true;
        c.session_id = -1;
        c.result_status = .ok;
        c.output_len = 0;
    }

    while (running_flag.load(.seq_cst) != 0 and runner.state == .running) {
        const start_ms = std.time.milliTimestamp();

        const status = batchIteration(runner, store, &clones, &active_count);

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

        if (active_count == 0) {
            std.time.sleep(100 * std.time.ns_per_ms);
        } else {
            std.time.sleep(10 * std.time.ns_per_ms);
        }
    }

    active_count = 0;
}
