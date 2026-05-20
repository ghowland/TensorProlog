// ============================================================
// src/runner/poller.zig
// ============================================================

const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const prolog_rule = @import("../prolog/rule.zig");
const prolog_query = @import("../prolog/query.zig");
const level_stats_mod = @import("../engine/level_stats.zig");
const audit_mod = @import("../safety/audit.zig");
const kb_types = @import("../kb/types.zig");
const scratchpad_mod = @import("../engine/scratchpad.zig");

const KBStore = kb_store_mod.KBStore;
const VlpFact = kb_types.VlpFact;

pub const PollerState = struct {
    runner_id: i32,
    store: *KBStore,
    interval_ms: i32,
    notification_kb_id: i32,
    log_kb_id: i32,
    max_consecutive_errors: i32,
    session_kb_root: i32,
    running: std.atomic.Value(i32),
    thread: ?std.Thread,
    level_stats: level_stats_mod.LevelStats,
    output_buf: [4096]u8,
};

pub fn createPoller(config: PollerConfig, store: *KBStore, table: *RunnerTable) ?i32 {
    const id = table.allocate() orelse return null;
    var runner = table.get(id) orelse return null;
    runner.runner_type = .poller;
    runner.session_id = config.session_id;
    runner.interval_ms = config.interval_ms;
    runner.max_consecutive_errors = config.max_consecutive_errors;
    runner.notification_kb_id = config.notification_kb_id;
    runner.log_kb_id = config.log_kb_id;
    runner.state = .stopped;
    _ = store;
    return id;
}

pub fn startPoller(table: *RunnerTable, pool: *ThreadPool, id: i32) VlpStatus {
    const runner = table.get(id) orelse return .err_kb_not_found;
    if (runner.state == .running) return .ok;
    runner.state = .running;
    _ = pool.submit(.{ .runner_id = id, .action = .run_cycle });
    return .ok;
}

pub fn pollerIteration(runner: *VlpRunner, store: *KBStore, output_buf: []u8) PollerIterationResult {
    var result = PollerIterationResult{
        .status = .ok,
        .rules_fired = 0,
        .tokens_consumed = 0,
        .output_len = 0,
        .level = .l3,
    };

    if (runner.session_id < 0) {
        result.status = .err_kb_not_found;
        return result;
    }

    const scope_kb = runner.notification_kb_id;
    if (scope_kb < 0) return result;

    var fired_ids: [64]i32 = undefined;
    var n_fired: i32 = 0;

    const fire_result = prolog_rule.fireAll(store, scope_kb, &fired_ids, 64, &n_fired);
    if (fire_result != .ok) {
        result.status = fire_result;
        return result;
    }

    result.rules_fired = n_fired;

    if (n_fired > 0) {
        result.level = .l3;
        result.tokens_consumed = 0;
    }

    _ = output_buf;

    return result;
}

pub const PollerIterationResult = struct {
    status: VlpStatus,
    rules_fired: i32,
    tokens_consumed: i32,
    output_len: i32,
    level: types.VlpExecutionLevel,
};

pub fn pollerLoop(runner: *VlpRunner, store: *KBStore, running_flag: *std.atomic.Value(i32)) void {
    var output_buf: [4096]u8 = undefined;

    while (running_flag.load(.seq_cst) != 0 and runner.state == .running) {
        const start_ms = std.time.milliTimestamp();

        const iter_result = pollerIteration(runner, store, &output_buf);

        runner.iterations_completed += 1;
        runner.last_iteration_timestamp = timestampNow();

        const end_ms = std.time.milliTimestamp();
        runner.last_iteration_ms = @intCast(end_ms - start_ms);

        if (iter_result.status != .ok) {
            runner.errors_consecutive += 1;
            runner.errors_total += 1;
            if (runner.errors_consecutive >= runner.max_consecutive_errors) {
                runner.state = .err;
                break;
            }
        } else {
            runner.errors_consecutive = 0;
        }

        if (iter_result.output_len > 0 and runner.notification_kb_id >= 0) {
            writeOutputToKB(store, runner.notification_kb_id, output_buf[0..@intCast(iter_result.output_len)]);
        }

        const elapsed_ms = end_ms - start_ms;
        const sleep_ms = @max(@as(i64, runner.interval_ms) - elapsed_ms, 0);
        if (sleep_ms > 0) {
            std.time.sleep(@intCast(sleep_ms * std.time.ns_per_ms));
        }
    }
}

fn writeOutputToKB(store: *KBStore, kb_id: i32, data: []const u8) void {
    const text_ref = store.text.append(data);
    const fact = VlpFact{
        .tag = .text,
        .value = .{ .v = text_ref.offset, .r0 = @intCast(text_ref.length) },
        .provenance = .{
            .source_type = .prolog_derivation,
            .source_kb_id = kb_id,
            .source_slot_id = 0,
            .confidence = .{ .v = Q16.D, .r0 = 0 },
            .timestamp = timestampNow(),
            .derivation_rule_id = -1,
        },
    };
    const kb = store.getKB(kb_id) orelse return;
    const slot = kb.facts_count;
    if (slot < kb.facts_capacity) {
        _ = fact_mod.factAssert(store, kb_id, slot, &fact);
    }
}
