// ============================================================
// src/runner/processor.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");
const kb_types = @import("../kb/types.zig");
const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const prolog_rule = @import("../prolog/rule.zig");
const level_stats_mod = @import("../engine/level_stats.zig");
const session_mod = @import("../session/lifecycle.zig");
const snapshot_mod = @import("../session/snapshot.zig");
const runner_types = @import("types.zig");
const pool_mod = @import("pool.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;
const VlpFact = kb_types.VlpFact;
const KBStore = kb_store_mod.KBStore;
const VlpRunner = runner_types.VlpRunner;
const ProcessorConfig = runner_types.ProcessorConfig;
const RunnerTable = pool_mod.RunnerTable;

pub const ConnectionState = struct {
    connected: bool,
    source_type: SourceType,
    reconnect_attempts: i32,
    last_connect_timestamp: i32,
    bytes_received: i64,
    items_ingested: i64,
};

pub const SourceType = enum(i8) {
    prometheus = 0,
    deploy_api = 1,
    alert_stream = 2,
    custom = 3,
};

pub const IngestResult = struct {
    status: VlpStatus,
    facts_written: i32,
    rules_fired: i32,
    used_llm: bool,
};

pub fn createProcessor(config: ProcessorConfig, store: *KBStore, table: *RunnerTable) ?i32 {
    const id = table.allocate() orelse return null;
    var runner = table.get(id) orelse return null;
    runner.runner_type = .processor;
    runner.session_id = config.session_id;
    runner.max_turns_before_recycle = config.max_turns_before_recycle;
    runner.max_consecutive_errors = config.max_consecutive_errors;
    runner.compact_rules_kb_id = config.compact_rules_kb_id;
    runner.log_kb_id = config.log_kb_id;
    runner.state = .stopped;
    _ = store;
    return id;
}

pub fn processorIteration(runner: *VlpRunner, store: *KBStore, data: []const u8, target_kb_id: i32) IngestResult {
    var result = IngestResult{
        .status = .ok,
        .facts_written = 0,
        .rules_fired = 0,
        .used_llm = false,
    };

    if (runner.compact_rules_kb_id >= 0) {
        var fired_ids: [64]i32 = undefined;
        var n_fired: i32 = 0;
        const fire_status = prolog_rule.fireAll(store, runner.compact_rules_kb_id, &fired_ids, 64, &n_fired);
        if (fire_status == .ok and n_fired > 0) {
            result.rules_fired = n_fired;
            result.used_llm = false;
            return result;
        }
    }

    if (data.len > 0 and target_kb_id >= 0) {
        const text_ref = store.text.append(data);
        const fact = VlpFact{
            .tag = .text,
            .value = .{ .v = text_ref.offset, .r0 = @intCast(text_ref.length) },
            .provenance = .{
                .source_type = .prometheus,
                .source_kb_id = target_kb_id,
                .source_slot_id = 0,
                .confidence = .{ .v = 62259, .r0 = 0 },
                .timestamp = timestampNow(),
                .derivation_rule_id = -1,
            },
        };
        const kb = store.getKB(target_kb_id);
        if (kb) |k| {
            if (k.facts_count < k.facts_capacity) {
                _ = fact_mod.factAssert(store, target_kb_id, k.facts_count, &fact);
                result.facts_written = 1;
            }
        }
    }

    return result;
}

pub const RecycleResult = struct {
    status: VlpStatus,
    old_session_id: i32,
    new_session_id: i32,
    snapshot_size: i64,
};

pub fn processorRecycle(runner: *VlpRunner, store: *KBStore) RecycleResult {
    var result = RecycleResult{
        .status = .ok,
        .old_session_id = runner.session_id,
        .new_session_id = -1,
        .snapshot_size = 0,
    };

    runner.state = .recycling;

    var snap_buf: [65536]u8 = undefined;
    var snap_len: i32 = 0;
    const snap_status = snapshot_mod.snapshotSave(store, runner.session_id, &snap_buf, @intCast(snap_buf.len), &snap_len);
    if (snap_status != .ok) {
        result.status = snap_status;
        runner.state = .err;
        return result;
    }
    result.snapshot_size = @intCast(snap_len);

    const new_session_id = runner.session_id + 1000;
    const restore_status = snapshot_mod.snapshotRestore(store, new_session_id, &snap_buf, snap_len);
    if (restore_status != .ok) {
        result.status = restore_status;
        runner.state = .err;
        return result;
    }

    result.new_session_id = new_session_id;
    runner.session_id = new_session_id;
    runner.recycle_count += 1;
    runner.last_recycle_timestamp = timestampNow();
    runner.state = .running;

    return result;
}

pub fn processorReconnect(conn: *ConnectionState, max_attempts: i32) VlpStatus {
    var backoff_ms: i64 = 1000;
    const ma: usize = @intCast(@max(max_attempts, 1));

    for (0..ma) |attempt| {
        std.time.sleep(@intCast(backoff_ms * std.time.ns_per_ms));

        conn.connected = true;
        conn.reconnect_attempts = @intCast(attempt + 1);
        conn.last_connect_timestamp = timestampNow();
        return .ok;

        backoff_ms = @min(backoff_ms * 2, 60000);
    }

    conn.connected = false;
    return .err_snapshot_failed;
}

pub fn processorLoop(runner: *VlpRunner, store: *KBStore, conn: *ConnectionState, running_flag: *std.atomic.Value(i32)) void {
    var turn_count: i32 = 0;
    var ingest_buf: [8192]u8 = undefined;

    while (running_flag.load(.seq_cst) != 0 and runner.state == .running) {
        if (!conn.connected) {
            const reconn_status = processorReconnect(conn, 10);
            if (reconn_status != .ok) {
                runner.state = .err;
                break;
            }
        }

        const data_len: usize = 0;
        const data = ingest_buf[0..data_len];

        if (data.len == 0) {
            std.time.sleep(100 * std.time.ns_per_ms);
            continue;
        }

        const target_kb = runner.log_kb_id;
        const iter_result = processorIteration(runner, store, data, target_kb);

        runner.iterations_completed += 1;
        runner.last_iteration_timestamp = timestampNow();
        turn_count += 1;

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

        if (turn_count >= runner.max_turns_before_recycle) {
            _ = processorRecycle(runner, store);
            turn_count = 0;
        }
    }
}

fn timestampNow() i32 {
    return @intCast(@divTrunc(std.time.milliTimestamp(), 1000));
}
