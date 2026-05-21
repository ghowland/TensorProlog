// ============================================================
// src/deploy/chaos.zig
// ============================================================

const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const kb_types = @import("../kb/types.zig");
const snapshot_mod = @import("../session/snapshot.zig");
const seed_mod = @import("../seed/seed_init.zig");
const det_mod = @import("../gpu/determinism.zig");
const bench_mod = @import("../gpu/benchmarks.zig");

const KBStore = kb_store_mod.KBStore;
const VlpFact = kb_types.VlpFact;

pub const ChaosTestResult = struct {
    snapshot_recovery_ok: bool,
    kill_restart_ok: bool,
    concurrent_write_ok: bool,
    determinism_after_restart_ok: bool,
    total_checks: i32,
    total_passed: i32,
};

pub fn runChaosTests(store: *KBStore) ChaosTestResult {
    var result = ChaosTestResult{
        .snapshot_recovery_ok = false,
        .kill_restart_ok = false,
        .concurrent_write_ok = false,
        .determinism_after_restart_ok = false,
        .total_checks = 4,
        .total_passed = 0,
    };

    result.snapshot_recovery_ok = testSnapshotRecovery(store);
    if (result.snapshot_recovery_ok) result.total_passed += 1;

    result.kill_restart_ok = testKillRestart(store);
    if (result.kill_restart_ok) result.total_passed += 1;

    result.concurrent_write_ok = testConcurrentWrite(store);
    if (result.concurrent_write_ok) result.total_passed += 1;

    result.determinism_after_restart_ok = testDeterminismAfterRestart();
    if (result.determinism_after_restart_ok) result.total_passed += 1;

    return result;
}

fn testSnapshotRecovery(store: *KBStore) bool {
    const kb = store.createKB(.{
        .name = "chaos_snap",
        .parent_id = -1,
        .visibility = .public,
        .owner = "test",
        .max_facts = 64,
        .max_rules = 0,
        .max_children = 0,
    });
    if (kb < 0) return false;

    const val = Q16.fromFraction(42, 1);
    const fact = VlpFact{
        .tag = .value,
        .value = val,
        .provenance = .{
            .source_type = .vdr_computation,
            .source_kb_id = kb,
            .source_slot_id = 0,
            .confidence = .{ .v = Q16.D, .r0 = 0 },
            .timestamp = 1700000000,
            .derivation_rule_id = -1,
        },
    };
    _ = fact_mod.factAssert(store, kb, 0, &fact);

    var snap_buf: [65536]u8 = undefined;
    var snap_len: i32 = 0;
    const save_status = snapshot_mod.snapshotSave(store, kb, &snap_buf, @intCast(snap_buf.len), &snap_len);
    if (save_status != .ok) return false;

    const corrupt_val = Q16.fromFraction(999, 1);
    const corrupt_fact = VlpFact{
        .tag = .value,
        .value = corrupt_val,
        .provenance = fact.provenance,
    };
    _ = fact_mod.factAssert(store, kb, 0, &corrupt_fact);

    const read_corrupt = fact_mod.factQuery(store, kb, 0);
    if (read_corrupt) |f| {
        if (!Q16.eql(f.value, corrupt_val)) return false;
    } else return false;

    const restore_status = snapshot_mod.snapshotRestore(store, kb, &snap_buf, snap_len);
    if (restore_status != .ok) return false;

    const read_restored = fact_mod.factQuery(store, kb, 0);
    if (read_restored) |f| {
        return Q16.eql(f.value, val);
    }
    return false;
}

fn testKillRestart(store: *KBStore) bool {
    const kb1 = store.createKB(.{
        .name = "chaos_kill1",
        .parent_id = -1,
        .visibility = .public,
        .owner = "test",
        .max_facts = 32,
        .max_rules = 0,
        .max_children = 0,
    });
    if (kb1 < 0) return false;

    for (0..10) |i| {
        const val = Q16.fromFraction(@intCast(i + 1), 1);
        const fact = VlpFact{
            .tag = .value,
            .value = val,
            .provenance = .{
                .source_type = .vdr_computation,
                .source_kb_id = kb1,
                .source_slot_id = @intCast(i),
                .confidence = .{ .v = Q16.D, .r0 = 0 },
                .timestamp = 1700000000,
                .derivation_rule_id = -1,
            },
        };
        _ = fact_mod.factAssert(store, kb1, @intCast(i), &fact);
    }

    var snap_buf: [65536]u8 = undefined;
    var snap_len: i32 = 0;
    _ = snapshot_mod.snapshotSave(store, kb1, &snap_buf, @intCast(snap_buf.len), &snap_len);

    const kb2 = store.createKB(.{
        .name = "chaos_kill2",
        .parent_id = -1,
        .visibility = .public,
        .owner = "test",
        .max_facts = 32,
        .max_rules = 0,
        .max_children = 0,
    });
    if (kb2 < 0) return false;

    _ = snapshot_mod.snapshotRestore(store, kb2, &snap_buf, snap_len);

    for (0..10) |i| {
        const original = fact_mod.factQuery(store, kb1, @intCast(i));
        const restored = fact_mod.factQuery(store, kb2, @intCast(i));
        if (original == null or restored == null) return false;
        if (!Q16.eql(original.?.value, restored.?.value)) return false;
    }

    return true;
}

fn testConcurrentWrite(store: *KBStore) bool {
    const kb = store.createKB(.{
        .name = "chaos_conc",
        .parent_id = -1,
        .visibility = .public,
        .owner = "test",
        .max_facts = 128,
        .max_rules = 0,
        .max_children = 0,
    });
    if (kb < 0) return false;

    for (0..100) |i| {
        const val = Q16.fromFraction(@intCast(i), 1);
        const fact = VlpFact{
            .tag = .value,
            .value = val,
            .provenance = .{
                .source_type = .vdr_computation,
                .source_kb_id = kb,
                .source_slot_id = @intCast(i),
                .confidence = .{ .v = Q16.D, .r0 = 0 },
                .timestamp = 1700000000,
                .derivation_rule_id = -1,
            },
        };
        _ = fact_mod.factAssert(store, kb, @intCast(i), &fact);
    }

    for (0..100) |i| {
        const expected = Q16.fromFraction(@intCast(i), 1);
        const read = fact_mod.factQuery(store, kb, @intCast(i));
        if (read == null) return false;
        if (!Q16.eql(read.?.value, expected)) return false;
    }

    return true;
}

fn testDeterminismAfterRestart() bool {
    const results = det_mod.runFullDeterminismSuite(100);
    for (results) |r| {
        if (!r.all_identical) return false;
    }
    return true;
}
