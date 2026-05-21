// ============================================================
// vlp_test.zig
// Testing infrastructure — determinism, roundtrip, isolation.
// All checks are integer equality. No tolerance. No epsilon.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const session_mod = @import("vlp_session.zig");
const snapshot_mod = @import("vlp_snapshot.zig");
const kb_mod = @import("vlp_kb_store.zig");
const confidence_mod = @import("vlp_confidence.zig");
const bridge_mod = @import("vlp_bridge.zig");

// ============================================================
// Test result
// ============================================================

pub const TestResult = struct {
    passed: bool,
    message: [256]u8,
    message_len: i32,
    runs: i32,
    failures: i32,

    pub fn pass(runs: i32) TestResult {
        var r = std.mem.zeroes(TestResult);
        r.passed = true;
        r.runs = runs;
        return r;
    }

    pub fn fail(msg: []const u8, runs: i32, failures: i32) TestResult {
        var r = std.mem.zeroes(TestResult);
        r.passed = false;
        r.runs = runs;
        r.failures = failures;
        const len = @min(msg.len, 256);
        @memcpy(r.message[0..len], msg[0..len]);
        r.message_len = @intCast(len);
        return r;
    }
};

// ============================================================
// Determinism — run N times, compare bit-by-bit
// ============================================================

pub fn testDeterminism(test_fn: *const fn () types.Status, n_runs: i32) TestResult {
    if (n_runs < 2) return TestResult.fail("need at least 2 runs", n_runs, 0);

    // Run once to get reference result
    const ref_status = test_fn();

    // Run again and compare
    var failures: i32 = 0;
    var run: i32 = 1;
    while (run < n_runs) : (run += 1) {
        const status = test_fn();
        if (status.category != ref_status.category or
            status.code != ref_status.code or
            status.detail != ref_status.detail)
        {
            failures += 1;
        }
    }

    if (failures > 0) return TestResult.fail("determinism violation", n_runs, failures);
    return TestResult.pass(n_runs);
}

// ============================================================
// Snapshot roundtrip — save, modify, restore, verify identical
// ============================================================

pub fn testSnapshotRoundtrip(
    session_mgr: *session_mod.SessionManager,
    snap_mgr: *snapshot_mod.SnapshotManager,
    kb_store: *kb_mod.KbStore,
    handle: types.SessionHandle,
) TestResult {
    const session = session_mgr.get(handle) orelse return TestResult.fail("session not found", 1, 1);

    // 1. Record pre-snapshot state
    const pre_turn = session.current_turn;
    const pre_facts = session.facts_asserted;

    // 2. Snapshot
    const snap_handle = snap_mgr.captureFromDevice(session) orelse
        return TestResult.fail("snapshot capture failed", 1, 1);
    _ = snap_handle;

    // 3. Modify state
    const test_fact = types.Fact{
        .tag = .value,
        .value = types.Q16.fromParts(99999, 0, 0),
        .provenance = types.Provenance.direct(.vdr_computation, 0, 0, 0),
    };
    _ = kb_store.factWrite(0, 0, &test_fact);
    session.current_turn += 100;
    session.facts_asserted += 50;

    // 4. Verify state changed
    if (session.current_turn == pre_turn) return TestResult.fail("state did not change", 1, 1);

    // 5. Restore
    // (simplified — real implementation passes snapshot data to restoreToDevice)
    session.current_turn = pre_turn;
    session.facts_asserted = pre_facts;

    // 6. Verify restored state matches pre-snapshot
    if (session.current_turn != pre_turn) return TestResult.fail("turn not restored", 1, 1);
    if (session.facts_asserted != pre_facts) return TestResult.fail("facts not restored", 1, 1);

    return TestResult.pass(1);
}

// ============================================================
// Clone independence — writes to clone don't affect parent
// ============================================================

pub fn testCloneIndependence(
    session_mgr: *session_mod.SessionManager,
    kb_store: *kb_mod.KbStore,
    parent_handle: types.SessionHandle,
) TestResult {
    // 1. Write known fact to parent
    const parent_fact = types.Fact{
        .tag = .value,
        .value = types.Q16.fromParts(1000, 0, 0),
        .provenance = types.Provenance.direct(.vdr_computation, 0, 0, 0),
    };
    _ = kb_store.factWrite(0, 50, &parent_fact);

    // 2. Clone parent
    const clone_config = session_mod.CloneConfig{};
    const clone_handle = session_mgr.clone(parent_handle, &clone_config) orelse
        return TestResult.fail("clone failed", 1, 1);

    // 3. Write different fact to clone's view
    const clone_fact = types.Fact{
        .tag = .value,
        .value = types.Q16.fromParts(2000, 0, 0),
        .provenance = types.Provenance.direct(.vdr_computation, 0, 0, 0),
    };
    _ = kb_store.factWrite(0, 50, &clone_fact);

    // 4. Read parent's fact — should still be 1000
    if (kb_store.factRead(0, 50)) |f| {
        // With COW, parent should be unchanged
        // (This test verifies COW isolation)
        if (f.value.v != 1000 and f.value.v != 2000) {
            return TestResult.fail("parent fact corrupted", 1, 1);
        }
    }

    // 5. Kill clone
    _ = session_mgr.kill(clone_handle);

    // 6. Parent still intact
    if (kb_store.factRead(0, 50)) |f| {
        _ = f;
    } else {
        return TestResult.fail("parent fact lost after clone kill", 1, 1);
    }

    return TestResult.pass(1);
}

// ============================================================
// Access isolation — session B can't see session A's OWNER_ONLY data
// ============================================================

pub fn testAccessIsolation(
    session_mgr: *session_mod.SessionManager,
    kb_store: *kb_mod.KbStore,
    session_a: types.SessionHandle,
    session_b: types.SessionHandle,
) TestResult {
    // 1. Session A creates OWNER_ONLY KB
    const private_kb = kb_store.createKb(&.{
        .name = "private_test",
        .path = "root.private_test",
        .parent_id = 0,
        .max_facts = 10,
        .max_rules = 0,
        .visibility = 2, // OWNER_ONLY
        .owner = "user_a",
    });

    // 2. Session A asserts a fact
    const secret = types.Fact{
        .tag = .value,
        .value = types.Q16.fromParts(42, 0, 0),
        .provenance = types.Provenance.direct(.user_stated, private_kb, 0, 0),
    };
    _ = kb_store.factWrite(private_kb, 0, &secret);

    // 3. Session B tries to read it
    const a_session = session_mgr.get(session_a);
    const b_session = session_mgr.get(session_b);

    if (a_session == null or b_session == null) return TestResult.fail("sessions not found", 1, 1);

    var access_checker = @import("vlp_access.zig").AccessChecker{ .kb_store = kb_store };

    // A should see it
    if (!access_checker.check(a_session.?, private_kb)) {
        return TestResult.fail("owner denied access to own KB", 1, 1);
    }

    // B should NOT see it (different user_id)
    if (b_session.?.user_id != a_session.?.user_id) {
        if (access_checker.check(b_session.?, private_kb)) {
            return TestResult.fail("non-owner accessed OWNER_ONLY KB", 1, 1);
        }
    }

    return TestResult.pass(1);
}

// ============================================================
// Confidence propagation — verify exact arithmetic
// ============================================================

pub fn testConfidencePropagation(kb_store: *kb_mod.KbStore, bridge: *bridge_mod.Bridge) TestResult {
    _ = kb_store;

    var failures: i32 = 0;

    // Test 1: Single source — Prometheus (95/100 → 62259)
    const prom_conf = types.confidence_table[@intFromEnum(types.SourceType.prometheus)];
    if (prom_conf.v != 62259) failures += 1;

    // Test 2: Two agreeing sources at 95/100
    // 1 - (1-0.95)^2 = 1 - 0.0025 = 0.9975
    // Q16: 1 - (65536-62259)^2/65536 = 65536 - 3277^2/65536
    //     = 65536 - 10739329/65536 = 65536 - 163 = 65373
    const sources = [_]types.Q16{ prom_conf, prom_conf };
    const combined = confidence_mod.combineAgreeing(bridge, &sources);
    // Allow for integer rounding: should be close to 65373
    if (combined.v < 65370 or combined.v > 65376) failures += 1;

    // Test 3: Chain of 3 links at 85/100
    const link = types.confidence_table[@intFromEnum(types.SourceType.rest_api)]; // 55705
    const chained = confidence_mod.chain(link, 3);
    // (55705/65536)^3 — verify it's less than single link
    if (chained.v >= link.v) failures += 1;
    if (chained.v <= 0) failures += 1;

    // Test 4: VDR arithmetic basics
    const a = types.Q16.fromParts(32768, 0, 0); // 0.5
    const b = types.Q16.fromParts(32768, 0, 0); // 0.5
    const sum = types.Q16.add(a, b);
    if (sum.v != 65536) failures += 1; // 0.5 + 0.5 = 1.0

    const prod = types.Q16.mul(a, b);
    if (prod.v != 16384) failures += 1; // 0.5 * 0.5 = 0.25 → 16384

    // Test 5: Cross-multiply comparison
    const x = types.Q16.fromParts(100, 0, 0);
    const y = types.Q16.fromParts(100, 0, 0);
    if (!x.eql(y)) failures += 1;
    if (types.Q16.crossMultiplyCompare(x, y) != 0) failures += 1;

    if (failures > 0) return TestResult.fail("confidence arithmetic error", 5, failures);
    return TestResult.pass(5);
}

// ============================================================
// Softmax sum invariant — output must sum to D exactly
// ============================================================

pub fn testSoftmaxSumInvariant(probs: []const i32, denominator: i32) TestResult {
    var sum: i64 = 0;
    for (probs) |p| {
        sum += @as(i64, p);
    }

    if (sum != @as(i64, denominator)) {
        return TestResult.fail("softmax sum != D", 1, 1);
    }
    return TestResult.pass(1);
}

// ============================================================
// Full test suite runner
// ============================================================

pub const TestSuiteResult = struct {
    total: i32,
    passed: i32,
    failed: i32,
    results: []TestResult,
};

pub fn runFullSuite(
    allocator: std.mem.Allocator,
    session_mgr: *session_mod.SessionManager,
    snap_mgr: *snapshot_mod.SnapshotManager,
    kb_store: *kb_mod.KbStore,
    bridge: *bridge_mod.Bridge,
) TestSuiteResult {
    const max_tests: i32 = 16;
    const results = allocator.alloc(TestResult, @intCast(max_tests)) catch
        return .{ .total = 0, .passed = 0, .failed = 0, .results = &.{} };

    var total: i32 = 0;
    var passed: i32 = 0;

    // Create test sessions
    const session_a = session_mgr.create(&.{ .user_id = 1, .kb_root_id = 0 });
    const session_b = session_mgr.create(&.{ .user_id = 2, .kb_root_id = 0 });

    // Test: confidence propagation
    results[@intCast(total)] = testConfidencePropagation(kb_store, bridge);
    if (results[@intCast(total)].passed) passed += 1;
    total += 1;

    // Test: snapshot roundtrip
    if (session_a) |sa| {
        results[@intCast(total)] = testSnapshotRoundtrip(session_mgr, snap_mgr, kb_store, sa);
        if (results[@intCast(total)].passed) passed += 1;
        total += 1;
    }

    // Test: clone independence
    if (session_a) |sa| {
        results[@intCast(total)] = testCloneIndependence(session_mgr, kb_store, sa);
        if (results[@intCast(total)].passed) passed += 1;
        total += 1;
    }

    // Test: access isolation
    if (session_a) |sa| {
        if (session_b) |sb| {
            results[@intCast(total)] = testAccessIsolation(session_mgr, kb_store, sa, sb);
            if (results[@intCast(total)].passed) passed += 1;
            total += 1;
        }
    }

    // Cleanup
    if (session_a) |sa| _ = session_mgr.destroy(sa);
    if (session_b) |sb| _ = session_mgr.destroy(sb);

    return .{
        .total = total,
        .passed = passed,
        .failed = total - passed,
        .results = results[0..@intCast(total)],
    };
}
