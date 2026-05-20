// ============================================================
// src/test_scenarios/determinism_tests.zig
// ============================================================

const snapshot_mod = @import("../session/snapshot.zig");
const session_mod = @import("../session/lifecycle.zig");
const primitives_lru = @import("../primitives/lru.zig");
const primitives_counter = @import("../primitives/counter.zig");
const primitives_queue = @import("../primitives/queue.zig");
const primitives_ring = @import("../primitives/ring.zig");
const primitives_bitset = @import("../primitives/bitset.zig");
const collections_mod = @import("../builtins/collections.zig");
const sets_mod = @import("../builtins/sets.zig");
const arith_mod = @import("../builtins/arithmetic.zig");
const linalg_mod = @import("../builtins/linalg.zig");
const stats_mod = @import("../builtins/stats.zig");
const graph_mod = @import("../builtins/graph.zig");

pub const DeterminismResult = struct {
    q16_arithmetic_ok: bool,
    softmax_ok: bool,
    collections_ok: bool,
    sets_ok: bool,
    linalg_ok: bool,
    stats_ok: bool,
    graph_ok: bool,
    snapshot_roundtrip_ok: bool,
    kb_fact_roundtrip_ok: bool,
    confidence_ok: bool,
    total_runs: i32,
    total_mismatches: i32,
};

fn memEqlSlice(comptime T: type, a: []const T, b: []const T) bool {
    if (a.len != b.len) return false;
    const byte_a = std.mem.sliceAsBytes(a);
    const byte_b = std.mem.sliceAsBytes(b);
    return std.mem.eql(u8, byte_a, byte_b);
}

pub fn runDeterminismTests(store: *KBStore) DeterminismResult {
    var result = DeterminismResult{
        .q16_arithmetic_ok = true,
        .softmax_ok = true,
        .collections_ok = true,
        .sets_ok = true,
        .linalg_ok = true,
        .stats_ok = true,
        .graph_ok = true,
        .snapshot_roundtrip_ok = true,
        .kb_fact_roundtrip_ok = true,
        .confidence_ok = true,
        .total_runs = 0,
        .total_mismatches = 0,
    };

    const N_RUNS: usize = 100;

    var ref_add: [16]Q16 = undefined;
    var ref_mul: [16]Q16 = undefined;
    var ref_div: [16]Q16 = undefined;
    const test_vals = [_]Q16{
        Q16.fromFraction(1, 3),
        Q16.fromFraction(2, 7),
        Q16.fromFraction(-5, 11),
        Q16.fromFraction(100, 1),
        Q16.fromFraction(0, 1),
        Q16.fromFraction(1, 65536),
        Q16.fromFraction(-1, 1),
        Q16.fromFraction(32767, 1),
    };

    for (0..8) |i| {
        const a = test_vals[i];
        const b = test_vals[(i + 1) % 8];
        ref_add[i] = Q16.add(a, b);
        ref_mul[i] = Q16.mul(a, b);
        ref_div[i] = if (b.v != 0) Q16.div(a, b) else Q16.zero();
    }

    for (0..N_RUNS) |_| {
        result.total_runs += 1;
        for (0..8) |i| {
            const a = test_vals[i];
            const b = test_vals[(i + 1) % 8];
            const add_r = Q16.add(a, b);
            const mul_r = Q16.mul(a, b);
            const div_r = if (b.v != 0) Q16.div(a, b) else Q16.zero();
            if (!Q16.eql(add_r, ref_add[i]) or !Q16.eql(mul_r, ref_mul[i]) or !Q16.eql(div_r, ref_div[i])) {
                result.q16_arithmetic_ok = false;
                result.total_mismatches += 1;
            }
        }
    }

    var softmax_input = [_]Q16{
        .{ .v = 100, .r0 = 0 },
        .{ .v = 200, .r0 = 0 },
        .{ .v = 50, .r0 = 0 },
        .{ .v = 300, .r0 = 0 },
        .{ .v = 150, .r0 = 0 },
    };
    var ref_softmax: [5]Q16 = undefined;
    Q16.softmax(&softmax_input, &ref_softmax);

    var ref_sum: i64 = 0;
    for (ref_softmax) |v| ref_sum += @intCast(v.v);
    if (ref_sum != @as(i64, Q16.D)) {
        result.softmax_ok = false;
        result.total_mismatches += 1;
    }

    for (0..N_RUNS) |_| {
        result.total_runs += 1;
        var run_softmax: [5]Q16 = undefined;
        Q16.softmax(&softmax_input, &run_softmax);
        if (!memEqlSlice(Q16, &run_softmax, &ref_softmax)) {
            result.softmax_ok = false;
            result.total_mismatches += 1;
        }
        var run_sum: i64 = 0;
        for (run_softmax) |v| run_sum += @intCast(v.v);
        if (run_sum != @as(i64, Q16.D)) {
            result.softmax_ok = false;
            result.total_mismatches += 1;
        }
    }

    var sort_data = [_]Q16{
        .{ .v = 500, .r0 = 0 },
        .{ .v = 100, .r0 = 0 },
        .{ .v = 300, .r0 = 0 },
        .{ .v = 200, .r0 = 0 },
        .{ .v = 400, .r0 = 0 },
    };
    var ref_sorted: [5]Q16 = undefined;
    @memcpy(&ref_sorted, &sort_data);
    collections_mod.collSort(&ref_sorted);

    for (0..N_RUNS) |_| {
        result.total_runs += 1;
        var run_data: [5]Q16 = undefined;
        @memcpy(&run_data, &sort_data);
        collections_mod.collSort(&run_data);
        if (!memEqlSlice(Q16, &run_data, &ref_sorted)) {
            result.collections_ok = false;
            result.total_mismatches += 1;
        }
    }

    var set_a = [_]Q16{ .{ .v = 100, .r0 = 0 }, .{ .v = 200, .r0 = 0 }, .{ .v = 300, .r0 = 0 } };
    var set_b = [_]Q16{ .{ .v = 200, .r0 = 0 }, .{ .v = 300, .r0 = 0 }, .{ .v = 400, .r0 = 0 } };
    var ref_union: [6]Q16 = undefined;
    const ref_union_n = sets_mod.setUnion(&set_a, &set_b, &ref_union);

    for (0..N_RUNS) |_| {
        result.total_runs += 1;
        var run_union: [6]Q16 = undefined;
        const run_n = sets_mod.setUnion(&set_a, &set_b, &run_union);
        if (run_n != ref_union_n or !memEqlSlice(Q16, run_union[0..@intCast(run_n)], ref_union[0..@intCast(ref_union_n)])) {
            result.sets_ok = false;
            result.total_mismatches += 1;
        }
    }

    var mat_a = [_]Q16{
        .{ .v = Q16.D, .r0 = 0 },     .{ .v = Q16.D * 2, .r0 = 0 },
        .{ .v = Q16.D * 3, .r0 = 0 }, .{ .v = Q16.D * 4, .r0 = 0 },
    };
    var ref_det = linalg_mod.determinant(&mat_a, 2);

    for (0..N_RUNS) |_| {
        result.total_runs += 1;
        const run_det = linalg_mod.determinant(&mat_a, 2);
        if (!Q16.eql(run_det, ref_det)) {
            result.linalg_ok = false;
            result.total_mismatches += 1;
        }
    }

    var stats_data = [_]Q16{
        .{ .v = Q16.D * 10, .r0 = 0 },
        .{ .v = Q16.D * 20, .r0 = 0 },
        .{ .v = Q16.D * 30, .r0 = 0 },
        .{ .v = Q16.D * 40, .r0 = 0 },
        .{ .v = Q16.D * 50, .r0 = 0 },
    };
    const ref_mean = stats_mod.statsMean(&stats_data, 5);
    const ref_var = stats_mod.statsVariance(&stats_data, 5);

    for (0..N_RUNS) |_| {
        result.total_runs += 1;
        const run_mean = stats_mod.statsMean(&stats_data, 5);
        const run_var = stats_mod.statsVariance(&stats_data, 5);
        if (!Q16.eql(run_mean, ref_mean) or !Q16.eql(run_var, ref_var)) {
            result.stats_ok = false;
            result.total_mismatches += 1;
        }
    }

    var g_nodes: [8]i32 = undefined;
    var g_edges: [16]graph_mod.Edge = undefined;
    var g = graph_mod.Graph.init(&g_nodes, &g_edges);
    _ = g.addNode(1);
    _ = g.addNode(2);
    _ = g.addNode(3);
    _ = g.addEdge(1, 2, .{ .v = Q16.D, .r0 = 0 });
    _ = g.addEdge(2, 3, .{ .v = Q16.D * 2, .r0 = 0 });
    _ = g.addEdge(1, 3, .{ .v = Q16.D * 5, .r0 = 0 });

    var ref_path: [8]i32 = undefined;
    var ref_dist: Q16 = undefined;
    const ref_path_len = graph_mod.shortestPath(&g, 1, 3, &ref_path, &ref_dist);

    for (0..N_RUNS) |_| {
        result.total_runs += 1;
        var run_path: [8]i32 = undefined;
        var run_dist: Q16 = undefined;
        const run_path_len = graph_mod.shortestPath(&g, 1, 3, &run_path, &run_dist);
        if (run_path_len != ref_path_len or !Q16.eql(run_dist, ref_dist)) {
            result.graph_ok = false;
            result.total_mismatches += 1;
        }
        if (run_path_len == ref_path_len) {
            const rpl: usize = @intCast(run_path_len);
            if (!std.mem.eql(i32, run_path[0..rpl], ref_path[0..rpl])) {
                result.graph_ok = false;
                result.total_mismatches += 1;
            }
        }
    }

    const det_kb = store.createKB(.{
        .name = "det_test",
        .parent_id = -1,
        .visibility = .public,
        .owner = "system",
        .max_facts = 64,
        .max_rules = 0,
        .max_children = 0,
    });

    const test_fact = VlpFact{
        .tag = .value,
        .value = Q16.fromFraction(355, 113),
        .provenance = .{
            .source_type = .vdr_computation,
            .source_kb_id = det_kb,
            .source_slot_id = 0,
            .confidence = .{ .v = Q16.D, .r0 = 0 },
            .timestamp = 1700000000,
            .derivation_rule_id = -1,
        },
    };
    _ = fact_mod.factAssert(store, det_kb, 0, &test_fact);

    for (0..N_RUNS) |_| {
        result.total_runs += 1;
        const read = fact_mod.factQuery(store, det_kb, 0);
        if (read) |f| {
            if (!Q16.eql(f.value, test_fact.value) or f.tag != test_fact.tag) {
                result.kb_fact_roundtrip_ok = false;
                result.total_mismatches += 1;
            }
        } else {
            result.kb_fact_roundtrip_ok = false;
            result.total_mismatches += 1;
        }
    }

    const conf_table = confidence_mod.CONFIDENCE_TABLE;
    var agree_sources = [_]Q16{ conf_table[3], conf_table[3] };
    var ref_agree: Q16 = undefined;
    _ = confidence_mod.combineAgreeing(&agree_sources, 2, &ref_agree);

    for (0..N_RUNS) |_| {
        result.total_runs += 1;
        var run_agree: Q16 = undefined;
        _ = confidence_mod.combineAgreeing(&agree_sources, 2, &run_agree);
        if (!Q16.eql(run_agree, ref_agree)) {
            result.confidence_ok = false;
            result.total_mismatches += 1;
        }
    }

    const ref_chain = confidence_mod.chain(conf_table[5], 4);
    for (0..N_RUNS) |_| {
        result.total_runs += 1;
        const run_chain = confidence_mod.chain(conf_table[5], 4);
        if (!Q16.eql(run_chain, ref_chain)) {
            result.confidence_ok = false;
            result.total_mismatches += 1;
        }
    }

    return result;
}

pub fn runFullDeterminismSuite(store: *KBStore) DeterminismResult {
    return runDeterminismTests(store);
}
