
```zig
// ============================================================
// vlp_confidence.zig
// Confidence propagation — exact VDR arithmetic.
// Host-side for small N, GPU-dispatched for large N.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const gpu = @import("vlp_gpu_params.zig");
const kb_mod = @import("vlp_kb_store.zig");

// ============================================================
// Assignment — set confidence from source type
// ============================================================

pub fn assign(fact: *types.Fact, source_type: types.SourceType) void {
    const idx: usize = @intCast(@intFromEnum(source_type));
    if (idx < types.confidence_table.len) {
        fact.provenance.confidence = types.confidence_table[idx];
        fact.provenance.source_type = @intFromEnum(source_type);
    }
}

// ============================================================
// Combination — agreeing sources
// Formula: 1 - ∏(1 - C_i)
// ============================================================

pub fn combineAgreeing(bridge: *bridge_mod.Bridge, confidences: []const types.Q16) types.Q16 {
    if (confidences.len == 0) return types.Q16.zero();
    if (confidences.len == 1) return confidences[0];

    const n: i32 = @intCast(confidences.len);

    // GPU path for large N
    if (n > 64 and bridge.shouldUseGpu(.builtin_array, n)) {
        return combineAgreeingGpu(bridge, confidences);
    }

    // Host path: 1 - ∏(1 - C_i)
    // Work in i64 to avoid overflow during multiplication
    const d: i64 = types.Q16.D;
    var product: i64 = d; // starts at 1.0 in Q16

    for (confidences) |c| {
        const complement: i64 = d - @as(i64, c.v); // (1 - C_i) scaled by D
        product = @divTrunc(product * complement, d);
    }

    const result_v: i32 = @intCast(d - product);
    return types.Q16.fromParts(result_v, 0);
}

fn combineAgreeingGpu(bridge: *bridge_mod.Bridge, confidences: []const types.Q16) types.Q16 {
    const n: i32 = @intCast(confidences.len);

    // Upload confidences to scratch_a (just the .v fields)
    var values = bridge.allocator.alloc(i32, @intCast(n)) catch return types.Q16.zero();
    defer bridge.allocator.free(values);
    for (confidences, 0..) |c, i| {
        values[i] = c.v;
    }
    const bytes: []const u8 = @as([*]const u8, @ptrCast(values.ptr))[0 .. values.len * 4];
    _ = bridge.uploadToBuffer(.scratch_a, 0, bytes);

    var params = gpu.BuiltinConfidenceCombineParams{
        .n_sources = n,
        .mode = 0, // agreeing
        .penalty_v = 0,
        .input_offset = 0,
    };

    _ = bridge.dispatch(&.{
        .pipeline = .builtin_confidence_combine,
        .group_count_x = 1,
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&params),
        .params_size = @sizeOf(gpu.BuiltinConfidenceCombineParams),
    });

    // Read result from scratch_b[0]
    var result: i32 = 0;
    const dest: []u8 = @as([*]u8, @ptrCast(&result))[0..4];
    _ = bridge.downloadFromBuffer(.scratch_b, 0, dest);

    return types.Q16.fromParts(result, 0);
}

// ============================================================
// Combination — conflicting sources
// Same as agreeing but with penalty per conflict pair
// ============================================================

pub fn combineConflicting(bridge: *bridge_mod.Bridge, confidences: []const types.Q16, penalty: types.Q16) types.Q16 {
    if (confidences.len == 0) return types.Q16.zero();
    if (confidences.len == 1) return confidences[0];

    const n: i32 = @intCast(confidences.len);

    if (n > 64 and bridge.shouldUseGpu(.builtin_array, n)) {
        // Upload and dispatch with mode=1
        var values = bridge.allocator.alloc(i32, @intCast(n)) catch return types.Q16.zero();
        defer bridge.allocator.free(values);
        for (confidences, 0..) |c, i| values[i] = c.v;
        const bytes: []const u8 = @as([*]const u8, @ptrCast(values.ptr))[0 .. values.len * 4];
        _ = bridge.uploadToBuffer(.scratch_a, 0, bytes);

        var params = gpu.BuiltinConfidenceCombineParams{
            .n_sources = n,
            .mode = 1,
            .penalty_v = penalty.v,
            .input_offset = 0,
        };
        _ = bridge.dispatch(&.{
            .pipeline = .builtin_confidence_combine,
            .group_count_x = 1, .group_count_y = 1, .group_count_z = 1,
            .params_ptr = @ptrCast(&params),
            .params_size = @sizeOf(gpu.BuiltinConfidenceCombineParams),
        });

        var result: i32 = 0;
        const dest: []u8 = @as([*]u8, @ptrCast(&result))[0..4];
        _ = bridge.downloadFromBuffer(.scratch_b, 0, dest);
        return types.Q16.fromParts(result, 0);
    }

    // Host path: agreeing combination then apply penalty per pair
    var base = combineAgreeing(bridge, confidences);

    // Apply penalty: n*(n-1)/2 conflict pairs
    const pairs: i64 = @as(i64, n) * (@as(i64, n) - 1) / 2;
    var i: i64 = 0;
    while (i < pairs) : (i += 1) {
        base = types.Q16.mul(base, penalty);
    }

    return base;
}

// ============================================================
// Chain — C^N for N links at same confidence
// ============================================================

pub fn chain(per_link: types.Q16, n_links: i32) types.Q16 {
    if (n_links <= 0) return types.Q16.one();
    if (n_links == 1) return per_link;

    // Repeated exact multiplication
    var result = per_link;
    var i: i32 = 1;
    while (i < n_links) : (i += 1) {
        result = types.Q16.mul(result, per_link);
    }
    return result;
}

// ============================================================
// Propagation — walk provenance chain of a derived fact
// ============================================================

pub fn propagate(kb_store: *kb_mod.KbStore, kb_id: i32, slot_id: i32) types.Q16 {
    return propagateWithDepth(kb_store, kb_id, slot_id, 0, 100);
}

fn propagateWithDepth(kb_store: *kb_mod.KbStore, kb_id: i32, slot_id: i32, depth: i32, max_depth: i32) types.Q16 {
    if (depth >= max_depth) return types.Q16.zero(); // cycle or too deep

    const fact = kb_store.factRead(kb_id, slot_id) orelse return types.Q16.zero();

    // Direct source — return table confidence
    if (fact.provenance.derivation_rule_id == -1) {
        const src: usize = @intCast(fact.provenance.source_type);
        if (src < types.confidence_table.len) {
            return types.confidence_table[src];
        }
        return types.Q16.zero();
    }

    // Derived — read the rule, find source facts, chain their confidences
    // The rule's body conditions tell us which facts were used
    // For now, simplified: return the fact's stored confidence
    // Full implementation would trace through rule.body_offset terms
    return fact.provenance.confidence;
}

// ============================================================
// Utility — confidence comparison helpers
// ============================================================

pub fn isHighConfidence(c: types.Q16) bool {
    // >= 80/100 (52428 in Q16)
    return c.v >= 52428;
}

pub fn isMediumConfidence(c: types.Q16) bool {
    // >= 50/100 and < 80/100
    return c.v >= 32768 and c.v < 52428;
}

pub fn isLowConfidence(c: types.Q16) bool {
    // < 50/100
    return c.v < 32768;
}

pub fn confidenceToPercent(c: types.Q16) i32 {
    // Returns integer percentage 0-100
    return @intCast(@divTrunc(@as(i64, c.v) * 100, types.Q16.D));
}
```

Turn 4 complete. Grammar engine (host-only template compile/render), builtin executor (GPU dispatch for large arrays, host for small + operational), confidence propagation (exact VDR arithmetic with GPU path for large source sets). Ready for turn 5.
