// ============================================================
// src/confidence/propagate.zig
// ============================================================

const conf_types = @import("types.zig");
const q16_mod = @import("../vdr/q16.zig");
const vdr_types = @import("../vdr/types.zig");
const kb_types = @import("../kb/types.zig");
const store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");

const Q16 = q16_mod.Q16;
const VlpSourceType = vdr_types.VlpSourceType;
const VlpFact = kb_types.VlpFact;
const KBStore = store_mod.KBStore;

pub fn assignFromSource(source_type: VlpSourceType) Q16 {
    const idx: usize = @intCast(@intFromEnum(source_type));
    if (idx >= conf_types.CONFIDENCE_TABLE.len) return Q16.zero();
    return conf_types.CONFIDENCE_TABLE[idx];
}

pub fn combineAgreeing(confidences: []const Q16) Q16 {
    if (confidences.len == 0) return Q16.zero();
    if (confidences.len == 1) return confidences[0];
    var product: i64 = Q16.D_i64;
    for (confidences) |c| {
        const comp: i64 = Q16.D_i64 - @as(i64, c.v);
        product = @divTrunc(product * comp, Q16.D_i64);
    }
    return .{ .v = @intCast(Q16.D_i64 - product), .r0 = 0 };
}

pub fn combineConflicting(confidences: []const Q16, penalty: Q16) Q16 {
    if (confidences.len == 0) return Q16.zero();
    var buf: [32]Q16 = undefined;
    const n = @min(confidences.len, 32);
    for (0..n) |i| {
        buf[i] = Q16.mul(confidences[i], penalty);
    }
    return combineAgreeing(buf[0..n]);
}

pub fn chain(per_link: Q16, n_links: i32) Q16 {
    if (n_links <= 0) return Q16.one();
    if (n_links == 1) return per_link;
    var r = per_link;
    var i: i32 = 1;
    while (i < n_links) : (i += 1) {
        r = Q16.mul(r, per_link);
    }
    return r;
}

pub fn propagate(store: *const KBStore, kb_id: i32, slot_id: i32) Q16 {
    var visited: [1024]bool = .{false} ** 1024;
    return propagateInner(store, kb_id, slot_id, &visited, 100);
}

fn propagateInner(store: *const KBStore, kb_id: i32, slot_id: i32, visited: *[1024]bool, depth: i32) Q16 {
    if (depth <= 0) return Q16.zero();
    const fact = fact_mod.query(store, kb_id, slot_id) orelse return Q16.zero();

    const flat = flatId(kb_id, slot_id);
    if (flat >= 0 and flat < 1024) {
        if (visited[@intCast(flat)]) return Q16.zero();
        visited[@intCast(flat)] = true;
    }

    if (fact.provenance.derivation_rule_id < 0) {
        return fact.provenance.confidence;
    }

    const src_kb = fact.provenance.source_kb_id;
    const src_slot = fact.provenance.source_slot_id;
    if (src_kb < 0 or src_slot < 0) return fact.provenance.confidence;

    const src_conf = propagateInner(store, src_kb, src_slot, visited, depth - 1);

    if (fact.provenance.confidence.v == Q16.D and fact.provenance.confidence.r0 == 0) {
        return src_conf;
    }
    return Q16.mul(src_conf, fact.provenance.confidence);
}

fn flatId(kb_id: i32, slot_id: i32) i32 {
    if (kb_id < 0 or slot_id < 0) return -1;
    return kb_id * 32 + slot_id;
}
