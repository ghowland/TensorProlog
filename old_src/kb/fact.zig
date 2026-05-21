// ============================================================
// src/kb/fact.zig
// ============================================================

const kb_types = @import("types.zig");
const store_mod = @import("store.zig");

const VlpFact = kb_types.VlpFact;
const VlpFactTag = kb_types.VlpFactTag;
const VlpStatus = kb_types.VlpStatus;
const KBStore = store_mod.KBStore;

pub fn assert(store: *KBStore, kb_id: i32, slot_id: i32, fact: *const VlpFact) VlpStatus {
    const kb = store.getKB(kb_id) orelse return .err_kb_not_found;
    if (kb.frozen) return .err_kb_frozen;
    if (slot_id < 0 or slot_id >= kb.facts_capacity) return .err_slot_out_of_range;
    const phys: usize = @intCast(kb.facts_offset + slot_id);
    store.facts[phys] = fact.*;
    return .ok;
}

pub fn query(store: *const KBStore, kb_id: i32, slot_id: i32) ?VlpFact {
    const kb = store.getKBConst(kb_id) orelse return null;
    if (slot_id < 0 or slot_id >= kb.facts_capacity) return null;
    const phys: usize = @intCast(kb.facts_offset + slot_id);
    const f = store.facts[phys];
    if (f.tag == .empty) return null;
    return f;
}

pub fn retract(store: *KBStore, kb_id: i32, slot_id: i32) VlpStatus {
    const kb = store.getKB(kb_id) orelse return .err_kb_not_found;
    if (kb.frozen) return .err_kb_frozen;
    if (slot_id < 0 or slot_id >= kb.facts_capacity) return .err_slot_out_of_range;
    const phys: usize = @intCast(kb.facts_offset + slot_id);
    store.facts[phys] = VlpFact{};
    return .ok;
}

pub fn search(store: *const KBStore, kb_id: i32, tag: VlpFactTag, results: []VlpFact) i32 {
    const kb = store.getKBConst(kb_id) orelse return 0;
    var found: i32 = 0;
    const max: i32 = @intCast(results.len);
    const s: usize = @intCast(kb.facts_offset);
    const e: usize = @intCast(kb.facts_offset + kb.facts_capacity);
    for (store.facts[s..e]) |f| {
        if (found >= max) break;
        if (f.tag == tag) {
            results[@intCast(found)] = f;
            found += 1;
        }
    }
    return found;
}

pub fn scopedSearch(store: *const KBStore, start_kb_id: i32, tag: VlpFactTag, results: []VlpFact) i32 {
    var cur = start_kb_id;
    var depth: i32 = 0;
    while (cur >= 0 and depth < 100) : (depth += 1) {
        const n = search(store, cur, tag, results);
        if (n > 0) return n;
        const kb = store.getKBConst(cur) orelse return 0;
        cur = kb.parent_id;
    }
    return 0;
}

pub fn countInKB(store: *const KBStore, kb_id: i32) i32 {
    const kb = store.getKBConst(kb_id) orelse return 0;
    var c: i32 = 0;
    const s: usize = @intCast(kb.facts_offset);
    const e: usize = @intCast(kb.facts_offset + kb.facts_capacity);
    for (store.facts[s..e]) |f| {
        if (f.tag != .empty) c += 1;
    }
    return c;
}

pub fn searchByValue(store: *const KBStore, kb_id: i32, tag: VlpFactTag, target_v: i32, results: []VlpFact) i32 {
    const kb = store.getKBConst(kb_id) orelse return 0;
    var found: i32 = 0;
    const max: i32 = @intCast(results.len);
    const s: usize = @intCast(kb.facts_offset);
    const e: usize = @intCast(kb.facts_offset + kb.facts_capacity);
    for (store.facts[s..e]) |f| {
        if (found >= max) break;
        if (f.tag == tag and f.value.v == target_v) {
            results[@intCast(found)] = f;
            found += 1;
        }
    }
    return found;
}

pub fn firstEmpty(store: *const KBStore, kb_id: i32) ?i32 {
    const kb = store.getKBConst(kb_id) orelse return null;
    const s: usize = @intCast(kb.facts_offset);
    const e: usize = @intCast(kb.facts_offset + kb.facts_capacity);
    for (store.facts[s..e], 0..) |f, i| {
        if (f.tag == .empty) return @intCast(i);
    }
    return null;
}
