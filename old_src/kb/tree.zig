// ============================================================
// src/kb/tree.zig
// ============================================================

const store_mod = @import("store.zig");
const KBStore = store_mod.KBStore;

pub fn getParent(store: *const KBStore, kb_id: i32) ?i32 {
    const kb = store.getKBConst(kb_id) orelse return null;
    if (kb.parent_id < 0) return null;
    if (store.getKBConst(kb.parent_id) == null) return null;
    return kb.parent_id;
}

pub fn getChildren(store: *const KBStore, kb_id: i32) []const i32 {
    return store.childrenSlice(kb_id);
}

pub fn getDepth(store: *const KBStore, kb_id: i32) i32 {
    var cur = kb_id;
    var d: i32 = 0;
    while (cur >= 0 and d < 100) : (d += 1) {
        const kb = store.getKBConst(cur) orelse break;
        cur = kb.parent_id;
    }
    return d;
}

pub fn isAncestor(store: *const KBStore, ancestor_id: i32, descendant_id: i32) bool {
    var cur = descendant_id;
    var d: i32 = 0;
    while (cur >= 0 and d < 100) : (d += 1) {
        if (cur == ancestor_id) return true;
        const kb = store.getKBConst(cur) orelse return false;
        cur = kb.parent_id;
    }
    return false;
}

pub fn getRoot(store: *const KBStore, kb_id: i32) i32 {
    var cur = kb_id;
    var d: i32 = 0;
    while (d < 100) : (d += 1) {
        const kb = store.getKBConst(cur) orelse return cur;
        if (kb.parent_id < 0) return cur;
        cur = kb.parent_id;
    }
    return cur;
}

pub fn subtreeSize(store: *const KBStore, kb_id: i32) i32 {
    if (store.getKBConst(kb_id) == null) return 0;
    var c: i32 = 1;
    for (store.childrenSlice(kb_id)) |cid| {
        c += subtreeSize(store, cid);
    }
    return c;
}

pub fn collectSubtree(store: *const KBStore, kb_id: i32, out: []i32) i32 {
    var idx: i32 = 0;
    collectInner(store, kb_id, out, &idx);
    return idx;
}

fn collectInner(store: *const KBStore, kb_id: i32, out: []i32, idx: *i32) void {
    if (store.getKBConst(kb_id) == null) return;
    if (idx.* >= @as(i32, @intCast(out.len))) return;
    out[@intCast(idx.*)] = kb_id;
    idx.* += 1;
    for (store.childrenSlice(kb_id)) |cid| {
        collectInner(store, cid, out, idx);
    }
}

pub fn siblingIds(store: *const KBStore, kb_id: i32, out: []i32) i32 {
    const kb = store.getKBConst(kb_id) orelse return 0;
    if (kb.parent_id < 0) return 0;
    const kids = store.childrenSlice(kb.parent_id);
    var c: i32 = 0;
    for (kids) |cid| {
        if (cid != kb_id and c < @as(i32, @intCast(out.len))) {
            out[@intCast(c)] = cid;
            c += 1;
        }
    }
    return c;
}
