// ============================================================
// src/kb/visibility.zig
// ============================================================

const store_mod = @import("store.zig");
const kb_types = @import("types.zig");

const KBStore = store_mod.KBStore;
const VlpVisibility = kb_types.VlpVisibility;

pub fn checkAccess(store: *const KBStore, user_id: i32, user_vis: VlpVisibility, kb_id: i32) bool {
    var cur = kb_id;
    var d: i32 = 0;
    while (cur >= 0 and d < 100) : (d += 1) {
        const kb = store.getKBConst(cur) orelse return false;
        switch (kb.visibility) {
            .public => {},
            .internal => {
                if (@intFromEnum(user_vis) > @intFromEnum(VlpVisibility.internal)) return false;
            },
            .owner_only => {
                if (kb.owner_id != user_id) return false;
            },
        }
        cur = kb.parent_id;
    }
    return true;
}

pub fn resolveVisible(store: *const KBStore, user_id: i32, user_vis: VlpVisibility, scope_id: i32, out: []i32) i32 {
    var c: i32 = 0;
    resolveInner(store, user_id, user_vis, scope_id, out, &c);
    return c;
}

fn resolveInner(store: *const KBStore, user_id: i32, user_vis: VlpVisibility, kb_id: i32, out: []i32, c: *i32) void {
    if (c.* >= @as(i32, @intCast(out.len))) return;
    if (!checkAccess(store, user_id, user_vis, kb_id)) return;
    out[@intCast(c.*)] = kb_id;
    c.* += 1;
    for (store.childrenSlice(kb_id)) |cid| {
        resolveInner(store, user_id, user_vis, cid, out, c);
    }
}

pub fn setOwner(store: *KBStore, kb_id: i32, owner_id: i32) void {
    const kb = store.getKB(kb_id) orelse return;
    kb.owner_id = owner_id;
}
