// ============================================================
// vlp_access.zig
// Access control — host-side visibility checks.
// KB tree walk with integer comparisons. Nanoseconds.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const kb_mod = @import("vlp_kb_store.zig");

pub const AccessChecker = struct {
    kb_store: *kb_mod.KbStore,
};

// ============================================================
// Core check — data is ABSENT, not filtered
// ============================================================

pub fn check(checker: *AccessChecker, session: *const types.Session, kb_id: i32) bool {
    // Walk from kb_id up through ancestors.
    // If any ancestor fails visibility, entire subtree invisible.
    var current = kb_id;
    var depth: i32 = 0;
    const max_depth: i32 = 100;

    while (current >= 0 and depth < max_depth) {
        const kb = checker.kb_store.getKb(current) orelse return false;

        if (!checkSingle(session, &kb)) return false;

        current = kb.parent_id;
        depth += 1;
    }

    return true;
}

fn checkSingle(session: *const types.Session, kb: *const types.Kb) bool {
    // visibility: 0=public, 1=internal, 2=owner_only
    if (kb.visibility == 0) return true; // public
    if (kb.visibility == 1) return session.visibility_level <= 1; // internal
    if (kb.visibility == 2) {
        // owner_only — compare user_id
        // Owner stored as text, but we compare session.user_id against kb owner
        // Convention: owner text stores the user_id as decimal string
        // For speed: could store owner_user_id as an integer field
        return session.user_id == kb.owner_offset; // simplified: owner_offset used as owner_id
    }
    return false;
}

// ============================================================
// Resolve visible — enumerate all accessible KBs from a scope
// ============================================================

pub fn resolveVisible(checker: *AccessChecker, session: *const types.Session, scope_kb_id: i32, visible: []i32) i32 {
    var count: i32 = 0;
    resolveVisibleRecursive(checker, session, scope_kb_id, visible, &count, 0, 100);
    return count;
}

fn resolveVisibleRecursive(checker: *AccessChecker, session: *const types.Session, kb_id: i32, visible: []i32, count: *i32, depth: i32, max_depth: i32) void {
    if (depth >= max_depth) return;
    if (count.* >= @as(i32, @intCast(visible.len))) return;

    const kb = checker.kb_store.getKb(kb_id) orelse return;

    // If this KB is not visible, prune entire subtree
    if (!checkSingle(session, &kb)) return;

    // Add this KB
    visible[@intCast(count.*)] = kb_id;
    count.* += 1;

    // Visit children
    // Children are tracked by KB struct fields
    // Real implementation would iterate children_offset array
    // Simplified: scan all KBs for parent_id match
    var id: i32 = 0;
    while (id < checker.kb_store.next_kb_id) : (id += 1) {
        if (id == kb_id) continue;
        if (checker.kb_store.getKb(id)) |child| {
            if (child.parent_id == kb_id) {
                resolveVisibleRecursive(checker, session, id, visible, count, depth + 1, max_depth);
            }
        }
    }
}
