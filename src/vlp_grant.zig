// ============================================================
// vlp_grant.zig
// Grant enforcer — host-side access control for operations.
// Integer checks only. No heuristics. No LLM evaluation.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const kb_mod = @import("vlp_kb_store.zig");

// ============================================================
// Grant check result
// ============================================================

pub const GrantResult = struct {
    granted: bool,
    grant_id: i32,
    remaining_uses: i32,

    pub fn allowed(id: i32, remaining: i32) GrantResult {
        return .{ .granted = true, .grant_id = id, .remaining_uses = remaining };
    }

    pub fn denied() GrantResult {
        return .{ .granted = false, .grant_id = -1, .remaining_uses = 0 };
    }
};

// ============================================================
// Grant index — (user_id, grant_class) → grant store indices
// ============================================================

const IndexEntry = struct {
    user_id: i32,
    grant_class: types.GrantClass,
    grant_idx: i32,
};

// ============================================================
// Grant Enforcer
// ============================================================

pub const GrantEnforcer = struct {
    allocator: std.mem.Allocator,
    kb_store: *kb_mod.KbStore,

    grants: []types.Grant,
    grant_count: i32,
    grant_capacity: i32,

    index: []IndexEntry,
    index_count: i32,
    index_capacity: i32,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(allocator: std.mem.Allocator, kb_store: *kb_mod.KbStore, max_grants: i32) GrantEnforcer {
    const cap: usize = @intCast(max_grants);
    const grants = allocator.alloc(types.Grant, cap) catch &.{};
    const idx_cap: usize = @intCast(max_grants * 2); // index may have multiple entries per grant
    const index = allocator.alloc(IndexEntry, idx_cap) catch &.{};

    return .{
        .allocator = allocator,
        .kb_store = kb_store,
        .grants = grants,
        .grant_count = 0,
        .grant_capacity = max_grants,
        .index = index,
        .index_count = 0,
        .index_capacity = @intCast(idx_cap),
    };
}

pub fn deinit(self: *GrantEnforcer) void {
    if (self.grants.len > 0) self.allocator.free(self.grants);
    if (self.index.len > 0) self.allocator.free(self.index);
}

// ============================================================
// Check — the critical path
// Every step is integer comparison. No heuristics.
// ============================================================

pub fn check(self: *GrantEnforcer, session: *const types.Session, grant_class: types.GrantClass, target: []const u8) GrantResult {
    const user_id = session.user_id;
    const now = currentTimestamp();

    // Scan index for matching (user_id, grant_class) pairs
    for (self.index[0..@intCast(self.index_count)]) |entry| {
        if (entry.user_id != user_id) continue;
        if (entry.grant_class != grant_class) continue;

        const idx: usize = @intCast(entry.grant_idx);
        if (idx >= self.grants.len) continue;
        var grant = &self.grants[idx];

        // Check state
        if (!grant.isActive()) continue;

        // Check expiry
        if (grant.isExpired(now)) {
            grant.state = .expired;
            continue;
        }

        // Check uses
        if (grant.isExhausted()) {
            grant.state = .exhausted;
            continue;
        }

        // Check target pattern match
        if (!self.matchTarget(grant, target)) continue;

        // Grant found — consume use
        _ = grant.consumeUse();

        return GrantResult.allowed(grant.id, grant.remaining_uses);
    }

    return GrantResult.denied();
}

// ============================================================
// Management
// ============================================================

pub fn create(self: *GrantEnforcer, admin_session: *const types.Session, grant: *const types.Grant) types.Status {
    // Verify admin has GRANT_ADMIN meta-privilege
    // (checked by existence of a special grant with class=admin)
    if (!self.isAdmin(admin_session.user_id)) {
        return types.Status.err(.grant, .grant_admin_required, admin_session.user_id);
    }

    if (self.grant_count >= self.grant_capacity) {
        return types.Status.err(.grant, .grant_exhausted, 0);
    }

    const idx: usize = @intCast(self.grant_count);
    self.grants[idx] = grant.*;
    self.grants[idx].id = self.grant_count;
    self.grants[idx].state = .active;
    self.grants[idx].created_at = currentTimestamp();
    self.grants[idx].created_by = admin_session.user_id;

    // Add to index
    self.addIndexEntry(grant.holder_user_id, grant.class, self.grant_count);

    self.grant_count += 1;
    return types.Status.ok();
}

pub fn revoke(self: *GrantEnforcer, admin_session: *const types.Session, grant_id: i32) types.Status {
    if (!self.isAdmin(admin_session.user_id)) {
        return types.Status.err(.grant, .grant_admin_required, admin_session.user_id);
    }

    if (grant_id < 0 or grant_id >= self.grant_count) {
        return types.Status.err(.grant, .grant_denied, grant_id);
    }

    const idx: usize = @intCast(grant_id);
    self.grants[idx].state = .revoked;
    self.grants[idx].revoked_at = currentTimestamp();
    self.grants[idx].revoked_by = admin_session.user_id;

    return types.Status.ok();
}

pub fn list(self: *GrantEnforcer, user_id: i32, out: []types.Grant) i32 {
    var count: i32 = 0;
    for (self.grants[0..@intCast(self.grant_count)]) |g| {
        if (g.holder_user_id == user_id and count < @as(i32, @intCast(out.len))) {
            out[@intCast(count)] = g;
            count += 1;
        }
    }
    return count;
}

pub fn cleanup(self: *GrantEnforcer) i32 {
    const now = currentTimestamp();
    var cleaned: i32 = 0;
    for (self.grants[0..@intCast(self.grant_count)]) |*g| {
        if (g.isActive()) {
            if (g.isExpired(now)) {
                g.state = .expired;
                cleaned += 1;
            } else if (g.isExhausted()) {
                g.state = .exhausted;
                cleaned += 1;
            }
        }
    }
    return cleaned;
}

// ============================================================
// Target pattern matching — prefix match on integer-indexed text
// ============================================================

fn matchTarget(self: *GrantEnforcer, grant: *const types.Grant, target: []const u8) bool {
    if (grant.target_pattern_length <= 0) return true; // empty pattern matches all

    // Read pattern from text store
    var pattern_buf: [512]u8 = undefined;
    const plen: usize = @intCast(@min(grant.target_pattern_length, 512));
    _ = self.kb_store.textRead(grant.target_pattern_offset, @intCast(plen), pattern_buf[0..plen]);

    // Wildcard "*" matches everything
    if (plen == 1 and pattern_buf[0] == '*') return true;

    // Prefix match: pattern "root.ops.*" matches "root.ops.metrics"
    if (plen > 0 and pattern_buf[plen - 1] == '*') {
        const prefix = pattern_buf[0 .. plen - 1];
        if (target.len < prefix.len) return false;
        return std.mem.eql(u8, target[0..prefix.len], prefix);
    }

    // Exact match
    if (target.len != plen) return false;
    return std.mem.eql(u8, target, pattern_buf[0..plen]);
}

// ============================================================
// Helpers
// ============================================================

fn addIndexEntry(self: *GrantEnforcer, user_id: i32, class: types.GrantClass, grant_idx: i32) void {
    if (self.index_count >= self.index_capacity) return;
    const idx: usize = @intCast(self.index_count);
    self.index[idx] = .{
        .user_id = user_id,
        .grant_class = class,
        .grant_idx = grant_idx,
    };
    self.index_count += 1;
}

fn isAdmin(self: *GrantEnforcer, user_id: i32) bool {
    // Check for meta-grant: any active grant with class filesystem
    // and user_id matching and target "*" (admin convention)
    // Real implementation would have a dedicated admin flag
    for (self.grants[0..@intCast(self.grant_count)]) |g| {
        if (g.holder_user_id == user_id and g.isActive()) {
            if (g.target_pattern_length == 1) {
                // Could be "*" — admin grant
                var buf: [1]u8 = undefined;
                _ = self.kb_store.textRead(g.target_pattern_offset, 1, &buf);
                if (buf[0] == '*') return true;
            }
        }
    }
    return false;
}

fn currentTimestamp() i32 {
    const ts = std.time.timestamp();
    return @intCast(@min(ts, std.math.maxInt(i32)));
}
