// ============================================================
// src/safety/grant.zig
// ============================================================

const std = @import("std");
const safety_types = @import("types.zig");
const text_mod = @import("../kb/text_store.zig");

const VlpGrant = safety_types.VlpGrant;
const VlpGrantClass = safety_types.VlpGrantClass;
const VlpGrantState = safety_types.VlpGrantState;
const VlpStatus = safety_types.VlpStatus;
const GrantCheckResult = safety_types.GrantCheckResult;
const TextStore = text_mod.TextStore;

pub const GrantStore = struct {
    grants: []VlpGrant,
    count: i32,
    text: *TextStore,

    pub fn init(backing: []VlpGrant, text: *TextStore) GrantStore {
        for (backing) |*g| g.* = VlpGrant{};
        return .{ .grants = backing, .count = 0, .text = text };
    }

    pub fn create(self: *GrantStore, class: VlpGrantClass, holder: i32, target: []const u8, max_uses: i32, expires_at: i32, created_by: i32, now: i32) ?i32 {
        if (self.count >= @as(i32, @intCast(self.grants.len))) return null;
        const id = self.count;
        const g = &self.grants[@intCast(id)];
        var toff: i32 = 0;
        var tlen: i16 = 0;
        if (target.len > 0) {
            const ref = self.text.append(target) orelse return null;
            toff = ref.offset;
            tlen = ref.length;
        }
        g.* = .{
            .id = id,
            .class = class,
            .state = .active,
            .holder_user_id = holder,
            .target_offset = toff,
            .target_length = tlen,
            .max_uses = max_uses,
            .remaining_uses = max_uses,
            .expires_at = expires_at,
            .created_at = now,
            .created_by = created_by,
        };
        self.count += 1;
        return id;
    }

    pub fn check(self: *GrantStore, user_id: i32, class: VlpGrantClass, target: []const u8, now: i32) GrantCheckResult {
        var i: i32 = 0;
        while (i < self.count) : (i += 1) {
            var g = &self.grants[@intCast(i)];
            if (g.state != .active) continue;
            if (g.holder_user_id != user_id) continue;
            if (g.class != class) continue;
            if (g.expires_at > 0 and now >= g.expires_at) {
                g.state = .expired;
                continue;
            }
            if (g.max_uses >= 0 and g.remaining_uses <= 0) {
                g.state = .exhausted;
                continue;
            }
            if (g.target_length > 0) {
                const pat = self.text.read(.{ .offset = g.target_offset, .length = g.target_length }) orelse continue;
                if (!prefixMatch(target, pat)) continue;
            }
            if (g.max_uses >= 0) {
                g.remaining_uses -= 1;
                if (g.remaining_uses <= 0) g.state = .exhausted;
            }
            return .{ .granted = true, .grant_id = g.id };
        }
        return .{ .granted = false, .grant_id = -1 };
    }

    pub fn revoke(self: *GrantStore, grant_id: i32, revoked_by: i32, now: i32) VlpStatus {
        if (grant_id < 0 or grant_id >= self.count) return .err_grant_denied;
        var g = &self.grants[@intCast(grant_id)];
        if (g.id < 0) return .err_grant_denied;
        g.state = .revoked;
        g.revoked_at = now;
        g.revoked_by = revoked_by;
        return .ok;
    }

    pub fn get(self: *const GrantStore, grant_id: i32) ?*const VlpGrant {
        if (grant_id < 0 or grant_id >= self.count) return null;
        const g = &self.grants[@intCast(grant_id)];
        if (g.id < 0) return null;
        return g;
    }

    pub fn list(self: *const GrantStore, user_id: i32, out: []VlpGrant) i32 {
        var found: i32 = 0;
        var i: i32 = 0;
        while (i < self.count) : (i += 1) {
            if (found >= @as(i32, @intCast(out.len))) break;
            if (self.grants[@intCast(i)].holder_user_id == user_id) {
                out[@intCast(found)] = self.grants[@intCast(i)];
                found += 1;
            }
        }
        return found;
    }

    pub fn listActive(self: *const GrantStore, user_id: i32, out: []VlpGrant) i32 {
        var found: i32 = 0;
        var i: i32 = 0;
        while (i < self.count) : (i += 1) {
            if (found >= @as(i32, @intCast(out.len))) break;
            const g = &self.grants[@intCast(i)];
            if (g.holder_user_id == user_id and g.state == .active) {
                out[@intCast(found)] = g.*;
                found += 1;
            }
        }
        return found;
    }

    pub fn cleanup(self: *GrantStore, now: i32) i32 {
        var cleaned: i32 = 0;
        var i: i32 = 0;
        while (i < self.count) : (i += 1) {
            var g = &self.grants[@intCast(i)];
            if (g.state != .active) continue;
            if (g.expires_at > 0 and now >= g.expires_at) {
                g.state = .expired;
                cleaned += 1;
            } else if (g.max_uses >= 0 and g.remaining_uses <= 0) {
                g.state = .exhausted;
                cleaned += 1;
            }
        }
        return cleaned;
    }

    pub fn activeCount(self: *const GrantStore) i32 {
        var c: i32 = 0;
        var i: i32 = 0;
        while (i < self.count) : (i += 1) {
            if (self.grants[@intCast(i)].state == .active) c += 1;
        }
        return c;
    }
};

fn prefixMatch(target: []const u8, pattern: []const u8) bool {
    if (pattern.len == 0) return true;
    if (pattern[pattern.len - 1] == '*') {
        const prefix = pattern[0 .. pattern.len - 1];
        if (target.len < prefix.len) return false;
        return std.mem.eql(u8, target[0..prefix.len], prefix);
    }
    return std.mem.eql(u8, target, pattern);
}
