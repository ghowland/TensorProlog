// ============================================================
// src/safety/audit.zig
// ============================================================

const safety_types = @import("types.zig");

const VlpAuditEntry = safety_types.VlpAuditEntry;
const VlpAuditAction = safety_types.VlpAuditAction;
const AuditFilter = safety_types.AuditFilter;

pub const AuditRing = struct {
    entries: []VlpAuditEntry,
    cap: i32,
    head: i32,
    count: i32,
    total: i64,

    pub fn init(backing: []VlpAuditEntry) AuditRing {
        for (backing) |*e| e.* = VlpAuditEntry{};
        return .{ .entries = backing, .cap = @intCast(backing.len), .head = 0, .count = 0, .total = 0 };
    }

    pub fn write(self: *AuditRing, entry: VlpAuditEntry) void {
        self.entries[@intCast(self.head)] = entry;
        self.head = @mod(self.head + 1, self.cap);
        if (self.count < self.cap) self.count += 1;
        self.total += 1;
    }

    pub fn query(self: *const AuditRing, filter: AuditFilter, out: []VlpAuditEntry) i32 {
        var found: i32 = 0;
        const max: i32 = @intCast(out.len);
        var i: i32 = 0;
        while (i < self.count) : (i += 1) {
            if (found >= max) break;
            const pos: usize = @intCast(@mod(if (self.count < self.cap) i else self.head + i, self.cap));
            const e = &self.entries[pos];
            if (matches(e, filter)) {
                out[@intCast(found)] = e.*;
                found += 1;
            }
        }
        return found;
    }

    pub fn latest(self: *const AuditRing) ?VlpAuditEntry {
        if (self.count == 0) return null;
        const pos = if (self.head == 0) self.cap - 1 else self.head - 1;
        return self.entries[@intCast(pos)];
    }

    pub fn oldest(self: *const AuditRing) ?VlpAuditEntry {
        if (self.count == 0) return null;
        if (self.count < self.cap) return self.entries[0];
        return self.entries[@intCast(self.head)];
    }

    pub fn clear(self: *AuditRing) void {
        self.head = 0;
        self.count = 0;
        self.total = 0;
    }
};

fn matches(e: *const VlpAuditEntry, f: AuditFilter) bool {
    if (f.user_id) |uid| {
        if (e.user_id != uid) return false;
    }
    if (f.action) |act| {
        if (e.action != act) return false;
    }
    if (f.after_ts) |after| {
        if (e.timestamp < after) return false;
    }
    if (f.before_ts) |before| {
        if (e.timestamp >= before) return false;
    }
    if (f.target_kb_id) |kid| {
        if (e.target_kb_id != kid) return false;
    }
    return true;
}

pub fn writeAudit(ring: *AuditRing, ts: i32, session_id: i32, user_id: i32, action: VlpAuditAction, kb_id: i32, slot_id: i32, grant_id: i32, result: i8) void {
    ring.write(.{
        .timestamp = ts,
        .session_id = session_id,
        .user_id = user_id,
        .action = action,
        .target_kb_id = kb_id,
        .target_slot_id = slot_id,
        .grant_id = grant_id,
        .result = result,
    });
}
