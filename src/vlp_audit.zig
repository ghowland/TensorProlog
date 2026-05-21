// ============================================================
// vlp_audit.zig
// Audit log — host-side append-only ring buffer.
// Every access check, grant check, fact change, rule fire,
// session event produces an entry. No gaps.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");

// ============================================================
// Filter for audit queries
// ============================================================

pub const AuditFilter = struct {
    session_id: ?i32 = null,
    user_id: ?i32 = null,
    action: ?types.AuditAction = null,
    target_kb_id: ?i32 = null,
    min_timestamp: ?i32 = null,
    max_timestamp: ?i32 = null,
    result: ?i8 = null, // 0=denied, 1=allowed

    pub fn matchesEntry(self: *const AuditFilter, entry: *const types.AuditEntry) bool {
        if (self.session_id) |sid| {
            if (entry.session_id != sid) return false;
        }
        if (self.user_id) |uid| {
            if (entry.user_id != uid) return false;
        }
        if (self.action) |act| {
            if (entry.action != act) return false;
        }
        if (self.target_kb_id) |kid| {
            if (entry.target_kb_id != kid) return false;
        }
        if (self.min_timestamp) |min_t| {
            if (entry.timestamp < min_t) return false;
        }
        if (self.max_timestamp) |max_t| {
            if (entry.timestamp > max_t) return false;
        }
        if (self.result) |r| {
            if (entry.result != r) return false;
        }
        return true;
    }
};

// ============================================================
// Audit Log
// ============================================================

pub const AuditLog = struct {
    entries: []types.AuditEntry,
    capacity: i32,
    head: i32, // next write position
    count: i32, // total written (may exceed capacity — ring wraps)
    total_written: i64,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(allocator: std.mem.Allocator, capacity: i32) AuditLog {
    const cap: usize = @intCast(capacity);
    const entries = allocator.alloc(types.AuditEntry, cap) catch &.{};
    if (entries.len > 0) @memset(entries, std.mem.zeroes(types.AuditEntry));

    return .{
        .entries = entries,
        .capacity = capacity,
        .head = 0,
        .count = 0,
        .total_written = 0,
    };
}

pub fn deinit(log: *AuditLog, allocator: std.mem.Allocator) void {
    if (log.entries.len > 0) allocator.free(log.entries);
    log.entries = &.{};
    log.count = 0;
}

// ============================================================
// Write — append-only, ring wraps oldest
// ============================================================

pub fn write(log: *AuditLog, entry: *const types.AuditEntry) void {
    if (log.capacity <= 0) return;
    const idx: usize = @intCast(log.head);
    log.entries[idx] = entry.*;
    log.head = @mod(log.head + 1, log.capacity);
    if (log.count < log.capacity) log.count += 1;
    log.total_written += 1;
}

// ============================================================
// Convenience writers
// ============================================================

pub fn writeAllowed(log: *AuditLog, time: i32, session_id: i32, user_id: i32, action: types.AuditAction, kb_id: i32, slot_id: i32) void {
    const entry = types.AuditEntry.allowed(time, session_id, user_id, action, kb_id, slot_id);
    write(log, &entry);
}

pub fn writeDenied(log: *AuditLog, time: i32, session_id: i32, user_id: i32, action: types.AuditAction, kb_id: i32, slot_id: i32) void {
    const entry = types.AuditEntry.denied(time, session_id, user_id, action, kb_id, slot_id);
    write(log, &entry);
}

pub fn writeGrantCheck(log: *AuditLog, time: i32, session_id: i32, user_id: i32, kb_id: i32, grant_id: i32, granted: bool) void {
    var entry = types.AuditEntry.allowed(time, session_id, user_id, .grant_check, kb_id, -1);
    entry.grant_id = grant_id;
    entry.result = if (granted) 1 else 0;
    write(log, &entry);
}

// ============================================================
// Query — scan ring buffer with filter
// ============================================================

pub fn query(log: *AuditLog, filter: *const AuditFilter, out: []types.AuditEntry) i32 {
    var result_count: i32 = 0;
    const max_results: i32 = @intCast(out.len);

    // Iterate from oldest to newest
    var i: i32 = 0;
    while (i < log.count and result_count < max_results) : (i += 1) {
        const idx = ringIndex(log, i);
        const entry = &log.entries[@intCast(idx)];
        if (filter.matchesEntry(entry)) {
            out[@intCast(result_count)] = entry.*;
            result_count += 1;
        }
    }

    return result_count;
}

pub fn count(log: *AuditLog, filter: *const AuditFilter) i32 {
    var result: i32 = 0;
    var i: i32 = 0;
    while (i < log.count) : (i += 1) {
        const idx = ringIndex(log, i);
        if (filter.matchesEntry(&log.entries[@intCast(idx)])) result += 1;
    }
    return result;
}

pub fn latest(log: *AuditLog, n: i32, out: []types.AuditEntry) i32 {
    const actual = @min(n, log.count);
    const max_out: i32 = @intCast(out.len);
    const to_return = @min(actual, max_out);

    // Read from newest backwards
    var i: i32 = 0;
    while (i < to_return) : (i += 1) {
        const ring_pos = @mod(log.head - 1 - i + log.capacity, log.capacity);
        out[@intCast(i)] = log.entries[@intCast(ring_pos)];
    }

    return to_return;
}

// ============================================================
// Stats
// ============================================================

pub fn totalWritten(log: *AuditLog) i64 {
    return log.total_written;
}

pub fn currentSize(log: *AuditLog) i32 {
    return log.count;
}

pub fn isFull(log: *AuditLog) bool {
    return log.count >= log.capacity;
}

// ============================================================
// Helpers
// ============================================================

fn ringIndex(log: *AuditLog, offset_from_oldest: i32) i32 {
    // Oldest entry is at (head - count + capacity) % capacity
    if (log.count < log.capacity) {
        return offset_from_oldest; // not wrapped yet
    }
    return @mod(log.head + offset_from_oldest, log.capacity);
}
