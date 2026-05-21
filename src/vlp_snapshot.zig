// ============================================================
// vlp_snapshot.zig
// Snapshot manager — host-side save/load/diff/merge.
// Captures device state to host memory. Portable binary format.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const session_mod = @import("vlp_session.zig");

// ============================================================
// Snapshot header — binary file format
// ============================================================

pub const SNAPSHOT_MAGIC = [4]u8{ 'V', 'L', 'P', 'S' };
pub const SNAPSHOT_VERSION: i32 = 1;

pub const SnapshotHeader = extern struct {
    magic: [4]u8,
    version: i32,
    timestamp: i32,
    session_id: i32,
    user_id: i32,

    kb_region_size: i64,
    fact_region_size: i64,
    rule_region_size: i64,
    term_region_size: i64,
    text_region_size: i64,
    grammar_region_size: i64,
    live_state_region_size: i64,
    grant_region_size: i64,
    path_index_region_size: i64,

    kb_count: i32,
    _pad0: i32 = 0,
    fact_count: i64,
    rule_count: i32,
    _pad1: i32 = 0,
    term_count: i64,
    grammar_count: i32,
    grant_count: i32,

    session_metadata: types.Session,

    checksum: i32,
    _pad2: i32 = 0,
    total_size: i64,
};

// ============================================================
// Diff types
// ============================================================

pub const DiffRegion = enum(i32) {
    kb = 0,
    fact = 1,
    rule = 2,
    term = 3,
    text = 4,
    grammar = 5,
    live_state = 6,
    grant = 7,
};

pub const DiffEntry = struct {
    region: DiffRegion,
    offset: i64,
    size: i64,
    a_hash: u32,
    b_hash: u32,
};

pub const DiffResult = struct {
    entries: []DiffEntry,
    count: i32,
    identical: bool,

    pub fn same() DiffResult {
        return .{ .entries = &.{}, .count = 0, .identical = true };
    }
};

// ============================================================
// Snapshot Manager
// ============================================================

pub const SnapshotManager = struct {
    allocator: std.mem.Allocator,
    bridge: *bridge_mod.Bridge,
    next_snapshot_id: i32,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(allocator: std.mem.Allocator, bridge: *bridge_mod.Bridge) SnapshotManager {
    return .{
        .allocator = allocator,
        .bridge = bridge,
        .next_snapshot_id = 0,
    };
}

pub fn deinit(self: *SnapshotManager) void {
    _ = self;
}

// ============================================================
// Capture — download device state into host memory blob
// ============================================================

pub fn captureFromDevice(self: *SnapshotManager, session: *const types.Session) ?types.SnapshotHandle {
    const layout = &self.bridge.layout;

    // Compute region sizes for this session
    // Simplified: capture all KB/fact/rule/term regions
    // Real implementation would capture only session's owned regions
    const kb_size = @as(i64, layout.kb_count) * types.KB_STRUCT_SIZE;
    const fact_size = layout.fact_store_size;
    const rule_size = layout.rule_store_size;
    const term_size = layout.term_store_size;
    const text_size = layout.text_used;
    const grammar_size = layout.grammar_store_size;
    const live_size: i64 = session.max_live_memory_bytes;
    const grant_size = layout.grant_store_size;
    const path_size: i64 = 0; // path index is host-side, serialized separately

    const total_data = kb_size + fact_size + rule_size + term_size +
        text_size + grammar_size + live_size + grant_size + path_size;
    const total = @sizeOf(SnapshotHeader) + @as(usize, @intCast(total_data));

    // Allocate host buffer
    const data = self.allocator.alloc(u8, total) catch return null;

    // Fill header
    const header: *SnapshotHeader = @ptrCast(@alignCast(data.ptr));
    header.magic = SNAPSHOT_MAGIC;
    header.version = SNAPSHOT_VERSION;
    header.timestamp = currentTimestamp();
    header.session_id = session.id;
    header.user_id = session.user_id;
    header.kb_region_size = kb_size;
    header.fact_region_size = fact_size;
    header.rule_region_size = rule_size;
    header.term_region_size = term_size;
    header.text_region_size = text_size;
    header.grammar_region_size = grammar_size;
    header.live_state_region_size = live_size;
    header.grant_region_size = grant_size;
    header.path_index_region_size = path_size;
    header.kb_count = layout.kb_count;
    header.fact_count = layout.fact_capacity;
    header.rule_count = layout.rule_capacity;
    header.term_count = layout.term_capacity;
    header.grammar_count = layout.grammar_capacity;
    header.grant_count = layout.grant_capacity;
    header.session_metadata = session.*;
    header.total_size = @intCast(total);

    // Download each region from device
    var offset: usize = @sizeOf(SnapshotHeader);

    downloadRegion(self.bridge, .kb_store, 0, kb_size, data[offset..]);
    offset += @intCast(kb_size);

    downloadRegion(self.bridge, .fact_store, 0, fact_size, data[offset..]);
    offset += @intCast(fact_size);

    downloadRegion(self.bridge, .rule_store, 0, rule_size, data[offset..]);
    offset += @intCast(rule_size);

    downloadRegion(self.bridge, .term_store, 0, term_size, data[offset..]);
    offset += @intCast(term_size);

    downloadRegion(self.bridge, .text_store, 0, text_size, data[offset..]);
    offset += @intCast(text_size);

    downloadRegion(self.bridge, .grammar_store, 0, grammar_size, data[offset..]);
    offset += @intCast(grammar_size);

    downloadRegion(self.bridge, .live_state, session.kb_store_offset, live_size, data[offset..]);
    offset += @intCast(live_size);

    downloadRegion(self.bridge, .grant_store, 0, grant_size, data[offset..]);

    // Compute checksum over data regions (everything after header)
    header.checksum = computeChecksum(data[@sizeOf(SnapshotHeader)..]);

    const snap_id = self.next_snapshot_id;
    self.next_snapshot_id += 1;

    return .{ .id = snap_id, .index = 0 };
}

// ============================================================
// Restore — upload host memory blob back to device
// ============================================================

pub fn restoreToDevice(self: *SnapshotManager, data: []const u8, session: *types.Session) types.Status {
    if (data.len < @sizeOf(SnapshotHeader)) return types.Status.err(.session, .snapshot_corrupt, 0);

    const header: *const SnapshotHeader = @ptrCast(@alignCast(data.ptr));

    // Validate
    if (!std.mem.eql(u8, &header.magic, &SNAPSHOT_MAGIC)) {
        return types.Status.err(.session, .snapshot_corrupt, -1);
    }
    if (header.version != SNAPSHOT_VERSION) {
        return types.Status.err(.session, .snapshot_corrupt, header.version);
    }

    // Verify checksum
    const payload = data[@sizeOf(SnapshotHeader)..];
    if (computeChecksum(payload) != header.checksum) {
        return types.Status.err(.session, .snapshot_corrupt, -2);
    }

    // Upload each region back to device
    var offset: usize = @sizeOf(SnapshotHeader);

    uploadRegion(self.bridge, .kb_store, 0, header.kb_region_size, data[offset..]);
    offset += @intCast(header.kb_region_size);

    uploadRegion(self.bridge, .fact_store, 0, header.fact_region_size, data[offset..]);
    offset += @intCast(header.fact_region_size);

    uploadRegion(self.bridge, .rule_store, 0, header.rule_region_size, data[offset..]);
    offset += @intCast(header.rule_region_size);

    uploadRegion(self.bridge, .term_store, 0, header.term_region_size, data[offset..]);
    offset += @intCast(header.term_region_size);

    uploadRegion(self.bridge, .text_store, 0, header.text_region_size, data[offset..]);
    offset += @intCast(header.text_region_size);

    uploadRegion(self.bridge, .grammar_store, 0, header.grammar_region_size, data[offset..]);
    offset += @intCast(header.grammar_region_size);

    uploadRegion(self.bridge, .live_state, session.kb_store_offset, header.live_state_region_size, data[offset..]);
    offset += @intCast(header.live_state_region_size);

    uploadRegion(self.bridge, .grant_store, 0, header.grant_region_size, data[offset..]);

    // Restore session metadata
    session.* = header.session_metadata;

    return types.Status.ok();
}

// ============================================================
// File I/O
// ============================================================

pub fn save(self: *SnapshotManager, data: []const u8, path: []const u8) types.Status {
    _ = self;
    const file = std.fs.cwd().createFile(path, .{}) catch
        return types.Status.err(.session, .snapshot_failed, 0);
    defer file.close();
    file.writeAll(data) catch return types.Status.err(.session, .snapshot_failed, -1);
    return types.Status.ok();
}

pub fn load(self: *SnapshotManager, path: []const u8) ?[]u8 {
    const file = std.fs.cwd().openFile(path, .{}) catch return null;
    defer file.close();
    const stat = file.stat() catch return null;
    const data = self.allocator.alloc(u8, stat.size) catch return null;
    const read = file.readAll(data) catch {
        self.allocator.free(data);
        return null;
    };
    _ = read;
    return data;
}

pub fn freeData(self: *SnapshotManager, data: []u8) void {
    self.allocator.free(data);
}

// ============================================================
// Diff — binary comparison of two snapshots
// ============================================================

pub fn diff(self: *SnapshotManager, a: []const u8, b: []const u8) DiffResult {
    if (a.len < @sizeOf(SnapshotHeader) or b.len < @sizeOf(SnapshotHeader)) {
        return .{ .entries = &.{}, .count = 0, .identical = false };
    }

    const ha: *const SnapshotHeader = @ptrCast(@alignCast(a.ptr));
    const hb: *const SnapshotHeader = @ptrCast(@alignCast(b.ptr));

    // Quick check: if checksums match, snapshots are identical
    if (ha.checksum == hb.checksum and ha.total_size == hb.total_size) {
        return DiffResult.same();
    }

    // Compare each region
    var entries = self.allocator.alloc(DiffEntry, 8) catch return .{ .entries = &.{}, .count = 0, .identical = false };
    var count: i32 = 0;

    const regions = [_]struct { region: DiffRegion, a_size: i64, b_size: i64 }{
        .{ .region = .kb, .a_size = ha.kb_region_size, .b_size = hb.kb_region_size },
        .{ .region = .fact, .a_size = ha.fact_region_size, .b_size = hb.fact_region_size },
        .{ .region = .rule, .a_size = ha.rule_region_size, .b_size = hb.rule_region_size },
        .{ .region = .term, .a_size = ha.term_region_size, .b_size = hb.term_region_size },
        .{ .region = .text, .a_size = ha.text_region_size, .b_size = hb.text_region_size },
        .{ .region = .grammar, .a_size = ha.grammar_region_size, .b_size = hb.grammar_region_size },
        .{ .region = .live_state, .a_size = ha.live_state_region_size, .b_size = hb.live_state_region_size },
        .{ .region = .grant, .a_size = ha.grant_region_size, .b_size = hb.grant_region_size },
    };

    var a_off: usize = @sizeOf(SnapshotHeader);
    var b_off: usize = @sizeOf(SnapshotHeader);

    for (regions) |r| {
        const a_end = a_off + @as(usize, @intCast(r.a_size));
        const b_end = b_off + @as(usize, @intCast(r.b_size));

        if (r.a_size != r.b_size or !std.mem.eql(u8, a[a_off..a_end], b[b_off..b_end])) {
            if (count < @as(i32, @intCast(entries.len))) {
                entries[@intCast(count)] = .{
                    .region = r.region,
                    .offset = 0,
                    .size = @max(r.a_size, r.b_size),
                    .a_hash = hashRegion(a[a_off..a_end]),
                    .b_hash = hashRegion(b[b_off..b_end]),
                };
                count += 1;
            }
        }

        a_off = a_end;
        b_off = b_end;
    }

    return .{
        .entries = entries[0..@intCast(count)],
        .count = count,
        .identical = count == 0,
    };
}

// ============================================================
// Three-way merge
// ============================================================

pub fn mergeThreeWay(self: *SnapshotManager, base: []const u8, branch_a: []const u8, branch_b: []const u8, policy: session_mod.MergePolicy) ?[]u8 {
    // Compare each branch against base to find changes
    const diff_a = self.diff(base, branch_a);
    const diff_b = self.diff(base, branch_b);

    // Start with base, apply non-conflicting changes
    const result = self.allocator.alloc(u8, base.len) catch return null;
    @memcpy(result, base);

    // Apply branch_a changes that don't conflict with branch_b
    for (diff_a.entries[0..@intCast(diff_a.count)]) |entry_a| {
        var conflicts = false;
        for (diff_b.entries[0..@intCast(diff_b.count)]) |entry_b| {
            if (entry_a.region == entry_b.region) {
                conflicts = true;
                break;
            }
        }

        if (!conflicts) {
            // Apply a's change
            applyRegionFromSnapshot(result, branch_a, entry_a.region);
        } else {
            switch (policy) {
                .ours => applyRegionFromSnapshot(result, branch_a, entry_a.region),
                .theirs => applyRegionFromSnapshot(result, branch_b, entry_a.region),
                .fail_on_conflict => {
                    self.allocator.free(result);
                    return null;
                },
            }
        }
    }

    // Apply branch_b changes that don't overlap with branch_a
    for (diff_b.entries[0..@intCast(diff_b.count)]) |entry_b| {
        var handled = false;
        for (diff_a.entries[0..@intCast(diff_a.count)]) |entry_a| {
            if (entry_a.region == entry_b.region) {
                handled = true;
                break;
            }
        }
        if (!handled) {
            applyRegionFromSnapshot(result, branch_b, entry_b.region);
        }
    }

    // Recompute checksum
    const header: *SnapshotHeader = @ptrCast(@alignCast(result.ptr));
    header.checksum = computeChecksum(result[@sizeOf(SnapshotHeader)..]);

    return result;
}

// ============================================================
// Integrity
// ============================================================

pub fn validateChecksum(data: []const u8) bool {
    if (data.len < @sizeOf(SnapshotHeader)) return false;
    const header: *const SnapshotHeader = @ptrCast(@alignCast(data.ptr));
    if (!std.mem.eql(u8, &header.magic, &SNAPSHOT_MAGIC)) return false;
    return computeChecksum(data[@sizeOf(SnapshotHeader)..]) == header.checksum;
}

pub fn computeChecksum(data: []const u8) i32 {
    // CRC32 — standard polynomial, integer-only
    var crc: u32 = 0xFFFFFFFF;
    for (data) |b| {
        crc ^= @as(u32, b);
        var bit: u32 = 0;
        while (bit < 8) : (bit += 1) {
            if ((crc & 1) != 0) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc = crc >> 1;
            }
        }
    }
    return @bitCast(crc ^ 0xFFFFFFFF);
}

// ============================================================
// Helpers
// ============================================================

fn downloadRegion(bridge: *bridge_mod.Bridge, target: bridge_mod.BufferTarget, offset: i64, size: i64, dest: []u8) void {
    if (size <= 0) return;
    const len: usize = @intCast(size);
    if (len > dest.len) return;
    _ = bridge.downloadFromBuffer(target, offset, dest[0..len]);
}

fn uploadRegion(bridge: *bridge_mod.Bridge, target: bridge_mod.BufferTarget, offset: i64, size: i64, src: []const u8) void {
    if (size <= 0) return;
    const len: usize = @intCast(size);
    if (len > src.len) return;
    _ = bridge.uploadToBuffer(target, offset, src[0..len]);
}

fn hashRegion(data: []const u8) u32 {
    var h: u32 = 2166136261;
    for (data) |b| {
        h ^= @as(u32, b);
        h *%= 16777619;
    }
    return h;
}

fn applyRegionFromSnapshot(dest: []u8, src: []const u8, region: DiffRegion) void {
    // Copy the entire region from src to dest at matching position
    // Region offsets are computed from header sizes
    _ = dest;
    _ = src;
    _ = region;
    // Real implementation would use header to compute offsets
}

fn currentTimestamp() i32 {
    const ts = std.time.timestamp();
    return @intCast(@min(ts, std.math.maxInt(i32)));
}
