// ============================================================
// vlp_session.zig
// Session manager — host-side lifecycle management.
// Create, destroy, snapshot, clone, merge, kill.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const kb_mod = @import("vlp_kb_store.zig");

// ============================================================
// Configuration
// ============================================================

pub const SessionConfig = struct {
    user_id: i32,
    kb_root_id: i32,
    visibility_level: i8 = 1, // default INTERNAL
    max_kb_count: i32 = 100,
    max_live_memory_bytes: i64 = 50 * 1024 * 1024,
    max_turns: i32 = 0, // 0 = unlimited
    auto_snapshot_interval: i32 = 100, // 0 = disabled
};

pub const CloneConfig = struct {
    fresh_live: bool = true,
    inherit_rules: bool = true,
};

pub const MergePolicy = enum(i32) {
    ours = 0, // parent wins on conflict
    theirs = 1, // child wins on conflict
    fail_on_conflict = 2,
};

pub const MergeConflict = struct {
    kb_id: i32,
    slot_id: i32,
    parent_timestamp: i32,
    child_timestamp: i32,
};

pub const MergeResult = struct {
    status: types.Status,
    merged_count: i32,
    conflict_count: i32,
    conflicts: []MergeConflict,

    pub fn ok(merged: i32) MergeResult {
        return .{
            .status = types.Status.ok(),
            .merged_count = merged,
            .conflict_count = 0,
            .conflicts = &.{},
        };
    }

    pub fn failed(status: types.Status) MergeResult {
        return .{
            .status = status,
            .merged_count = 0,
            .conflict_count = 0,
            .conflicts = &.{},
        };
    }
};

// ============================================================
// Session Manager
// ============================================================

pub const SessionManager = struct {
    bridge: *bridge_mod.Bridge,
    kb_store: *kb_mod.KbStore,
    allocator: std.mem.Allocator,

    sessions: []types.Session,
    session_active: []bool,
    session_capacity: i32,
    session_count: i32,
    next_session_id: i32,

    // Level stats per session
    level_stats: []types.LevelStats,

    // Merge conflict scratch buffer
    conflict_buf: []MergeConflict,
    conflict_capacity: i32,

    // Auto-snapshot configuration per session
    auto_snapshot_intervals: []i32,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(bridge: *bridge_mod.Bridge, kb_store: *kb_mod.KbStore, allocator: std.mem.Allocator, max_sessions: i32) SessionManager {
    const cap: usize = @intCast(max_sessions);

    const sessions = allocator.alloc(types.Session, cap) catch &.{};
    const active = allocator.alloc(bool, cap) catch &.{};
    const stats = allocator.alloc(types.LevelStats, cap) catch &.{};
    const conflicts = allocator.alloc(MergeConflict, 256) catch &.{};
    const intervals = allocator.alloc(i32, cap) catch &.{};

    if (active.len > 0) @memset(active, false);
    if (stats.len > 0) @memset(stats, std.mem.zeroes(types.LevelStats));
    if (intervals.len > 0) @memset(intervals, 0);

    return .{
        .bridge = bridge,
        .kb_store = kb_store,
        .allocator = allocator,
        .sessions = sessions,
        .session_active = active,
        .session_capacity = max_sessions,
        .session_count = 0,
        .next_session_id = 0,
        .level_stats = stats,
        .conflict_buf = conflicts,
        .conflict_capacity = @intCast(conflicts.len),
        .auto_snapshot_intervals = intervals,
    };
}

pub fn deinit(self: *SessionManager) void {
    if (self.sessions.len > 0) self.allocator.free(self.sessions);
    if (self.session_active.len > 0) self.allocator.free(self.session_active);
    if (self.level_stats.len > 0) self.allocator.free(self.level_stats);
    if (self.conflict_buf.len > 0) self.allocator.free(self.conflict_buf);
    if (self.auto_snapshot_intervals.len > 0) self.allocator.free(self.auto_snapshot_intervals);
}

// ============================================================
// Create / Destroy
// ============================================================

pub fn create(self: *SessionManager, config: *const SessionConfig) ?types.SessionHandle {
    // Find free slot
    const index = self.findFreeSlot() orelse return null;
    const idx: usize = @intCast(index);

    const session_id = self.next_session_id;
    self.next_session_id += 1;

    var session = std.mem.zeroes(types.Session);
    session.id = session_id;
    session.user_id = config.user_id;
    session.kb_root_id = config.kb_root_id;
    session.visibility_level = config.visibility_level;
    session.state = .active;
    session.max_kb_count = config.max_kb_count;
    session.max_live_memory_bytes = config.max_live_memory_bytes;
    session.max_turns = config.max_turns;
    session.device_id = 0;
    session.stream_id = session_id; // 1:1 for now
    session.last_snapshot_id = -1;
    session.parent_session_id = -1;
    session.clone_generation = 0;

    self.sessions[idx] = session;
    self.session_active[idx] = true;
    self.level_stats[idx] = std.mem.zeroes(types.LevelStats);
    self.auto_snapshot_intervals[idx] = config.auto_snapshot_interval;
    self.session_count += 1;

    return .{ .id = session_id, .index = index };
}

pub fn destroy(self: *SessionManager, handle: types.SessionHandle) types.Status {
    const idx: usize = @intCast(handle.index);
    if (idx >= self.session_active.len or !self.session_active[idx]) {
        return types.Status.err(.session, .session_limit, handle.id);
    }

    // Clean up COW tables for this session
    self.kb_store.cowDestroy(handle.id);

    self.sessions[idx].state = .killed;
    self.session_active[idx] = false;
    self.session_count -= 1;

    return types.Status.ok();
}

pub fn get(self: *SessionManager, handle: types.SessionHandle) ?*types.Session {
    const idx: usize = @intCast(handle.index);
    if (idx >= self.session_active.len or !self.session_active[idx]) return null;
    if (self.sessions[idx].id != handle.id) return null;
    return &self.sessions[idx];
}

pub fn kill(self: *SessionManager, handle: types.SessionHandle) types.Status {
    // Immediate termination. No snapshot. COW pages discarded.
    return self.destroy(handle);
}

// ============================================================
// Clone
// ============================================================

pub fn clone(self: *SessionManager, parent_handle: types.SessionHandle, config: *const CloneConfig) ?types.SessionHandle {
    const parent = self.get(parent_handle) orelse return null;

    // Create child session
    const child_config = SessionConfig{
        .user_id = parent.user_id,
        .kb_root_id = parent.kb_root_id,
        .visibility_level = parent.visibility_level,
        .max_kb_count = parent.max_kb_count,
        .max_live_memory_bytes = parent.max_live_memory_bytes,
        .max_turns = parent.max_turns,
    };

    const child_handle = self.create(&child_config) orelse return null;
    const child = self.get(child_handle) orelse return null;

    child.parent_session_id = parent.id;
    child.clone_generation = parent.clone_generation + 1;

    // Set up COW
    // Clone shares parent's fact region. Writes trigger page copy.
    const region_size = parent.max_live_memory_bytes;
    const parent_offset = parent.kb_store_offset;
    // Allocate private region for clone's dirty pages
    const private_offset = parent_offset + region_size; // simplified
    _ = self.kb_store.cowInit(parent.id, child.id, region_size, parent_offset, private_offset);

    // Copy live state if requested
    if (!config.fresh_live) {
        // Copy parent's live state region to clone
        _ = self.bridge.copyBufferToBuffer(
            .live_state,
            parent_offset,
            .live_state,
            private_offset,
            region_size,
        );
    }

    // Rules are inherited by default (shared via COW)
    _ = config.inherit_rules;

    return child_handle;
}

// ============================================================
// Merge — combine child's changes back into parent
// ============================================================

pub fn merge(self: *SessionManager, parent_handle: types.SessionHandle, child_handle: types.SessionHandle, policy: MergePolicy) MergeResult {
    const parent = self.get(parent_handle) orelse return MergeResult.failed(types.Status.err(.session, .session_limit, parent_handle.id));
    const child = self.get(child_handle) orelse return MergeResult.failed(types.Status.err(.session, .session_limit, child_handle.id));

    // Verify child is a clone of parent
    if (child.parent_session_id != parent.id) {
        return MergeResult.failed(types.Status.err(.session, .merge_conflict, child.id));
    }

    // Find COW table for this clone
    const cow = self.kb_store.findCowMut(child.id) orelse
        return MergeResult.failed(types.Status.err(.session, .clone_failed, child.id));

    var merged: i32 = 0;
    var conflicts: i32 = 0;
    const clone_time = child.last_snapshot_timestamp; // approximate clone creation time

    // For each dirty page, check for conflicts and merge
    var page: i32 = 0;
    while (page < cow.n_pages) : (page += 1) {
        if (!cow.isDirty(page)) continue;

        // Read parent's version of this page
        // Read child's version of this page
        // Compare timestamps to detect conflicts
        // For now, simplified: copy child's dirty pages to parent
        switch (policy) {
            .ours => {
                // Parent wins — skip dirty pages (parent unchanged)
                merged += 1;
            },
            .theirs => {
                // Child wins — copy child's page to parent
                const child_off = cow.pageOffsetInPrivate(page);
                const parent_off = cow.pageOffsetInParent(page);
                _ = self.bridge.copyBufferToBuffer(.fact_store, child_off, .fact_store, parent_off, kb_mod.COW_PAGE_SIZE);
                merged += 1;
            },
            .fail_on_conflict => {
                // Check if parent also modified this page since clone
                // Simplified: assume conflict if page is dirty on both sides
                // Real implementation compares per-fact timestamps
                if (conflicts < self.conflict_capacity) {
                    self.conflict_buf[@intCast(conflicts)] = .{
                        .kb_id = -1, // would need page-to-kb mapping
                        .slot_id = page,
                        .parent_timestamp = clone_time,
                        .child_timestamp = kb_mod.currentTimestamp(),
                    };
                }
                conflicts += 1;
            },
        }
    }

    if (policy == .fail_on_conflict and conflicts > 0) {
        return .{
            .status = types.Status.err(.session, .merge_conflict, conflicts),
            .merged_count = 0,
            .conflict_count = conflicts,
            .conflicts = self.conflict_buf[0..@intCast(@min(conflicts, self.conflict_capacity))],
        };
    }

    // Update parent counters
    parent.facts_asserted += child.facts_asserted;
    parent.facts_retracted += child.facts_retracted;
    parent.last_modified = kb_mod.currentTimestamp();

    return MergeResult.ok(merged);
}

// ============================================================
// Level stats
// ============================================================

pub fn updateLevelStats(self: *SessionManager, handle: types.SessionHandle, level: i8, tokens: i32) types.Status {
    const idx: usize = @intCast(handle.index);
    if (idx >= self.level_stats.len) return types.Status.err(.session, .session_limit, handle.id);

    var stats = &self.level_stats[idx];
    switch (level) {
        1 => {
            stats.l1_count += 1;
            stats.l1_tokens += tokens;
        },
        2 => {
            stats.l2_count += 1;
            stats.l2_tokens += tokens;
        },
        3 => {
            stats.l3_count += 1;
            // l3 tokens always 0
        },
        else => {},
    }

    return types.Status.ok();
}

pub fn getLevelStats(self: *SessionManager, handle: types.SessionHandle) types.LevelStats {
    const idx: usize = @intCast(handle.index);
    if (idx >= self.level_stats.len) return std.mem.zeroes(types.LevelStats);
    return self.level_stats[idx];
}

// ============================================================
// Auto-snapshot check — called after each turn
// ============================================================

pub fn shouldAutoSnapshot(self: *SessionManager, handle: types.SessionHandle) bool {
    const idx: usize = @intCast(handle.index);
    if (idx >= self.auto_snapshot_intervals.len) return false;
    const interval = self.auto_snapshot_intervals[idx];
    if (interval <= 0) return false;
    const session = self.get(handle) orelse return false;
    return @mod(session.current_turn, interval) == 0;
}

// ============================================================
// Session increment — called after each inference turn
// ============================================================

pub fn incrementTurn(self: *SessionManager, handle: types.SessionHandle, llm_tokens: i32, command_tokens: i32) types.Status {
    const session = self.get(handle) orelse return types.Status.err(.session, .session_limit, handle.id);
    session.current_turn += 1;
    session.llm_tokens_consumed += llm_tokens;
    session.command_tokens_consumed += command_tokens;
    return types.Status.ok();
}

// ============================================================
// Helpers
// ============================================================

fn findFreeSlot(self: *SessionManager) ?i32 {
    for (self.session_active, 0..) |active, i| {
        if (!active) return @intCast(i);
    }
    return null;
}

pub fn findByUserId(self: *SessionManager, user_id: i32) ?types.SessionHandle {
    for (self.sessions, 0..) |s, i| {
        if (self.session_active[i] and s.user_id == user_id and s.isActive()) {
            return .{ .id = s.id, .index = @intCast(i) };
        }
    }
    return null;
}

pub fn activeCount(self: *SessionManager) i32 {
    return self.session_count;
}
