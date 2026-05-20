// ============================================================
// src/session/types.zig
// ============================================================

const vdr_types = @import("../vdr/types.zig");
const q16_mod = @import("../vdr/q16.zig");

pub const VlpStatus = vdr_types.VlpStatus;
pub const VlpSessionState = vdr_types.VlpSessionState;
pub const VlpVisibility = vdr_types.VlpVisibility;
pub const VlpMergePolicy = vdr_types.VlpMergePolicy;
pub const VlpExecutionLevel = vdr_types.VlpExecutionLevel;
pub const Q16 = q16_mod.Q16;

pub const VlpSession = struct {
    id: i32 = -1,
    user_id: i32 = -1,
    kb_root_id: i32 = -1,
    visibility_level: VlpVisibility = .public,
    state: VlpSessionState = .active,
    max_kb_count: i32 = 100,
    max_live_bytes: i64 = 16 * 1024 * 1024,
    max_turns: i32 = 0,
    current_turn: i32 = 0,
    facts_asserted: i32 = 0,
    facts_retracted: i32 = 0,
    rules_fired: i64 = 0,
    prolog_queries: i64 = 0,
    primitive_calls: i64 = 0,
    grammar_renders: i64 = 0,
    llm_tokens: i64 = 0,
    command_tokens: i64 = 0,
    l1_count: i64 = 0,
    l2_count: i64 = 0,
    l3_count: i64 = 0,
    last_snapshot_id: i32 = -1,
    last_snapshot_ts: i32 = 0,
    parent_session_id: i32 = -1,
    clone_generation: i32 = 0,
    alive: bool = false,
};

pub const SessionConfig = struct {
    kb_root_id: i32 = -1,
    user_id: i32 = -1,
    visibility_level: VlpVisibility = .public,
    max_kb_count: i32 = 100,
    max_live_bytes: i64 = 16 * 1024 * 1024,
    max_turns: i32 = 0,
};

pub const CloneConfig = struct {
    fresh_live: bool = true,
    inherit_rules: bool = true,
    max_turns: i32 = 0,
};

pub const MergeConflict = struct {
    kb_id: i32,
    slot_id: i32,
};

pub const MergeResult = struct {
    status: VlpStatus = .ok,
    merged_count: i32 = 0,
    conflict_count: i32 = 0,
    conflicts: [64]MergeConflict = undefined,
};
