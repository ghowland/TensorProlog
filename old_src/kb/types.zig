// ============================================================
// src/kb/types.zig
// ============================================================

const q16_mod = @import("../vdr/q16.zig");
const vdr_types = @import("../vdr/types.zig");

pub const Q16 = q16_mod.Q16;
pub const VlpStatus = vdr_types.VlpStatus;
pub const VlpFactTag = vdr_types.VlpFactTag;
pub const VlpSourceType = vdr_types.VlpSourceType;
pub const VlpVisibility = vdr_types.VlpVisibility;

pub const VlpProvenance = struct {
    source_type: VlpSourceType = .unknown,
    source_kb_id: i32 = -1,
    source_slot_id: i32 = -1,
    confidence: Q16 = Q16.zero(),
    timestamp: i32 = 0,
    derivation_rule_id: i32 = -1,
};

pub const VlpFact = struct {
    tag: VlpFactTag = .empty,
    value: Q16 = Q16.zero(),
    provenance: VlpProvenance = .{},
};

pub const VlpKB = struct {
    name_offset: i32 = 0,
    name_length: i16 = 0,
    path_offset: i32 = 0,
    path_length: i16 = 0,
    id: i32 = -1,
    facts_offset: i32 = 0,
    facts_count: i32 = 0,
    facts_capacity: i32 = 0,
    rules_offset: i32 = 0,
    rules_count: i32 = 0,
    rules_capacity: i32 = 0,
    constraints_offset: i32 = 0,
    constraints_count: i32 = 0,
    connections_offset: i32 = 0,
    connections_count: i32 = 0,
    grammars_offset: i32 = 0,
    grammars_count: i32 = 0,
    iose_offset: i32 = -1,
    working_data_offset: i32 = 0,
    lru_table_offset: i32 = 0,
    lru_count: i16 = 0,
    counter_table_offset: i32 = 0,
    counter_count: i16 = 0,
    lock_table_offset: i32 = 0,
    lock_count: i16 = 0,
    queue_table_offset: i32 = 0,
    queue_count: i16 = 0,
    stack_table_offset: i32 = 0,
    stack_count: i16 = 0,
    ring_table_offset: i32 = 0,
    ring_count: i16 = 0,
    bitset_table_offset: i32 = 0,
    bitset_count: i16 = 0,
    parent_id: i32 = -1,
    children_offset: i32 = 0,
    children_count: i16 = 0,
    children_capacity: i16 = 0,
    mounts_offset: i32 = 0,
    mounts_count: i16 = 0,
    visibility: VlpVisibility = .public,
    frozen: bool = false,
    owner_id: i32 = -1,
    created_at: i32 = 0,
    last_modified: i32 = 0,
    alive: bool = false,
};

pub const KBCreateConfig = struct {
    name: []const u8,
    parent_id: i32 = -1,
    visibility: VlpVisibility = .public,
    owner_id: i32 = -1,
    max_facts: i32 = 256,
    max_rules: i32 = 64,
    max_children: i32 = 64,
};
