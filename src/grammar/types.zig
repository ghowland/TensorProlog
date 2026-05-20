// ============================================================
// src/grammar/types.zig
// ============================================================

const vdr_types = @import("../vdr/types.zig");
const q16_mod = @import("../vdr/q16.zig");

pub const VlpStatus = vdr_types.VlpStatus;
pub const VlpSlotType = vdr_types.VlpSlotType;
pub const Q16 = q16_mod.Q16;

pub const GrammarSlot = struct {
    name_offset: i32 = 0,
    name_length: i16 = 0,
    slot_type: VlpSlotType = .text,
    template_pos: i32 = 0,
    template_end: i32 = 0,
    enum_offset: i32 = 0,
    enum_count: i16 = 0,
    default_kb_id: i32 = -1,
    default_slot_id: i32 = -1,
};

pub const VlpGrammar = struct {
    id: i32 = -1,
    template_offset: i32 = 0,
    template_length: i32 = 0,
    slots: [32]GrammarSlot = undefined,
    slot_count: i16 = 0,
    literal_ranges: [64]LiteralRange = undefined,
    literal_count: i16 = 0,
    validated: bool = false,
    created_at: i32 = 0,
    creator_session_id: i32 = -1,
};

pub const LiteralRange = struct {
    start: i32 = 0,
    end: i32 = 0,
};

pub const GrammarFill = struct {
    slot_index: i16 = 0,
    fill_type: VlpSlotType = .text,
    vdr_value: Q16 = Q16.zero(),
    text_ptr: ?[]const u8 = null,
    int_value: i32 = 0,
    enum_index: i16 = 0,
};

pub const GrammarKBMapping = struct {
    slot_index: i16 = 0,
    kb_id: i32 = -1,
    slot_id: i32 = -1,
};

pub const RenderResult = struct {
    len: i32 = 0,
    status: VlpStatus = .ok,
};
