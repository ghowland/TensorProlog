// ============================================================
// src/grammar/compile.zig
// ============================================================

const std = @import("std");
const grammar_types = @import("types.zig");
const text_mod = @import("../kb/text_store.zig");

const VlpGrammar = grammar_types.VlpGrammar;
const GrammarSlot = grammar_types.GrammarSlot;
const LiteralRange = grammar_types.LiteralRange;
const VlpSlotType = grammar_types.VlpSlotType;
const VlpStatus = grammar_types.VlpStatus;
const TextStore = text_mod.TextStore;

pub fn compile(template: []const u8, text: *TextStore, grammar: *VlpGrammar) VlpStatus {
    grammar.* = VlpGrammar{};

    const tref = text.append(template) orelse return .err_out_of_memory;
    grammar.template_offset = tref.offset;
    grammar.template_length = @intCast(template.len);

    var pos: i32 = 0;
    var lit_start: i32 = 0;
    const tlen: i32 = @intCast(template.len);

    while (pos < tlen) {
        if (template[@intCast(pos)] == '{') {
            if (pos > lit_start) {
                if (grammar.literal_count >= 64) return .err_grammar_invalid;
                grammar.literal_ranges[@intCast(grammar.literal_count)] = .{ .start = lit_start, .end = pos };
                grammar.literal_count += 1;
            }

            const close = findClose(template, pos);
            if (close < 0) return .err_grammar_invalid;

            const inner = template[@intCast(pos + 1)..@intCast(close)];
            const slot_status = parseSlot(inner, text, grammar);
            if (slot_status != .ok) return slot_status;

            pos = close + 1;
            lit_start = pos;
        } else {
            pos += 1;
        }
    }

    if (pos > lit_start) {
        if (grammar.literal_count >= 64) return .err_grammar_invalid;
        grammar.literal_ranges[@intCast(grammar.literal_count)] = .{ .start = lit_start, .end = pos };
        grammar.literal_count += 1;
    }

    grammar.validated = true;
    return .ok;
}

fn findClose(template: []const u8, open_pos: i32) i32 {
    var depth: i32 = 0;
    var p: i32 = open_pos;
    const tlen: i32 = @intCast(template.len);
    while (p < tlen) : (p += 1) {
        if (template[@intCast(p)] == '{') depth += 1;
        if (template[@intCast(p)] == '}') {
            depth -= 1;
            if (depth == 0) return p;
        }
    }
    return -1;
}

fn parseSlot(inner: []const u8, text: *TextStore, grammar: *VlpGrammar) VlpStatus {
    if (grammar.slot_count >= 32) return .err_grammar_invalid;

    const colon_pos = findByte(inner, ':');
    if (colon_pos < 0) return .err_grammar_invalid;

    const name = inner[0..@intCast(colon_pos)];
    const type_str = inner[@intCast(colon_pos + 1)..];

    if (name.len == 0) return .err_grammar_invalid;
    if (type_str.len == 0) return .err_grammar_invalid;

    const nref = text.append(name) orelse return .err_out_of_memory;

    var slot = GrammarSlot{};
    slot.name_offset = nref.offset;
    slot.name_length = nref.length;

    if (std.mem.eql(u8, type_str, "text")) {
        slot.slot_type = .text;
    } else if (std.mem.eql(u8, type_str, "integer")) {
        slot.slot_type = .integer;
    } else if (std.mem.eql(u8, type_str, "vdr_value")) {
        slot.slot_type = .vdr_value;
    } else if (std.mem.startsWith(u8, type_str, "enum(")) {
        slot.slot_type = .enum_val;
        const enum_status = parseEnum(type_str, text, &slot);
        if (enum_status != .ok) return enum_status;
    } else if (std.mem.eql(u8, type_str, "kb_ref")) {
        slot.slot_type = .kb_ref;
    } else if (std.mem.eql(u8, type_str, "grammar")) {
        slot.slot_type = .grammar;
    } else {
        return .err_grammar_invalid;
    }

    grammar.slots[@intCast(grammar.slot_count)] = slot;
    grammar.slot_count += 1;
    return .ok;
}

fn parseEnum(type_str: []const u8, text: *TextStore, slot: *GrammarSlot) VlpStatus {
    if (type_str.len < 6) return .err_grammar_invalid;
    if (type_str[type_str.len - 1] != ')') return .err_grammar_invalid;

    const values_str = type_str[5 .. type_str.len - 1];
    if (values_str.len == 0) return .err_grammar_invalid;

    const first_ref = text.append(values_str) orelse return .err_out_of_memory;
    slot.enum_offset = first_ref.offset;

    var count: i16 = 1;
    for (values_str) |c| {
        if (c == '|') count += 1;
    }
    slot.enum_count = count;
    return .ok;
}

fn findByte(data: []const u8, needle: u8) i32 {
    for (data, 0..) |b, i| {
        if (b == needle) return @intCast(i);
    }
    return -1;
}

pub fn slotName(grammar: *const VlpGrammar, slot_idx: i16, text: *const TextStore) ?[]const u8 {
    if (slot_idx < 0 or slot_idx >= grammar.slot_count) return null;
    const slot = &grammar.slots[@intCast(slot_idx)];
    return text.read(.{ .offset = slot.name_offset, .length = slot.name_length });
}

pub fn slotType(grammar: *const VlpGrammar, slot_idx: i16) ?VlpSlotType {
    if (slot_idx < 0 or slot_idx >= grammar.slot_count) return null;
    return grammar.slots[@intCast(slot_idx)].slot_type;
}

pub fn enumValues(grammar: *const VlpGrammar, slot_idx: i16, text: *const TextStore) ?[]const u8 {
    if (slot_idx < 0 or slot_idx >= grammar.slot_count) return null;
    const slot = &grammar.slots[@intCast(slot_idx)];
    if (slot.slot_type != .enum_val) return null;
    if (slot.enum_count <= 0) return null;
    return text.read(.{ .offset = slot.enum_offset, .length = @intCast(slot.enum_count * 16) });
}
