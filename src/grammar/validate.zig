// ============================================================
// src/grammar/validate.zig
// ============================================================

const grammar_types = @import("types.zig");
const text_mod = @import("../kb/text_store.zig");

const VlpGrammar = grammar_types.VlpGrammar;
const TextStore = text_mod.TextStore;

pub const ValidationResult = struct {
    valid: bool = false,
    error_pos: i32 = -1,
};

pub fn validate(grammar: *const VlpGrammar, text: *const TextStore) ValidationResult {
    const template = text.read(.{
        .offset = grammar.template_offset,
        .length = @intCast(grammar.template_length),
    }) orelse return .{ .valid = false, .error_pos = 0 };

    var depth: i32 = 0;
    var slot_count: i16 = 0;
    var pos: i32 = 0;
    const tlen: i32 = @intCast(template.len);

    while (pos < tlen) : (pos += 1) {
        if (template[@intCast(pos)] == '{') {
            depth += 1;
            if (depth == 1) slot_count += 1;
        }
        if (template[@intCast(pos)] == '}') {
            depth -= 1;
            if (depth < 0) return .{ .valid = false, .error_pos = pos };
        }
    }

    if (depth != 0) return .{ .valid = false, .error_pos = pos };
    if (slot_count != grammar.slot_count) return .{ .valid = false, .error_pos = -1 };

    var si: i16 = 0;
    while (si < grammar.slot_count) : (si += 1) {
        const slot = &grammar.slots[@intCast(si)];
        const name = text.read(.{ .offset = slot.name_offset, .length = slot.name_length }) orelse
            return .{ .valid = false, .error_pos = @as(i32, si) };
        if (name.len == 0) return .{ .valid = false, .error_pos = @as(i32, si) };

        switch (slot.slot_type) {
            .enum_val => {
                if (slot.enum_count <= 0) return .{ .valid = false, .error_pos = @as(i32, si) };
            },
            else => {},
        }
    }

    return .{ .valid = true, .error_pos = -1 };
}

pub fn revalidate(grammar: *VlpGrammar, text: *const TextStore) void {
    const r = validate(grammar, text);
    grammar.validated = r.valid;
}
