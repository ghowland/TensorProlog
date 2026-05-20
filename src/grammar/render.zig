// ============================================================
// src/grammar/render.zig
// ============================================================

const std = @import("std");
const grammar_types = @import("types.zig");
const text_mod = @import("../kb/text_store.zig");
const store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const q16_mod = @import("../vdr/q16.zig");

const VlpGrammar = grammar_types.VlpGrammar;
const GrammarFill = grammar_types.GrammarFill;
const GrammarKBMapping = grammar_types.GrammarKBMapping;
const RenderResult = grammar_types.RenderResult;
const VlpSlotType = grammar_types.VlpSlotType;
const VlpStatus = grammar_types.VlpStatus;
const Q16 = q16_mod.Q16;
const TextStore = text_mod.TextStore;
const KBStore = store_mod.KBStore;

pub fn render(
    grammar: *const VlpGrammar,
    text: *const TextStore,
    fills: []const GrammarFill,
    output: []u8,
) RenderResult {
    if (!grammar.validated) return .{ .len = 0, .status = .err_grammar_invalid };

    const template = text.read(.{
        .offset = grammar.template_offset,
        .length = @intCast(grammar.template_length),
    }) orelse return .{ .len = 0, .status = .err_grammar_invalid };

    var out_pos: i32 = 0;
    var slot_idx: i16 = 0;
    const lit_idx: i16 = 0;
    var tmpl_pos: i32 = 0;
    const tmpl_len: i32 = @intCast(template.len);
    const out_cap: i32 = @intCast(output.len);

    while (tmpl_pos < tmpl_len) {
        if (template[@intCast(tmpl_pos)] == '{') {
            const close = findClose(template, tmpl_pos);
            if (close < 0) return .{ .len = out_pos, .status = .err_grammar_invalid };

            if (slot_idx < grammar.slot_count) {
                const fill = findFill(fills, slot_idx);
                if (fill) |f| {
                    const wrote = renderFill(f, text, output, out_pos, out_cap);
                    if (wrote < 0) return .{ .len = out_pos, .status = .err_grammar_capacity };
                    out_pos += wrote;
                }
                slot_idx += 1;
            }
            tmpl_pos = close + 1;
        } else {
            if (out_pos >= out_cap) return .{ .len = out_pos, .status = .err_grammar_capacity };
            output[@intCast(out_pos)] = template[@intCast(tmpl_pos)];
            out_pos += 1;
            tmpl_pos += 1;
        }
    }
    _ = lit_idx;

    return .{ .len = out_pos, .status = .ok };
}

pub fn renderFromKB(
    grammar: *const VlpGrammar,
    text: *const TextStore,
    kb_store: *const KBStore,
    mappings: []const GrammarKBMapping,
    output: []u8,
) RenderResult {
    var fills: [32]GrammarFill = undefined;
    var fill_count: i16 = 0;

    for (mappings) |m| {
        if (fill_count >= 32) break;
        const fact = fact_mod.query(kb_store, m.kb_id, m.slot_id) orelse continue;
        fills[@intCast(fill_count)] = .{
            .slot_index = m.slot_index,
            .fill_type = .vdr_value,
            .vdr_value = fact.value,
        };
        fill_count += 1;
    }

    return render(grammar, text, fills[0..@intCast(fill_count)], output);
}

fn findFill(fills: []const GrammarFill, slot_idx: i16) ?*const GrammarFill {
    for (fills) |*f| {
        if (f.slot_index == slot_idx) return f;
    }
    return null;
}

fn renderFill(fill: *const GrammarFill, text: *const TextStore, output: []u8, pos: i32, cap: i32) i32 {
    _ = text;
    return switch (fill.fill_type) {
        .text => renderText(fill, output, pos, cap),
        .integer => renderInt(fill.int_value, output, pos, cap),
        .vdr_value => renderVdr(fill.vdr_value, output, pos, cap),
        .enum_val => renderInt(@as(i32, fill.enum_index), output, pos, cap),
        .kb_ref => renderInt(fill.int_value, output, pos, cap),
        .grammar => 0,
    };
}

fn renderText(fill: *const GrammarFill, output: []u8, pos: i32, cap: i32) i32 {
    const txt = fill.text_ptr orelse return 0;
    const len: i32 = @intCast(txt.len);
    if (pos + len > cap) return -1;
    @memcpy(output[@intCast(pos)..@intCast(pos + len)], txt);
    return len;
}

fn renderInt(val: i32, output: []u8, pos: i32, cap: i32) i32 {
    var buf: [12]u8 = undefined;
    var v: i32 = val;
    var negative = false;
    if (v < 0) {
        negative = true;
        v = -v;
    }
    if (v == 0) {
        if (pos >= cap) return -1;
        output[@intCast(pos)] = '0';
        return 1;
    }
    var len: i32 = 0;
    while (v > 0) : (len += 1) {
        buf[@intCast(len)] = @intCast(@as(u32, @intCast(@mod(v, 10))) + '0');
        v = @divTrunc(v, 10);
    }
    if (negative) {
        buf[@intCast(len)] = '-';
        len += 1;
    }
    if (pos + len > cap) return -1;
    var i: i32 = 0;
    while (i < len) : (i += 1) {
        output[@intCast(pos + i)] = buf[@intCast(len - 1 - i)];
    }
    return len;
}

fn renderVdr(val: Q16, output: []u8, pos: i32, cap: i32) i32 {
    var wrote: i32 = 0;

    const int_part = @divTrunc(val.v, Q16.D);
    const frac_part = @mod(val.v, Q16.D);

    const ip = renderInt(int_part, output, pos, cap);
    if (ip < 0) return -1;
    wrote += ip;

    if (frac_part == 0 and val.r0 == 0) return wrote;

    if (pos + wrote >= cap) return -1;
    output[@intCast(pos + wrote)] = '.';
    wrote += 1;

    var rem: i64 = @as(i64, frac_part);
    var digits: i32 = 0;
    while (digits < 6) : (digits += 1) {
        rem *= 10;
        const digit: i32 = @intCast(@divTrunc(rem, Q16.D_i64));
        rem = @mod(rem, Q16.D_i64);
        if (pos + wrote >= cap) return -1;
        output[@intCast(pos + wrote)] = @intCast(@as(u32, @intCast(digit)) + '0');
        wrote += 1;
    }

    return wrote;
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
