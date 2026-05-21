// ============================================================
// vlp_grammar.zig
// Grammar engine — entirely host-side.
// Compile templates, render with fills, inherit through KB tree.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const kb_mod = @import("vlp_kb_store.zig");

// ============================================================
// Compile result — parsed template ready for rendering
// ============================================================

pub const LiteralRange = struct {
    start: i32, // byte offset into template
    length: i32,
};

pub const SlotPosition = struct {
    slot_index: i16,
    template_offset: i32, // where the slot marker begins in template
    marker_length: i32, // length of the {slot_name} marker
};

pub const CompileResult = struct {
    grammar: types.Grammar,
    slots: []types.GrammarSlot,
    literals: []LiteralRange,
    slot_positions: []SlotPosition,
    n_literals: i32,
    n_slots: i32,
    status: types.Status,

    pub fn failed(status: types.Status) CompileResult {
        return .{
            .grammar = std.mem.zeroes(types.Grammar),
            .slots = &.{},
            .literals = &.{},
            .slot_positions = &.{},
            .n_literals = 0,
            .n_slots = 0,
            .status = status,
        };
    }
};

pub const RenderConfig = struct {
    max_output_bytes: i32 = 16384,
    recursive_depth_limit: i32 = 10,
};

// ============================================================
// Grammar Engine
// ============================================================

pub const GrammarEngine = struct {
    allocator: std.mem.Allocator,
    kb_store: *kb_mod.KbStore,

    // Reusable render buffer
    render_buf: []u8,
    render_capacity: i32,

    // Compile scratch
    slot_buf: []types.GrammarSlot,
    literal_buf: []LiteralRange,
    position_buf: []SlotPosition,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(allocator: std.mem.Allocator, kb_store: *kb_mod.KbStore) GrammarEngine {
    const render_buff = allocator.alloc(u8, 16384) catch &.{};
    const slots = allocator.alloc(types.GrammarSlot, 64) catch &.{};
    const lits = allocator.alloc(LiteralRange, 128) catch &.{};
    const positions = allocator.alloc(SlotPosition, 64) catch &.{};

    return .{
        .allocator = allocator,
        .kb_store = kb_store,
        .render_buf = render_buff,
        .render_capacity = @intCast(render_buff.len),
        .slot_buf = slots,
        .literal_buf = lits,
        .position_buf = positions,
    };
}

pub fn deinit(self: *GrammarEngine) void {
    if (self.render_buf.len > 0) self.allocator.free(self.render_buf);
    if (self.slot_buf.len > 0) self.allocator.free(self.slot_buf);
    if (self.literal_buf.len > 0) self.allocator.free(self.literal_buf);
    if (self.position_buf.len > 0) self.allocator.free(self.position_buf);
}

// ============================================================
// Compile — parse template into grammar struct + slot table
// ============================================================

pub fn compile(self: *GrammarEngine, template: []const u8, grammar_id: i32, session_id: i32) CompileResult {
    var n_slots: i32 = 0;
    var n_literals: i32 = 0;
    var pos: usize = 0;
    var lit_start: usize = 0;

    // Scan template for {slot_name} markers
    while (pos < template.len) {
        if (template[pos] == '{') {
            // Record literal range before this slot
            if (pos > lit_start and n_literals < @as(i32, @intCast(self.literal_buf.len))) {
                self.literal_buf[@intCast(n_literals)] = .{
                    .start = @intCast(lit_start),
                    .length = @intCast(pos - lit_start),
                };
                n_literals += 1;
            }

            // Find closing brace
            const brace_start = pos;
            pos += 1;
            const name_start = pos;
            while (pos < template.len and template[pos] != '}') : (pos += 1) {}

            if (pos >= template.len) {
                return CompileResult.failed(types.Status.err(.grammar, .invalid_template, @intCast(brace_start)));
            }

            // Extract slot name
            const name = template[name_start..pos];
            pos += 1; // skip '}'
            lit_start = pos;

            if (n_slots < @as(i32, @intCast(self.slot_buf.len)) and
                n_slots < @as(i32, @intCast(self.position_buf.len)))
            {
                // Check for duplicate slot names
                if (self.findSlotByName(name, n_slots)) {
                    return CompileResult.failed(types.Status.err(.grammar, .invalid_template, @intCast(brace_start)));
                }

                // Store slot name in text store
                const name_off = self.kb_store.textAppend(name);

                self.slot_buf[@intCast(n_slots)] = .{
                    .name_offset = name_off,
                    .name_length = @intCast(name.len),
                    .type = .text, // default, caller can override
                    .enum_values_offset = -1,
                    .enum_count = 0,
                    .kb_id = -1,
                    .kb_slot_id = -1,
                };

                self.position_buf[@intCast(n_slots)] = .{
                    .slot_index = @intCast(n_slots),
                    .template_offset = @intCast(brace_start),
                    .marker_length = @intCast(pos - brace_start),
                };

                n_slots += 1;
            }
        } else {
            pos += 1;
        }
    }

    // Final literal range
    if (lit_start < template.len and n_literals < @as(i32, @intCast(self.literal_buf.len))) {
        self.literal_buf[@intCast(n_literals)] = .{
            .start = @intCast(lit_start),
            .length = @intCast(template.len - lit_start),
        };
        n_literals += 1;
    }

    // Store template in text store
    const template_off = self.kb_store.textAppend(template);

    var grammar = std.mem.zeroes(types.Grammar);
    grammar.id = grammar_id;
    grammar.template_offset = template_off;
    grammar.template_length = @intCast(template.len);
    grammar.slots_count = @intCast(n_slots);
    grammar.validated = 1;
    grammar.created_at = kb_mod.currentTimestamp();
    grammar.creator_session_id = session_id;

    return .{
        .grammar = grammar,
        .slots = self.slot_buf[0..@intCast(n_slots)],
        .literals = self.literal_buf[0..@intCast(n_literals)],
        .slot_positions = self.position_buf[0..@intCast(n_slots)],
        .n_literals = n_literals,
        .n_slots = n_slots,
        .status = types.Status.ok(),
    };
}

pub fn validate(self: *GrammarEngine, grammar: *const types.Grammar) types.Status {
    // Re-parse template to verify structure
    if (grammar.template_length <= 0) return types.Status.err(.grammar, .invalid_template, 0);

    // Read template from text store
    var template_buf: [16384]u8 = undefined;
    const len: usize = @intCast(@min(grammar.template_length, 16384));
    const status = self.kb_store.textRead(grammar.template_offset, @intCast(len), template_buf[0..len]);
    if (status.isErr()) return status;

    // Verify all braces are matched
    var depth: i32 = 0;
    for (template_buf[0..len]) |b| {
        if (b == '{') depth += 1;
        if (b == '}') depth -= 1;
        if (depth < 0) return types.Status.err(.grammar, .invalid_template, -1);
    }
    if (depth != 0) return types.Status.err(.grammar, .invalid_template, -2);

    return types.Status.ok();
}

// ============================================================
// Render — fill slots and produce output
// ============================================================

pub fn render(self: *GrammarEngine, grammar: *const types.Grammar, compiled: *const CompileResult, fills: []const types.GrammarFill, config: *const RenderConfig, output: []u8) i32 {
    if (fills.len != @as(usize, @intCast(grammar.slots_count))) return -1;

    // Read template
    var template_buf: [16384]u8 = undefined;
    const tlen: usize = @intCast(@min(grammar.template_length, 16384));
    _ = self.kb_store.textRead(grammar.template_offset, @intCast(tlen), template_buf[0..tlen]);

    var out_pos: usize = 0;
    var template_pos: usize = 0;
    const max_out: usize = @intCast(@min(config.max_output_bytes, @as(i32, @intCast(output.len))));

    // Walk template: copy literals, substitute slots
    var slot_idx: usize = 0;
    for (compiled.slot_positions[0..@intCast(compiled.n_slots)], 0..) |sp, si| {
        _ = si;
        const marker_start: usize = @intCast(sp.template_offset);

        // Copy literal bytes before this slot
        if (marker_start > template_pos) {
            const lit_len = marker_start - template_pos;
            const copy_len = @min(lit_len, max_out - out_pos);
            if (copy_len > 0) {
                @memcpy(output[out_pos .. out_pos + copy_len], template_buf[template_pos .. template_pos + copy_len]);
                out_pos += copy_len;
            }
        }

        // Render slot fill
        if (slot_idx < fills.len and out_pos < max_out) {
            const fill = &fills[slot_idx];
            const rendered = self.renderFill(fill, config, output[out_pos..max_out]);
            out_pos += @intCast(rendered);
        }

        template_pos = @intCast(@as(i32, sp.template_offset) + sp.marker_length);
        slot_idx += 1;
    }

    // Copy remaining literal bytes after last slot
    if (template_pos < tlen) {
        const remaining = tlen - template_pos;
        const copy_len = @min(remaining, max_out - out_pos);
        if (copy_len > 0) {
            @memcpy(output[out_pos .. out_pos + copy_len], template_buf[template_pos .. template_pos + copy_len]);
            out_pos += copy_len;
        }
    }

    return @intCast(out_pos);
}

pub fn renderFromKb(self: *GrammarEngine, grammar: *const types.Grammar, compiled: *const CompileResult, mappings: []const types.GrammarKbMapping, config: *const RenderConfig, output: []u8) i32 {
    // Build fills from KB facts
    var fills: [64]types.GrammarFill = undefined;
    const n = @min(mappings.len, fills.len);

    for (mappings[0..n], 0..) |m, i| {
        const fact = self.kb_store.factRead(m.kb_id, m.slot_id);
        fills[i] = if (fact) |f| factToFill(@intCast(i), &f) else emptyFill(@intCast(i));
    }

    return self.render(grammar, compiled, fills[0..n], config, output);
}

// ============================================================
// Inheritance — walk KB tree for grammar
// ============================================================

pub fn inherit(self: *GrammarEngine, kb_id: i32, grammar_slot: i32) ?types.Grammar {
    var current = kb_id;
    var depth: i32 = 0;
    while (current >= 0 and depth < 100) {
        const kb = self.kb_store.getKb(current) orelse return null;
        if (kb.grammars_count > grammar_slot) {
            // Read grammar from grammar store
            if (kb.grammars_offset >= 0) {
                var grammar: types.Grammar = undefined;
                const offset = @as(i64, kb.grammars_offset + grammar_slot) * @sizeOf(types.Grammar);
                const dest: []u8 = @as([*]u8, @ptrCast(&grammar))[0..@sizeOf(types.Grammar)];
                const status = self.kb_store.bridge.downloadFromBuffer(.grammar_store, offset, dest);
                if (status.isOk() and grammar.isValid()) return grammar;
            }
        }
        current = kb.parent_id;
        depth += 1;
    }
    return null;
}

// ============================================================
// VDR / integer formatting — host-side
// ============================================================

pub fn q16ToString(value: types.Q16, buf: []u8) i32 {
    // Render as "integer_part.fractional_part"
    // integer = v / D, fractional = (v % D) * 10000 / D (4 decimal places)
    const d: i64 = types.Q16.D;
    const v: i64 = @as(i64, value.v);
    const int_part = @divTrunc(v, d);
    const frac_num = @mod(@mod(v, d) * 10000, d * 10000);
    const frac = @divTrunc(frac_num, d);

    return intToDecimal(@intCast(int_part), buf) + writeFractional(@intCast(frac), buf);
}

pub fn i32ToString(value: i32, buf: []u8) i32 {
    return intToDecimal(value, buf);
}

fn intToDecimal(value: i32, buf: []u8) i32 {
    if (buf.len == 0) return 0;
    var v = value;
    var pos: usize = 0;

    if (v < 0) {
        buf[pos] = '-';
        pos += 1;
        v = -v;
    }

    if (v == 0) {
        if (pos < buf.len) {
            buf[pos] = '0';
            return @intCast(pos + 1);
        }
        return @intCast(pos);
    }

    // Write digits in reverse
    var digits: [12]u8 = undefined;
    var n_digits: usize = 0;
    while (v > 0 and n_digits < 12) {
        digits[n_digits] = @intCast(@mod(v, 10) + '0');
        v = @divTrunc(v, 10);
        n_digits += 1;
    }

    // Reverse into output
    var i: usize = 0;
    while (i < n_digits and pos < buf.len) {
        buf[pos] = digits[n_digits - 1 - i];
        pos += 1;
        i += 1;
    }

    return @intCast(pos);
}

fn writeFractional(frac: i32, buf: []u8) i32 {
    if (frac == 0) return 0;
    // Find current end position (scan for first zero or end)
    var pos: usize = 0;
    while (pos < buf.len and buf[pos] != 0) : (pos += 1) {}
    if (pos >= buf.len) return 0;
    buf[pos] = '.';
    pos += 1;
    return intToDecimal(frac, buf[pos..]) + 1;
}

// ============================================================
// Helpers
// ============================================================

fn renderFill(self: *GrammarEngine, fill: *const types.GrammarFill, config: *const RenderConfig, output: []u8) i32 {
    _ = config;
    return switch (fill.fill_type) {
        .vdr_value => q16ToString(fill.vdr_value, output),
        .integer => i32ToString(fill.int_value, output),
        .text => blk: {
            if (fill.text_length <= 0) break :blk 0;
            const len: usize = @intCast(@min(fill.text_length, @as(i16, @intCast(output.len))));
            _ = self.kb_store.textRead(fill.text_offset, @intCast(len), output[0..len]);
            break :blk @intCast(len);
        },
        .@"enum" => blk: {
            // Render enum index as integer for now
            break :blk i32ToString(@as(i32, fill.enum_index), output);
        },
        .kb_ref => blk: {
            // Would need to read from KB — simplified
            break :blk 0;
        },
        .grammar => blk: {
            // Recursive grammar render — not implemented in flat version
            break :blk 0;
        },
    };
}

fn findSlotByName(self: *GrammarEngine, name: []const u8, n_slots: i32) bool {
    // Check if name already exists among declared slots
    // Compare by reading names back from text store
    var buf: [256]u8 = undefined;
    var i: i32 = 0;
    while (i < n_slots) : (i += 1) {
        const slot = &self.slot_buf[@intCast(i)];
        const len: usize = @intCast(@min(slot.name_length, 256));
        _ = self.kb_store.textRead(slot.name_offset, @intCast(len), buf[0..len]);
        if (std.mem.eql(u8, buf[0..len], name)) return true;
    }
    return false;
}

fn factToFill(slot_index: i16, fact: *const types.Fact) types.GrammarFill {
    return .{
        .slot_index = slot_index,
        .fill_type = switch (fact.tag) {
            .value => .vdr_value,
            .text => .text,
            .boolean, .@"enum", .counter, .timestamp, .reference => .integer,
            else => .text,
        },
        .vdr_value = fact.value,
        .text_offset = fact.value.v,
        .text_length = fact.value.r0,
        .int_value = fact.value.v,
        .enum_index = @intCast(fact.value.v),
    };
}

fn emptyFill(slot_index: i16) types.GrammarFill {
    return .{
        .slot_index = slot_index,
        .fill_type = .text,
        .vdr_value = types.Q16.zero(),
        .text_offset = 0,
        .text_length = 0,
        .int_value = 0,
        .enum_index = 0,
    };
}
