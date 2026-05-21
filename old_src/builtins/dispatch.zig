// ============================================================
// src/builtins/dispatch.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");
const kb_types = @import("../kb/types.zig");
const kb_store_mod = @import("../kb/store.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;
const VlpFact = kb_types.VlpFact;
const VlpFactTag = types.VlpFactTag;
const KBStore = kb_store_mod.KBStore;

pub const BuiltinFn = *const fn (*BuiltinArgs) BuiltinResult;

pub const BuiltinEntry = struct {
    id: i32,
    name_offset: i32,
    name_length: i16,
    fn_ptr: BuiltinFn,
    pure: bool,
    input_count: i8,
    output_type: VlpFactTag,
};

pub const BuiltinArgs = struct {
    store: *KBStore,
    input_facts: [8]VlpFact,
    input_count: i32,
    target_kb_id: i32,
    target_slot_id: i32,
    text_in: [4][]const u8,
    text_in_count: i32,
    text_out: []u8,
    text_out_len: *i32,
    int_args: [4]i32,
    int_arg_count: i32,
};

pub const BuiltinResult = struct {
    status: VlpStatus,
    output_fact: VlpFact,
    output_kb_id: i32,
    output_slot_id: i32,
    output_int: i32,
    output_bool: bool,
};

pub fn emptyResult() BuiltinResult {
    return .{
        .status = .ok,
        .output_fact = .{
            .tag = .empty,
            .value = Q16.zero(),
            .provenance = .{
                .source_type = .vdr_computation,
                .source_kb_id = -1,
                .source_slot_id = -1,
                .confidence = .{ .v = Q16.D, .r0 = 0 },
                .timestamp = 0,
                .derivation_rule_id = -1,
            },
        },
        .output_kb_id = -1,
        .output_slot_id = -1,
        .output_int = 0,
        .output_bool = false,
    };
}

pub fn factResult(val: Q16) BuiltinResult {
    var r = emptyResult();
    r.output_fact.tag = .value;
    r.output_fact.value = val;
    return r;
}

pub fn intResult(val: i32) BuiltinResult {
    var r = emptyResult();
    r.output_int = val;
    return r;
}

pub fn boolResult(val: bool) BuiltinResult {
    var r = emptyResult();
    r.output_bool = val;
    return r;
}

pub fn errorResult(status: VlpStatus) BuiltinResult {
    var r = emptyResult();
    r.status = status;
    return r;
}

pub const BuiltinTable = struct {
    entries: [512]?BuiltinEntry,
    count: i32,
    name_buf: [16384]u8,
    name_buf_len: i32,

    pub fn init() BuiltinTable {
        const t = BuiltinTable{
            .entries = [_]?BuiltinEntry{null} ** 512,
            .count = 0,
            .name_buf = [_]u8{0} ** 16384,
            .name_buf_len = 0,
        };
        return t;
    }

    pub fn register(self: *BuiltinTable, id: i32, name: []const u8, fn_ptr: BuiltinFn, pure: bool, input_count: i8, output_type: VlpFactTag) void {
        const idx: usize = @intCast(id);
        if (idx >= self.entries.len) return;

        const offset = self.name_buf_len;
        const len: i16 = @intCast(name.len);
        const o: usize = @intCast(offset);
        const e: usize = o + name.len;
        if (e > self.name_buf.len) return;

        @memcpy(self.name_buf[o..e], name);
        self.name_buf_len += @intCast(name.len);

        self.entries[idx] = .{
            .id = id,
            .name_offset = offset,
            .name_length = len,
            .fn_ptr = fn_ptr,
            .pure = pure,
            .input_count = input_count,
            .output_type = output_type,
        };
        self.count += 1;
    }

    pub fn dispatch(self: *const BuiltinTable, id: i32, args: *BuiltinArgs) BuiltinResult {
        const idx: usize = @intCast(id);
        if (idx >= self.entries.len) return errorResult(.err_invalid_qbasis);

        const entry = self.entries[idx] orelse return errorResult(.err_invalid_qbasis);

        if (args.input_count < entry.input_count) {
            return errorResult(.err_command_parse);
        }

        return entry.fn_ptr(args);
    }

    pub fn lookup(self: *const BuiltinTable, name: []const u8) ?i32 {
        for (self.entries, 0..) |maybe_entry, i| {
            const entry = maybe_entry orelse continue;
            const o: usize = @intCast(entry.name_offset);
            const l: usize = @intCast(entry.name_length);
            const stored_name = self.name_buf[o .. o + l];
            if (std.mem.eql(u8, stored_name, name)) {
                return @intCast(i);
            }
        }
        return null;
    }

    pub fn isRegistered(self: *const BuiltinTable, id: i32) bool {
        const idx: usize = @intCast(id);
        if (idx >= self.entries.len) return false;
        return self.entries[idx] != null;
    }

    pub fn isPure(self: *const BuiltinTable, id: i32) bool {
        const idx: usize = @intCast(id);
        if (idx >= self.entries.len) return false;
        const entry = self.entries[idx] orelse return false;
        return entry.pure;
    }
};

pub fn registerTextBuiltins(table: *BuiltinTable) void {
    const text = @import("text.zig");
    table.register(100, "text_reverse", text.builtinTextReverse, true, 1, .text);
    table.register(101, "text_split", text.builtinTextSplit, true, 1, .text);
    table.register(102, "text_contains", text.builtinTextContains, true, 1, .boolean);
    table.register(103, "text_replace", text.builtinTextReplace, true, 1, .text);
    table.register(104, "text_join", text.builtinTextJoin, true, 1, .text);
    table.register(105, "text_trim", text.builtinTextTrim, true, 1, .text);
    table.register(106, "text_upper", text.builtinTextUpper, true, 1, .text);
    table.register(107, "text_lower", text.builtinTextLower, true, 1, .text);
    table.register(108, "text_starts_with", text.builtinTextStartsWith, true, 1, .boolean);
    table.register(109, "text_ends_with", text.builtinTextEndsWith, true, 1, .boolean);
    table.register(110, "text_index_of", text.builtinTextIndexOf, true, 1, .value);
    table.register(111, "text_substring", text.builtinTextSubstring, true, 1, .text);
    table.register(112, "text_repeat", text.builtinTextRepeat, true, 1, .text);
    table.register(113, "text_pad_left", text.builtinTextPadLeft, true, 1, .text);
    table.register(114, "text_pad_right", text.builtinTextPadRight, true, 1, .text);
    table.register(115, "text_char_at", text.builtinTextCharAt, true, 1, .text);
    table.register(116, "text_length", text.builtinTextLength, true, 1, .value);
}

pub fn registerArithmeticBuiltins(table: *BuiltinTable) void {
    const arith = @import("arithmetic.zig");
    table.register(0, "arith_add", arith.builtinAdd, true, 2, .value);
    table.register(1, "arith_sub", arith.builtinSub, true, 2, .value);
    table.register(2, "arith_mul", arith.builtinMul, true, 2, .value);
    table.register(3, "arith_div", arith.builtinDiv, true, 2, .value);
    table.register(4, "arith_pow", arith.builtinPow, true, 2, .value);
    table.register(5, "arith_reciprocal", arith.builtinReciprocal, true, 1, .value);
    table.register(6, "arith_compare", arith.builtinCompare, true, 2, .value);
    table.register(7, "arith_equal", arith.builtinEqual, true, 2, .boolean);
    table.register(8, "arith_min", arith.builtinMin, true, 2, .value);
    table.register(9, "arith_max", arith.builtinMax, true, 2, .value);
    table.register(10, "arith_sign", arith.builtinSign, true, 1, .value);
    table.register(11, "arith_is_zero", arith.builtinIsZero, true, 1, .boolean);
    table.register(12, "arith_floor", arith.builtinFloor, true, 1, .value);
    table.register(13, "arith_ceil", arith.builtinCeil, true, 1, .value);
    table.register(14, "arith_round", arith.builtinRound, true, 1, .value);
    table.register(15, "arith_numerator", arith.builtinNumerator, true, 1, .value);
    table.register(16, "arith_denominator", arith.builtinDenominator, true, 0, .value);
    table.register(17, "arith_abs", arith.builtinAbs, true, 1, .value);
    table.register(18, "arith_negate", arith.builtinNegate, true, 1, .value);
    table.register(19, "arith_clamp", arith.builtinClamp, true, 3, .value);
    table.register(20, "arith_from_int", arith.builtinFromInt, true, 1, .value);
    table.register(21, "arith_to_int", arith.builtinToInt, true, 1, .value);
    table.register(22, "arith_lerp", arith.builtinLerp, true, 3, .value);
    table.register(23, "arith_midpoint", arith.builtinMidpoint, true, 2, .value);
    table.register(24, "arith_distance", arith.builtinDistance, true, 2, .value);
}

pub fn registerAllBuiltins(table: *BuiltinTable) void {
    registerArithmeticBuiltins(table);
    registerTextBuiltins(table);
}
