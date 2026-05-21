// ============================================================
// src/builtins/text.zig
// ============================================================

const std = @import("std");

const dispatch = @import("dispatch.zig");
const BuiltinArgs = dispatch.BuiltinArgs;
const BuiltinResult = dispatch.BuiltinResult;

pub fn textReverse(input: []const u8, output: []u8) i32 {
    const n = @min(input.len, output.len);
    for (0..n) |i| {
        output[i] = input[n - 1 - i];
    }
    return @intCast(n);
}

pub fn textSplit(input: []const u8, delimiter: u8, parts: [][]const u8) i32 {
    var count: usize = 0;
    var start: usize = 0;

    for (input, 0..) |c, i| {
        if (c == delimiter) {
            if (count < parts.len) {
                parts[count] = input[start..i];
                count += 1;
            }
            start = i + 1;
        }
    }

    if (start <= input.len and count < parts.len) {
        parts[count] = input[start..];
        count += 1;
    }

    return @intCast(count);
}

pub fn textContains(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;

    const limit = haystack.len - needle.len + 1;
    for (0..limit) |i| {
        if (std.mem.eql(u8, haystack[i .. i + needle.len], needle)) {
            return true;
        }
    }
    return false;
}

pub fn textReplace(input: []const u8, old: []const u8, new: []const u8, output: []u8) i32 {
    if (old.len == 0) {
        const n = @min(input.len, output.len);
        @memcpy(output[0..n], input[0..n]);
        return @intCast(n);
    }

    var out_pos: usize = 0;
    var in_pos: usize = 0;

    while (in_pos <= input.len - old.len) {
        if (std.mem.eql(u8, input[in_pos .. in_pos + old.len], old)) {
            const copy_len = @min(new.len, output.len - out_pos);
            @memcpy(output[out_pos .. out_pos + copy_len], new[0..copy_len]);
            out_pos += copy_len;
            in_pos += old.len;
        } else {
            if (out_pos < output.len) {
                output[out_pos] = input[in_pos];
                out_pos += 1;
            }
            in_pos += 1;
        }
    }

    while (in_pos < input.len and out_pos < output.len) {
        output[out_pos] = input[in_pos];
        out_pos += 1;
        in_pos += 1;
    }

    return @intCast(out_pos);
}

pub fn textJoin(parts: []const []const u8, separator: []const u8, output: []u8) i32 {
    var out_pos: usize = 0;

    for (parts, 0..) |part, i| {
        if (i > 0) {
            const sep_len = @min(separator.len, output.len - out_pos);
            @memcpy(output[out_pos .. out_pos + sep_len], separator[0..sep_len]);
            out_pos += sep_len;
        }
        const part_len = @min(part.len, output.len - out_pos);
        @memcpy(output[out_pos .. out_pos + part_len], part[0..part_len]);
        out_pos += part_len;
    }

    return @intCast(out_pos);
}

pub fn textTrim(input: []const u8, output: []u8) i32 {
    var start: usize = 0;
    while (start < input.len and (input[start] == ' ' or input[start] == '\t' or input[start] == '\n' or input[start] == '\r')) {
        start += 1;
    }

    var end = input.len;
    while (end > start and (input[end - 1] == ' ' or input[end - 1] == '\t' or input[end - 1] == '\n' or input[end - 1] == '\r')) {
        end -= 1;
    }

    const trimmed = input[start..end];
    const n = @min(trimmed.len, output.len);
    @memcpy(output[0..n], trimmed[0..n]);
    return @intCast(n);
}

pub fn textUpper(input: []u8) void {
    for (input) |*c| {
        if (c.* >= 'a' and c.* <= 'z') {
            c.* -= 32;
        }
    }
}

pub fn textLower(input: []u8) void {
    for (input) |*c| {
        if (c.* >= 'A' and c.* <= 'Z') {
            c.* += 32;
        }
    }
}

pub fn textStartsWith(input: []const u8, prefix: []const u8) bool {
    if (prefix.len > input.len) return false;
    return std.mem.eql(u8, input[0..prefix.len], prefix);
}

pub fn textEndsWith(input: []const u8, suffix: []const u8) bool {
    if (suffix.len > input.len) return false;
    return std.mem.eql(u8, input[input.len - suffix.len ..], suffix);
}

pub fn textIndexOf(haystack: []const u8, needle: []const u8) i32 {
    if (needle.len == 0) return 0;
    if (needle.len > haystack.len) return -1;

    const limit = haystack.len - needle.len + 1;
    for (0..limit) |i| {
        if (std.mem.eql(u8, haystack[i .. i + needle.len], needle)) {
            return @intCast(i);
        }
    }
    return -1;
}

pub fn textSubstring(input: []const u8, start: i32, end: i32, output: []u8) i32 {
    const s: usize = @intCast(@max(start, 0));
    const e: usize = @intCast(@min(end, @as(i32, @intCast(input.len))));
    if (s >= e) return 0;

    const slice = input[s..e];
    const n = @min(slice.len, output.len);
    @memcpy(output[0..n], slice[0..n]);
    return @intCast(n);
}

pub fn textRepeat(input: []const u8, count: i32, output: []u8) i32 {
    var out_pos: usize = 0;
    const c: usize = @intCast(@max(count, 0));

    for (0..c) |_| {
        const copy_len = @min(input.len, output.len - out_pos);
        if (copy_len == 0) break;
        @memcpy(output[out_pos .. out_pos + copy_len], input[0..copy_len]);
        out_pos += copy_len;
    }

    return @intCast(out_pos);
}

pub fn textPadLeft(input: []const u8, width: i32, pad_char: u8, output: []u8) i32 {
    const w: usize = @intCast(@max(width, 0));
    const total = @max(w, input.len);
    const n = @min(total, output.len);

    if (input.len >= w) {
        @memcpy(output[0..@min(input.len, n)], input[0..@min(input.len, n)]);
        return @intCast(@min(input.len, n));
    }

    const pad_count = w - input.len;
    const pad_n = @min(pad_count, n);
    @memset(output[0..pad_n], pad_char);

    const copy_n = @min(input.len, n - pad_n);
    @memcpy(output[pad_n .. pad_n + copy_n], input[0..copy_n]);

    return @intCast(pad_n + copy_n);
}

pub fn textPadRight(input: []const u8, width: i32, pad_char: u8, output: []u8) i32 {
    const w: usize = @intCast(@max(width, 0));
    const total = @max(w, input.len);
    const n = @min(total, output.len);

    const copy_n = @min(input.len, n);
    @memcpy(output[0..copy_n], input[0..copy_n]);

    if (copy_n < n) {
        const pad_n = @min(w - input.len, n - copy_n);
        @memset(output[copy_n .. copy_n + pad_n], pad_char);
        return @intCast(copy_n + pad_n);
    }

    return @intCast(copy_n);
}

pub fn textCharAt(input: []const u8, index: i32) ?u8 {
    const i: usize = @intCast(index);
    if (i >= input.len) return null;
    return input[i];
}

pub fn textLength(input: []const u8) i32 {
    return @intCast(input.len);
}

pub fn builtinTextReverse(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch.errorResult(.err_command_parse);
    const len = textReverse(args.text_in[0], args.text_out);
    args.text_out_len.* = len;
    return dispatch.emptyResult();
}

pub fn builtinTextSplit(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch.emptyResult();
}

pub fn builtinTextContains(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 2) return dispatch.errorResult(.err_command_parse);
    return dispatch.boolResult(textContains(args.text_in[0], args.text_in[1]));
}

pub fn builtinTextReplace(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 3) return dispatch.errorResult(.err_command_parse);
    const len = textReplace(args.text_in[0], args.text_in[1], args.text_in[2], args.text_out);
    args.text_out_len.* = len;
    return dispatch.emptyResult();
}

pub fn builtinTextJoin(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch.emptyResult();
}

pub fn builtinTextTrim(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch.errorResult(.err_command_parse);
    const len = textTrim(args.text_in[0], args.text_out);
    args.text_out_len.* = len;
    return dispatch.emptyResult();
}

pub fn builtinTextUpper(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch.errorResult(.err_command_parse);
    const src = args.text_in[0];
    const n = @min(src.len, args.text_out.len);
    @memcpy(args.text_out[0..n], src[0..n]);
    textUpper(args.text_out[0..n]);
    args.text_out_len.* = @intCast(n);
    return dispatch.emptyResult();
}

pub fn builtinTextLower(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch.errorResult(.err_command_parse);
    const src = args.text_in[0];
    const n = @min(src.len, args.text_out.len);
    @memcpy(args.text_out[0..n], src[0..n]);
    textLower(args.text_out[0..n]);
    args.text_out_len.* = @intCast(n);
    return dispatch.emptyResult();
}

pub fn builtinTextStartsWith(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 2) return dispatch.errorResult(.err_command_parse);
    return dispatch.boolResult(textStartsWith(args.text_in[0], args.text_in[1]));
}

pub fn builtinTextEndsWith(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 2) return dispatch.errorResult(.err_command_parse);
    return dispatch.boolResult(textEndsWith(args.text_in[0], args.text_in[1]));
}

pub fn builtinTextIndexOf(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 2) return dispatch.errorResult(.err_command_parse);
    return dispatch.intResult(textIndexOf(args.text_in[0], args.text_in[1]));
}

pub fn builtinTextSubstring(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1 or args.int_arg_count < 2) return dispatch.errorResult(.err_command_parse);
    const len = textSubstring(args.text_in[0], args.int_args[0], args.int_args[1], args.text_out);
    args.text_out_len.* = len;
    return dispatch.emptyResult();
}

pub fn builtinTextRepeat(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1 or args.int_arg_count < 1) return dispatch.errorResult(.err_command_parse);
    const len = textRepeat(args.text_in[0], args.int_args[0], args.text_out);
    args.text_out_len.* = len;
    return dispatch.emptyResult();
}

pub fn builtinTextPadLeft(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1 or args.int_arg_count < 2) return dispatch.errorResult(.err_command_parse);
    const len = textPadLeft(args.text_in[0], args.int_args[0], @intCast(args.int_args[1]), args.text_out);
    args.text_out_len.* = len;
    return dispatch.emptyResult();
}

pub fn builtinTextPadRight(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1 or args.int_arg_count < 2) return dispatch.errorResult(.err_command_parse);
    const len = textPadRight(args.text_in[0], args.int_args[0], @intCast(args.int_args[1]), args.text_out);
    args.text_out_len.* = len;
    return dispatch.emptyResult();
}

pub fn builtinTextCharAt(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1 or args.int_arg_count < 1) return dispatch.errorResult(.err_command_parse);
    const ch = textCharAt(args.text_in[0], args.int_args[0]);
    if (ch) |c| {
        args.text_out[0] = c;
        args.text_out_len.* = 1;
    } else {
        args.text_out_len.* = 0;
    }
    return dispatch.emptyResult();
}

pub fn builtinTextLength(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch.errorResult(.err_command_parse);
    return dispatch.intResult(textLength(args.text_in[0]));
}
