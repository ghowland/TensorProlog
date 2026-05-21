// ============================================================
// src/builtins/time_ops.zig
// ============================================================

const conversion_mod = @import("conversion.zig");

pub fn timestampNow() i32 {
    return @intCast(@divTrunc(std.time.milliTimestamp(), 1000));
}

pub fn timestampDiff(a: i32, b: i32) i32 {
    return a -% b;
}

pub fn timestampAdd(ts: i32, seconds: i32) i32 {
    return ts +% seconds;
}

pub fn durationSeconds(seconds: i32) i32 {
    return seconds;
}

pub fn durationMinutes(minutes: i32) i32 {
    return minutes *% 60;
}

pub fn durationHours(hours: i32) i32 {
    return hours *% 3600;
}

pub fn durationDays(days: i32) i32 {
    return days *% 86400;
}

pub fn durationCompare(a: i32, b: i32) i32 {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

pub fn durationFormat(seconds: i32, output: []u8) i32 {
    const abs_s = if (seconds < 0) -seconds else seconds;
    var pos: usize = 0;
    if (seconds < 0 and pos < output.len) {
        output[pos] = '-';
        pos += 1;
    }
    const d = @divTrunc(abs_s, 86400);
    const h = @divTrunc(@mod(abs_s, 86400), 3600);
    const m = @divTrunc(@mod(abs_s, 3600), 60);
    const s = @mod(abs_s, 60);
    if (d > 0) {
        pos += conversion_mod.i32ToAscii(d, output[pos..]);
        if (pos < output.len) {
            output[pos] = 'd';
            pos += 1;
        }
    }
    pos += conversion_mod.i32ToAscii(h, output[pos..]);
    if (pos < output.len) {
        output[pos] = 'h';
        pos += 1;
    }
    pos += conversion_mod.i32ToAscii(m, output[pos..]);
    if (pos < output.len) {
        output[pos] = 'm';
        pos += 1;
    }
    pos += conversion_mod.i32ToAscii(s, output[pos..]);
    if (pos < output.len) {
        output[pos] = 's';
        pos += 1;
    }
    return @intCast(pos);
}

pub fn timestampFields(ts: i32) struct { year: i32, month: i32, day: i32, hour: i32, minute: i32, second: i32 } {
    return conversion_mod.timestampToFields(ts);
}

pub fn builtinTimestampNow(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.intResult(timestampNow());
}
pub fn builtinTimestampDiff(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(timestampDiff(args.int_args[0], args.int_args[1]));
}
pub fn builtinTimestampAdd(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(timestampAdd(args.int_args[0], args.int_args[1]));
}
pub fn builtinDurationSeconds(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(durationSeconds(args.int_args[0]));
}
pub fn builtinDurationMinutes(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(durationMinutes(args.int_args[0]));
}
pub fn builtinDurationHours(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(durationHours(args.int_args[0]));
}
pub fn builtinDurationDays(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(durationDays(args.int_args[0]));
}
pub fn builtinDurationCompare(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(durationCompare(args.int_args[0], args.int_args[1]));
}
pub fn builtinDurationFormat(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const len = durationFormat(args.int_args[0], args.text_out);
    args.text_out_len.* = len;
    return dispatch_mod.emptyResult();
}
pub fn builtinTimestampFields(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const fields = timestampFields(args.int_args[0]);
    return dispatch_mod.intResult(fields.year * 10000 + fields.month * 100 + fields.day);
}
