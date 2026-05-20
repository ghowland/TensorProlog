// ============================================================
// src/builtins/integer_ops.zig
// ============================================================

pub fn intAdd(a: i32, b: i32) i32 {
    return a +% b;
}

pub fn intSub(a: i32, b: i32) i32 {
    return a -% b;
}

pub fn intMul(a: i32, b: i32) i32 {
    return a *% b;
}

pub fn intDiv(a: i32, b: i32) i32 {
    if (b == 0) return 0;
    return @divTrunc(a, b);
}

pub fn intMod(a: i32, b: i32) i32 {
    if (b == 0) return 0;
    return @mod(a, b);
}

pub fn intAbs(a: i32) i32 {
    return if (a < 0) -a else a;
}

pub fn intSign(a: i32) i32 {
    if (a > 0) return 1;
    if (a < 0) return -1;
    return 0;
}

pub fn intMin(a: i32, b: i32) i32 {
    return if (a < b) a else b;
}

pub fn intMax(a: i32, b: i32) i32 {
    return if (a > b) a else b;
}

pub fn intClamp(val: i32, lo: i32, hi: i32) i32 {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

pub fn intPow(base: i32, exp: i32) i32 {
    if (exp <= 0) return 1;
    var result: i32 = 1;
    var b = base;
    var e = exp;
    while (e > 0) {
        if (e & 1 == 1) result *%= b;
        b *%= b;
        e >>= 1;
    }
    return result;
}

pub fn intFactorial(n: i32) i64 {
    if (n <= 1) return 1;
    var result: i64 = 1;
    var i: i64 = 2;
    while (i <= @as(i64, @intCast(n))) {
        result *%= i;
        i += 1;
    }
    return result;
}

pub fn intChoose(n: i32, k: i32) i64 {
    if (k < 0 or k > n) return 0;
    var kk = k;
    if (kk > n - kk) kk = n - kk;
    var result: i64 = 1;
    for (0..@as(usize, @intCast(kk))) |i| {
        const ii: i64 = @intCast(i);
        result = @divTrunc(result * (@as(i64, @intCast(n)) - ii), ii + 1);
    }
    return result;
}

pub fn bitAnd(a: i32, b: i32) i32 {
    return a & b;
}

pub fn bitOr(a: i32, b: i32) i32 {
    return a | b;
}

pub fn bitXor(a: i32, b: i32) i32 {
    return a ^ b;
}

pub fn bitNot(a: i32) i32 {
    return ~a;
}

pub fn bitShiftLeft(a: i32, amount: i32) i32 {
    if (amount < 0 or amount >= 32) return 0;
    const u: u5 = @intCast(amount);
    return a << u;
}

pub fn bitShiftRight(a: i32, amount: i32) i32 {
    if (amount < 0 or amount >= 32) return 0;
    const u: u5 = @intCast(amount);
    return a >> u;
}

pub fn bitPopcount(a: i32) i32 {
    const u: u32 = @bitCast(a);
    return @intCast(@popCount(u));
}

pub fn bitReverse(a: i32) i32 {
    const u: u32 = @bitCast(a);
    return @bitCast(@bitReverse(u));
}

pub fn builtinIntAdd(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(intAdd(args.int_args[0], args.int_args[1]));
}
pub fn builtinIntSub(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(intSub(args.int_args[0], args.int_args[1]));
}
pub fn builtinIntMul(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(intMul(args.int_args[0], args.int_args[1]));
}
pub fn builtinIntDiv(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(intDiv(args.int_args[0], args.int_args[1]));
}
pub fn builtinIntMod(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(intMod(args.int_args[0], args.int_args[1]));
}
pub fn builtinIntAbs(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(intAbs(args.int_args[0]));
}
pub fn builtinIntSign(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(intSign(args.int_args[0]));
}
pub fn builtinIntMin(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(intMin(args.int_args[0], args.int_args[1]));
}
pub fn builtinIntMax(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(intMax(args.int_args[0], args.int_args[1]));
}
pub fn builtinIntClamp(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 3) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(intClamp(args.int_args[0], args.int_args[1], args.int_args[2]));
}
pub fn builtinIntPow(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(intPow(args.int_args[0], args.int_args[1]));
}
pub fn builtinIntFactorial(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(@intCast(intFactorial(args.int_args[0])));
}
pub fn builtinIntChoose(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(@intCast(intChoose(args.int_args[0], args.int_args[1])));
}
pub fn builtinBitAnd(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(bitAnd(args.int_args[0], args.int_args[1]));
}
pub fn builtinBitOr(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(bitOr(args.int_args[0], args.int_args[1]));
}
pub fn builtinBitXor(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(bitXor(args.int_args[0], args.int_args[1]));
}
pub fn builtinBitNot(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(bitNot(args.int_args[0]));
}
pub fn builtinBitShl(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(bitShiftLeft(args.int_args[0], args.int_args[1]));
}
pub fn builtinBitShr(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(bitShiftRight(args.int_args[0], args.int_args[1]));
}
pub fn builtinBitPopcount(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(bitPopcount(args.int_args[0]));
}
pub fn builtinBitReverse(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.intResult(bitReverse(args.int_args[0]));
}
