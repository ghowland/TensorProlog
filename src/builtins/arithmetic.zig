// ============================================================
// src/builtins/arithmetic.zig
// ============================================================

const q16_mod = @import("../vdr/q16.zig");
const Q16A = q16_mod.Q16;
const dispatch_mod = @import("dispatch.zig");
const BuiltinArgsA = dispatch_mod.BuiltinArgs;
const BuiltinResultA = dispatch_mod.BuiltinResult;

pub fn arithAdd(a: Q16A, b: Q16A) Q16A {
    return Q16A.add(a, b);
}

pub fn arithSub(a: Q16A, b: Q16A) Q16A {
    return Q16A.sub(a, b);
}

pub fn arithMul(a: Q16A, b: Q16A) Q16A {
    return Q16A.mul(a, b);
}

pub fn arithDiv(a: Q16A, b: Q16A) Q16A {
    return Q16A.div(a, b);
}

pub fn arithPow(base: Q16A, exp: i32) Q16A {
    if (exp == 0) return Q16A.one();
    if (exp == 1) return base;

    var result = Q16A.one();
    var e = exp;
    var b = base;

    if (e < 0) {
        b = Q16A.div(Q16A.one(), b);
        e = -e;
    }

    while (e > 0) {
        if (e & 1 == 1) {
            result = Q16A.mul(result, b);
        }
        b = Q16A.mul(b, b);
        e >>= 1;
    }

    return result;
}

pub fn arithReciprocal(a: Q16A) Q16A {
    return Q16A.div(Q16A.one(), a);
}

pub fn arithCompare(a: Q16A, b: Q16A) i32 {
    return Q16A.compare(a, b);
}

pub fn arithEqual(a: Q16A, b: Q16A) bool {
    return Q16A.eql(a, b);
}

pub fn arithMin(a: Q16A, b: Q16A) Q16A {
    return Q16A.min_val(a, b);
}

pub fn arithMax(a: Q16A, b: Q16A) Q16A {
    return Q16A.max_val(a, b);
}

pub fn arithSign(a: Q16A) i32 {
    return Q16A.sign(a);
}

pub fn arithIsZero(a: Q16A) bool {
    return a.v == 0 and a.r0 == 0;
}

pub fn arithFloor(a: Q16A) i32 {
    if (a.v >= 0) {
        return @divTrunc(a.v, Q16A.D);
    }
    const d = @divTrunc(a.v, Q16A.D);
    if (@mod(a.v, Q16A.D) != 0) {
        return d - 1;
    }
    return d;
}

pub fn arithCeil(a: Q16A) i32 {
    if (a.v <= 0) {
        return @divTrunc(a.v, Q16A.D);
    }
    const d = @divTrunc(a.v, Q16A.D);
    if (@mod(a.v, Q16A.D) != 0) {
        return d + 1;
    }
    return d;
}

pub fn arithRound(a: Q16A) i32 {
    const half_d = @divTrunc(Q16A.D, 2);
    const remainder = @mod(a.v, Q16A.D);
    const floor_val = arithFloor(a);

    if (a.v >= 0) {
        if (remainder >= half_d) return floor_val + 1;
        return floor_val;
    } else {
        if (-remainder >= half_d) return floor_val;
        return floor_val + 1;
    }
}

pub fn arithNumerator(a: Q16A) i32 {
    return a.v;
}

pub fn arithDenominator() i32 {
    return Q16A.D;
}

pub fn arithAbs(a: Q16A) Q16A {
    return Q16A.abs_val(a);
}

pub fn arithNegate(a: Q16A) Q16A {
    return Q16A.negate(a);
}

pub fn arithClamp(a: Q16A, min_v: Q16A, max_v: Q16A) Q16A {
    if (Q16A.compare(a, min_v) < 0) return min_v;
    if (Q16A.compare(a, max_v) > 0) return max_v;
    return a;
}

pub fn arithFromInt(val: i32) Q16A {
    return .{ .v = val *% Q16A.D, .r0 = 0 };
}

pub fn arithToInt(a: Q16A) i32 {
    return @divTrunc(a.v, Q16A.D);
}

pub fn arithLerp(a: Q16A, b: Q16A, t: Q16A) Q16A {
    const one_minus_t = Q16A.sub(Q16A.one(), t);
    const term_a = Q16A.mul(a, one_minus_t);
    const term_b = Q16A.mul(b, t);
    return Q16A.add(term_a, term_b);
}

pub fn arithMidpoint(a: Q16A, b: Q16A) Q16A {
    const sum = Q16A.add(a, b);
    const half: Q16A = .{ .v = @divTrunc(Q16A.D, 2), .r0 = 0 };
    return Q16A.mul(sum, half);
}

pub fn arithDistance(a: Q16A, b: Q16A) Q16A {
    return Q16A.abs_val(Q16A.sub(a, b));
}

fn getArg(args: *BuiltinArgsA, index: usize) Q16A {
    return args.input_facts[index].value;
}

pub fn builtinAdd(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.factResult(arithAdd(getArg(args, 0), getArg(args, 1)));
}

pub fn builtinSub(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.factResult(arithSub(getArg(args, 0), getArg(args, 1)));
}

pub fn builtinMul(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.factResult(arithMul(getArg(args, 0), getArg(args, 1)));
}

pub fn builtinDiv(args: *BuiltinArgsA) BuiltinResultA {
    const b = getArg(args, 1);
    if (b.v == 0) return dispatch_mod.errorResult(.err_remainder_overflow);
    return dispatch_mod.factResult(arithDiv(getArg(args, 0), b));
}

pub fn builtinPow(args: *BuiltinArgsA) BuiltinResultA {
    const exp = @divTrunc(getArg(args, 1).v, Q16A.D);
    return dispatch_mod.factResult(arithPow(getArg(args, 0), exp));
}

pub fn builtinReciprocal(args: *BuiltinArgsA) BuiltinResultA {
    const a = getArg(args, 0);
    if (a.v == 0) return dispatch_mod.errorResult(.err_remainder_overflow);
    return dispatch_mod.factResult(arithReciprocal(a));
}

pub fn builtinCompare(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.intResult(arithCompare(getArg(args, 0), getArg(args, 1)));
}

pub fn builtinEqual(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.boolResult(arithEqual(getArg(args, 0), getArg(args, 1)));
}

pub fn builtinMin(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.factResult(arithMin(getArg(args, 0), getArg(args, 1)));
}

pub fn builtinMax(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.factResult(arithMax(getArg(args, 0), getArg(args, 1)));
}

pub fn builtinSign(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.intResult(arithSign(getArg(args, 0)));
}

pub fn builtinIsZero(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.boolResult(arithIsZero(getArg(args, 0)));
}

pub fn builtinFloor(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.intResult(arithFloor(getArg(args, 0)));
}

pub fn builtinCeil(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.intResult(arithCeil(getArg(args, 0)));
}

pub fn builtinRound(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.intResult(arithRound(getArg(args, 0)));
}

pub fn builtinNumerator(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.intResult(arithNumerator(getArg(args, 0)));
}

pub fn builtinDenominator(args: *BuiltinArgsA) BuiltinResultA {
    _ = args;
    return dispatch_mod.intResult(arithDenominator());
}

pub fn builtinAbs(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.factResult(arithAbs(getArg(args, 0)));
}

pub fn builtinNegate(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.factResult(arithNegate(getArg(args, 0)));
}

pub fn builtinClamp(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.factResult(arithClamp(getArg(args, 0), getArg(args, 1), getArg(args, 2)));
}

pub fn builtinFromInt(args: *BuiltinArgsA) BuiltinResultA {
    const val = @divTrunc(getArg(args, 0).v, Q16A.D);
    return dispatch_mod.factResult(arithFromInt(val));
}

pub fn builtinToInt(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.intResult(arithToInt(getArg(args, 0)));
}

pub fn builtinLerp(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.factResult(arithLerp(getArg(args, 0), getArg(args, 1), getArg(args, 2)));
}

pub fn builtinMidpoint(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.factResult(arithMidpoint(getArg(args, 0), getArg(args, 1)));
}

pub fn builtinDistance(args: *BuiltinArgsA) BuiltinResultA {
    return dispatch_mod.factResult(arithDistance(getArg(args, 0), getArg(args, 1)));
}
