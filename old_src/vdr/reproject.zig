// ============================================================
// src/vdr/reproject.zig
// ============================================================

const q16_mod = @import("q16.zig");
const q32_mod = @import("q32.zig");
const q335_mod = @import("q335.zig");

const Q16 = q16_mod.Q16;
const Q32 = q32_mod.Q32;
const Q335 = q335_mod.Q335;

pub fn q16ToQ32(val: Q16) Q32 {
    return .{
        .v = @as(i64, val.v) * 65536,
        .r0 = @intCast(@as(i64, val.r0) * 65536),
        .r1 = 0,
    };
}

pub fn q32ToQ16(val: Q32) Q16 {
    return .{
        .v = @intCast(@divTrunc(val.v, 65536)),
        .r0 = @intCast(@rem(val.v, 65536)),
    };
}

pub fn q16ToQ335(val: Q16) Q335 {
    var limbs = q335_mod.zeroLimbs();
    limbs[0] = @as(i64, val.v);
    var rl = q335_mod.zeroLimbs();
    rl[0] = @as(i64, val.r0);
    const z = q335_mod.zeroLimbs();
    return .{
        .v = q335_mod.shlLimbs(limbs, 319),
        .r0 = q335_mod.shlLimbs(rl, 319),
        .r1 = z,
        .r2 = z,
        .r3 = z,
    };
}

pub fn q335ToQ16(val: Q335) Q16 {
    const shifted = q335_mod.shrLimbs(val.v, 319);
    const low = q335_mod.maskLowBits(val.v, 319);
    const rs = q335_mod.shrLimbs(low, 303);
    return .{
        .v = @intCast(shifted[0]),
        .r0 = @intCast(@mod(rs[0], 65536)),
    };
}

pub fn q32ToQ335(val: Q32) Q335 {
    var limbs = q335_mod.zeroLimbs();
    limbs[0] = val.v;
    var rl = q335_mod.zeroLimbs();
    rl[0] = @as(i64, val.r0);
    const z = q335_mod.zeroLimbs();
    return .{
        .v = q335_mod.shlLimbs(limbs, 303),
        .r0 = q335_mod.shlLimbs(rl, 303),
        .r1 = z,
        .r2 = z,
        .r3 = z,
    };
}

pub fn q335ToQ32(val: Q335) Q32 {
    const shifted = q335_mod.shrLimbs(val.v, 303);
    const low = q335_mod.maskLowBits(val.v, 303);
    const rs = q335_mod.shrLimbs(low, 271);
    return .{
        .v = shifted[0],
        .r0 = @intCast(@mod(rs[0], 4294967296)),
        .r1 = 0,
    };
}
