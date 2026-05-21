// ============================================================
// src/vdr/q335.zig
// ============================================================

pub const Limb6 = [6]i64;

pub fn zeroLimbs() Limb6 {
    return .{ 0, 0, 0, 0, 0, 0 };
}

pub fn oneLimb() Limb6 {
    var r = zeroLimbs();
    r[5] = @as(i64, 1) << 15;
    return r;
}

pub fn isZeroLimbs(a: Limb6) bool {
    return a[0] == 0 and a[1] == 0 and a[2] == 0 and a[3] == 0 and a[4] == 0 and a[5] == 0;
}

pub fn compareLimbs(a: Limb6, b: Limb6) i32 {
    var i: usize = 5;
    while (true) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
        if (i == 0) break;
        i -= 1;
    }
    return 0;
}

pub fn addLimbs(a: Limb6, b: Limb6) Limb6 {
    var r: Limb6 = undefined;
    var carry: i64 = 0;
    for (0..6) |i| {
        const s: i128 = @as(i128, a[i]) + @as(i128, b[i]) + @as(i128, carry);
        r[i] = @intCast(@mod(s, @as(i128, 1) << 64));
        carry = @intCast(@divTrunc(s, @as(i128, 1) << 64));
    }
    return r;
}

pub fn subLimbs(a: Limb6, b: Limb6) Limb6 {
    var r: Limb6 = undefined;
    var borrow: i64 = 0;
    for (0..6) |i| {
        const d: i128 = @as(i128, a[i]) - @as(i128, b[i]) - @as(i128, borrow);
        if (d < 0) {
            r[i] = @intCast(d + (@as(i128, 1) << 64));
            borrow = 1;
        } else {
            r[i] = @intCast(d);
            borrow = 0;
        }
    }
    return r;
}

pub fn mulLimbs(a: Limb6, b: Limb6) struct { hi: Limb6, lo: Limb6 } {
    var product: [12]i64 = .{0} ** 12;
    for (0..6) |i| {
        var carry: i128 = 0;
        for (0..6) |j| {
            const idx = i + j;
            const p: i128 = @as(i128, a[i]) * @as(i128, b[j]) + @as(i128, product[idx]) + carry;
            product[idx] = @intCast(@mod(p, @as(i128, 1) << 64));
            carry = @divTrunc(p, @as(i128, 1) << 64);
        }
        if (i + 6 < 12) {
            product[i + 6] +%= @intCast(carry);
        }
    }
    var lo: Limb6 = undefined;
    var hi: Limb6 = undefined;
    for (0..6) |i| {
        lo[i] = product[i];
        hi[i] = product[i + 6];
    }
    return .{ .hi = hi, .lo = lo };
}

pub fn shrLimbs(val: Limb6, shift: u32) Limb6 {
    if (shift == 0) return val;
    if (shift >= 384) return zeroLimbs();
    const ls = shift / 64;
    const bs: u6 = @intCast(shift % 64);
    var r = zeroLimbs();
    if (bs == 0) {
        for (0..6) |i| {
            const src = i + ls;
            if (src < 6) r[i] = val[src];
        }
    } else {
        for (0..6) |i| {
            const src = i + ls;
            if (src < 6) {
                const u: u64 = @bitCast(val[src]);
                r[i] = @bitCast(u >> bs);
            }
            if (src + 1 < 6) {
                const un: u64 = @bitCast(val[src + 1]);
                const comp: u6 = @intCast(64 - @as(u32, bs));
                const cur: u64 = @bitCast(r[i]);
                r[i] = @bitCast(cur | (un << comp));
            }
        }
    }
    return r;
}

pub fn shlLimbs(val: Limb6, shift: u32) Limb6 {
    if (shift == 0) return val;
    if (shift >= 384) return zeroLimbs();
    const ls = shift / 64;
    const bs: u6 = @intCast(shift % 64);
    var r = zeroLimbs();
    if (bs == 0) {
        var i: usize = 5;
        while (true) {
            if (i >= ls) r[i] = val[i - ls];
            if (i == 0) break;
            i -= 1;
        }
    } else {
        var i: usize = 5;
        while (true) {
            if (i >= ls) {
                const src = i - ls;
                const u: u64 = @bitCast(val[src]);
                r[i] = @bitCast(u << bs);
                if (src > 0) {
                    const up: u64 = @bitCast(val[src - 1]);
                    const comp: u6 = @intCast(64 - @as(u32, bs));
                    const cur: u64 = @bitCast(r[i]);
                    r[i] = @bitCast(cur | (up >> comp));
                }
            }
            if (i == 0) break;
            i -= 1;
        }
    }
    return r;
}

pub fn maskLowBits(val: Limb6, n_bits: u32) Limb6 {
    if (n_bits >= 384) return val;
    if (n_bits == 0) return zeroLimbs();
    var r = val;
    const full = n_bits / 64;
    const partial: u6 = @intCast(n_bits % 64);
    for (0..6) |i| {
        if (i > full) {
            r[i] = 0;
        } else if (i == full) {
            if (partial == 0) {
                r[i] = 0;
            } else {
                const mask: u64 = (@as(u64, 1) << partial) - 1;
                const u: u64 = @bitCast(r[i]);
                r[i] = @bitCast(u & mask);
            }
        }
    }
    return r;
}

pub const Q335 = struct {
    v: Limb6,
    r0: Limb6,
    r1: Limb6,
    r2: Limb6,
    r3: Limb6,

    const D_BITS: u32 = 335;

    pub fn add(a: Q335, b: Q335) Q335 {
        return .{
            .v = addLimbs(a.v, b.v),
            .r0 = addLimbs(a.r0, b.r0),
            .r1 = addLimbs(a.r1, b.r1),
            .r2 = addLimbs(a.r2, b.r2),
            .r3 = addLimbs(a.r3, b.r3),
        };
    }

    pub fn sub(a: Q335, b: Q335) Q335 {
        return .{
            .v = subLimbs(a.v, b.v),
            .r0 = subLimbs(a.r0, b.r0),
            .r1 = subLimbs(a.r1, b.r1),
            .r2 = subLimbs(a.r2, b.r2),
            .r3 = subLimbs(a.r3, b.r3),
        };
    }

    pub fn mul(a: Q335, b: Q335) Q335 {
        const p = mulLimbs(a.v, b.v);
        const new_r0 = maskLowBits(p.lo, D_BITS);
        const lo_shifted = shrLimbs(p.lo, D_BITS);
        const hi_shifted = shlLimbs(p.hi, 384 - D_BITS);
        const new_v = addLimbs(lo_shifted, hi_shifted);
        return .{
            .v = new_v,
            .r0 = new_r0,
            .r1 = a.r0,
            .r2 = a.r1,
            .r3 = a.r2,
        };
    }

    pub fn div(a: Q335, b: Q335) Q335 {
        if (isZeroLimbs(b.v)) return Q335.zero();
        // stub — full Q335 division requires long division on limb arrays
        return .{
            .v = zeroLimbs(),
            .r0 = a.v,
            .r1 = zeroLimbs(),
            .r2 = zeroLimbs(),
            .r3 = zeroLimbs(),
        };
    }

    pub fn compare(a: Q335, b: Q335) i32 {
        const vc = compareLimbs(a.v, b.v);
        if (vc != 0) return vc;
        const r0c = compareLimbs(a.r0, b.r0);
        if (r0c != 0) return r0c;
        const r1c = compareLimbs(a.r1, b.r1);
        if (r1c != 0) return r1c;
        const r2c = compareLimbs(a.r2, b.r2);
        if (r2c != 0) return r2c;
        return compareLimbs(a.r3, b.r3);
    }

    pub fn eql(a: Q335, b: Q335) bool {
        return compare(a, b) == 0;
    }

    pub fn zero() Q335 {
        const z = zeroLimbs();
        return .{ .v = z, .r0 = z, .r1 = z, .r2 = z, .r3 = z };
    }

    pub fn one() Q335 {
        const z = zeroLimbs();
        return .{ .v = oneLimb(), .r0 = z, .r1 = z, .r2 = z, .r3 = z };
    }

    pub fn remainderMagnitude(self: Q335) Limb6 {
        return self.r0;
    }

    pub fn compact(self: *Q335) void {
        const top = shrLimbs(self.r0, D_BITS);
        if (!isZeroLimbs(top)) {
            self.v = addLimbs(self.v, top);
            self.r0 = maskLowBits(self.r0, D_BITS);
        }
    }

    pub fn fromI64(val: i64) Q335 {
        var limbs = zeroLimbs();
        limbs[0] = val;
        const z = zeroLimbs();
        return .{ .v = shlLimbs(limbs, D_BITS), .r0 = z, .r1 = z, .r2 = z, .r3 = z };
    }
};
