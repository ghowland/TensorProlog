// ============================================================
// src/vdr/q16.zig
// ============================================================

pub const Q16 = struct {
    v: i32,
    r0: i16,
    _pad: i16 = 0,

    pub const D: i32 = 65536;
    pub const D_i64: i64 = 65536;
    pub const HALF_D: i32 = 32768;

    pub fn add(a: Q16, b: Q16) Q16 {
        const sr: i32 = @as(i32, a.r0) + @as(i32, b.r0);
        if (sr >= D) return .{ .v = a.v + b.v + 1, .r0 = @intCast(sr - D) };
        if (sr < 0) return .{ .v = a.v + b.v - 1, .r0 = @intCast(sr + D) };
        return .{ .v = a.v + b.v, .r0 = @intCast(sr) };
    }

    pub fn sub(a: Q16, b: Q16) Q16 {
        const dr: i32 = @as(i32, a.r0) - @as(i32, b.r0);
        if (dr < 0) return .{ .v = a.v - b.v - 1, .r0 = @intCast(dr + D) };
        if (dr >= D) return .{ .v = a.v - b.v + 1, .r0 = @intCast(dr - D) };
        return .{ .v = a.v - b.v, .r0 = @intCast(dr) };
    }

    pub fn mul(a: Q16, b: Q16) Q16 {
        const p: i64 = @as(i64, a.v) * @as(i64, b.v);
        return .{
            .v = @intCast(@divTrunc(p, D_i64)),
            .r0 = @intCast(@rem(p, D_i64)),
        };
    }

    pub fn div(a: Q16, b: Q16) Q16 {
        if (b.v == 0) return .{ .v = 0, .r0 = 0 };
        const n: i64 = @as(i64, a.v) * D_i64;
        const bv: i64 = @as(i64, b.v);
        return .{
            .v = @intCast(@divTrunc(n, bv)),
            .r0 = @intCast(@rem(n, bv)),
        };
    }

    pub fn compare(a: Q16, b: Q16) i32 {
        if (a.v < b.v) return -1;
        if (a.v > b.v) return 1;
        if (a.r0 < b.r0) return -1;
        if (a.r0 > b.r0) return 1;
        return 0;
    }

    pub fn eql(a: Q16, b: Q16) bool {
        return a.v == b.v and a.r0 == b.r0;
    }

    pub fn fromFraction(num: i32, den: i32) Q16 {
        if (den == 0) return .{ .v = 0, .r0 = 0 };
        const n: i64 = @as(i64, num) * D_i64;
        const d: i64 = @as(i64, den);
        return .{
            .v = @intCast(@divTrunc(n, d)),
            .r0 = @intCast(@rem(n, d)),
        };
    }

    pub fn toFraction(self: Q16) struct { num: i32, den: i32 } {
        return .{ .num = self.v, .den = D };
    }

    pub fn zero() Q16 {
        return .{ .v = 0, .r0 = 0 };
    }

    pub fn one() Q16 {
        return .{ .v = D, .r0 = 0 };
    }

    pub fn negate(self: Q16) Q16 {
        if (self.r0 == 0) return .{ .v = -self.v, .r0 = 0 };
        return .{ .v = -self.v - 1, .r0 = @intCast(D - @as(i32, self.r0)) };
    }

    pub fn abs_val(self: Q16) Q16 {
        if (self.v < 0 or (self.v == 0 and self.r0 < 0)) return self.negate();
        return self;
    }

    pub fn sign(self: Q16) i32 {
        if (self.v > 0 or (self.v == 0 and self.r0 > 0)) return 1;
        if (self.v < 0 or (self.v == 0 and self.r0 < 0)) return -1;
        return 0;
    }

    pub fn min_val(a: Q16, b: Q16) Q16 {
        if (compare(a, b) <= 0) return a;
        return b;
    }

    pub fn max_val(a: Q16, b: Q16) Q16 {
        if (compare(a, b) >= 0) return a;
        return b;
    }

    pub fn remainderMagnitude(self: Q16) i32 {
        const r: i32 = @as(i32, self.r0);
        if (r < 0) return -r;
        return r;
    }

    pub fn compact(self: *Q16) void {
        _ = self;
    }

    pub fn softmax(logits: []const Q16, probs: []Q16) void {
        if (logits.len == 0) return;
        const n = logits.len;

        var min_v: i32 = logits[0].v;
        for (logits[1..]) |l| {
            if (l.v < min_v) min_v = l.v;
        }

        var squares: [4096]i64 = undefined;
        var sum: i64 = 0;
        for (logits, 0..) |l, i| {
            const s: i64 = @as(i64, l.v - min_v);
            squares[i] = s * s;
            sum += squares[i];
        }

        if (sum == 0) {
            const each: i32 = @divTrunc(D, @as(i32, @intCast(n)));
            var assigned: i32 = 0;
            for (probs[0 .. n - 1]) |*p| {
                p.* = .{ .v = each, .r0 = 0 };
                assigned += each;
            }
            probs[n - 1] = .{ .v = D - assigned, .r0 = 0 };
            return;
        }

        var assigned: i32 = 0;
        for (0..n - 1) |i| {
            const pv: i64 = @divTrunc(squares[i] * D_i64, sum);
            probs[i] = .{ .v = @intCast(pv), .r0 = 0 };
            assigned += @intCast(pv);
        }
        probs[n - 1] = .{ .v = D - assigned, .r0 = 0 };
    }

    pub fn dotProduct(a: []const Q16, b: []const Q16) Q16 {
        const len = @min(a.len, b.len);
        var acc: i64 = 0;
        for (0..len) |i| {
            acc += @as(i64, a[i].v) * @as(i64, b[i].v);
        }
        return .{
            .v = @intCast(@divTrunc(acc, D_i64)),
            .r0 = @intCast(@rem(acc, D_i64)),
        };
    }
};
