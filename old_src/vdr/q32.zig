// ============================================================
// src/vdr/q32.zig
// ============================================================

pub const Q32 = struct {
    v: i64,
    r0: i32,
    r1: i32,

    pub const D: i64 = 4294967296;
    pub const D_i128: i128 = 4294967296;

    pub fn add(a: Q32, b: Q32) Q32 {
        const sr: i64 = @as(i64, a.r0) + @as(i64, b.r0);
        if (sr >= D) return .{ .v = a.v + b.v + 1, .r0 = @intCast(sr - D), .r1 = a.r1 +% b.r1 };
        if (sr < 0) return .{ .v = a.v + b.v - 1, .r0 = @intCast(sr + D), .r1 = a.r1 +% b.r1 };
        return .{ .v = a.v + b.v, .r0 = @intCast(sr), .r1 = a.r1 +% b.r1 };
    }

    pub fn sub(a: Q32, b: Q32) Q32 {
        const dr: i64 = @as(i64, a.r0) - @as(i64, b.r0);
        if (dr < 0) return .{ .v = a.v - b.v - 1, .r0 = @intCast(dr + D), .r1 = a.r1 -% b.r1 };
        if (dr >= D) return .{ .v = a.v - b.v + 1, .r0 = @intCast(dr - D), .r1 = a.r1 -% b.r1 };
        return .{ .v = a.v - b.v, .r0 = @intCast(dr), .r1 = a.r1 -% b.r1 };
    }

    pub fn mul(a: Q32, b: Q32) Q32 {
        const p: i128 = @as(i128, a.v) * @as(i128, b.v);
        return .{
            .v = @intCast(@divTrunc(p, D_i128)),
            .r0 = @intCast(@rem(p, D_i128)),
            .r1 = 0,
        };
    }

    pub fn div(a: Q32, b: Q32) Q32 {
        if (b.v == 0) return .{ .v = 0, .r0 = 0, .r1 = 0 };
        const n: i128 = @as(i128, a.v) * D_i128;
        const bv: i128 = @as(i128, b.v);
        return .{
            .v = @intCast(@divTrunc(n, bv)),
            .r0 = @intCast(@rem(n, bv)),
            .r1 = 0,
        };
    }

    pub fn compare(a: Q32, b: Q32) i32 {
        if (a.v < b.v) return -1;
        if (a.v > b.v) return 1;
        if (a.r0 < b.r0) return -1;
        if (a.r0 > b.r0) return 1;
        if (a.r1 < b.r1) return -1;
        if (a.r1 > b.r1) return 1;
        return 0;
    }

    pub fn eql(a: Q32, b: Q32) bool {
        return a.v == b.v and a.r0 == b.r0 and a.r1 == b.r1;
    }

    pub fn fromFraction(num: i64, den: i64) Q32 {
        if (den == 0) return .{ .v = 0, .r0 = 0, .r1 = 0 };
        const n: i128 = @as(i128, num) * D_i128;
        const d: i128 = @as(i128, den);
        return .{
            .v = @intCast(@divTrunc(n, d)),
            .r0 = @intCast(@rem(n, d)),
            .r1 = 0,
        };
    }

    pub fn toFraction(self: Q32) struct { num: i64, den: i64 } {
        return .{ .num = self.v, .den = @intCast(D) };
    }

    pub fn zero() Q32 {
        return .{ .v = 0, .r0 = 0, .r1 = 0 };
    }

    pub fn one() Q32 {
        return .{ .v = @intCast(D), .r0 = 0, .r1 = 0 };
    }

    pub fn negate(self: Q32) Q32 {
        if (self.r0 == 0 and self.r1 == 0) return .{ .v = -self.v, .r0 = 0, .r1 = 0 };
        return .{ .v = -self.v - 1, .r0 = @intCast(D - @as(i64, self.r0)), .r1 = -self.r1 };
    }

    pub fn abs_val(self: Q32) Q32 {
        if (self.v < 0) return self.negate();
        return self;
    }

    pub fn remainderMagnitude(self: Q32) i64 {
        const r: i64 = @as(i64, self.r0);
        if (r < 0) return -r;
        return r;
    }

    pub fn compact(self: *Q32) void {
        _ = self;
    }
};
