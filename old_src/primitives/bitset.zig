// ============================================================
// src/primitives/bitset.zig
// ============================================================

pub const Bitset = struct {
    bits: []u8,
    n_bits: i32,

    pub fn init(backing: []u8, n_bits: i32) Bitset {
        @memset(backing, 0);
        return .{ .bits = backing, .n_bits = n_bits };
    }

    pub fn set(self: *Bitset, bit: i32) void {
        if (bit < 0 or bit >= self.n_bits) return;
        const byte_idx: usize = @intCast(@divTrunc(bit, 8));
        const bit_idx: u3 = @intCast(@mod(bit, 8));
        self.bits[byte_idx] |= @as(u8, 1) << bit_idx;
    }

    pub fn clearBit(self: *Bitset, bit: i32) void {
        if (bit < 0 or bit >= self.n_bits) return;
        const byte_idx: usize = @intCast(@divTrunc(bit, 8));
        const bit_idx: u3 = @intCast(@mod(bit, 8));
        self.bits[byte_idx] &= ~(@as(u8, 1) << bit_idx);
    }

    pub fn get(self: *const Bitset, bit: i32) bool {
        if (bit < 0 or bit >= self.n_bits) return false;
        const byte_idx: usize = @intCast(@divTrunc(bit, 8));
        const bit_idx: u3 = @intCast(@mod(bit, 8));
        return (self.bits[byte_idx] & (@as(u8, 1) << bit_idx)) != 0;
    }

    pub fn toggle(self: *Bitset, bit: i32) void {
        if (bit < 0 or bit >= self.n_bits) return;
        const byte_idx: usize = @intCast(@divTrunc(bit, 8));
        const bit_idx: u3 = @intCast(@mod(bit, 8));
        self.bits[byte_idx] ^= @as(u8, 1) << bit_idx;
    }

    pub fn popcount(self: *const Bitset) i32 {
        var total: i32 = 0;
        const byte_count: usize = @intCast(@divTrunc(self.n_bits + 7, 8));
        for (self.bits[0..byte_count]) |b| {
            total += @as(i32, @popCount(b));
        }
        return total;
    }

    pub fn clearAll(self: *Bitset) void {
        const byte_count: usize = @intCast(@divTrunc(self.n_bits + 7, 8));
        @memset(self.bits[0..byte_count], 0);
    }

    pub fn setAll(self: *Bitset) void {
        const full_bytes: usize = @intCast(@divTrunc(self.n_bits, 8));
        const remainder: u3 = @intCast(@mod(self.n_bits, 8));
        @memset(self.bits[0..full_bytes], 0xFF);
        if (remainder > 0) {
            self.bits[full_bytes] = (@as(u8, 1) << remainder) - 1;
        }
    }

    pub fn firstSet(self: *const Bitset) ?i32 {
        const byte_count: usize = @intCast(@divTrunc(self.n_bits + 7, 8));
        for (self.bits[0..byte_count], 0..) |b, bi| {
            if (b != 0) {
                const bit_in_byte: i32 = @intCast(@ctz(b));
                const result: i32 = @as(i32, @intCast(bi)) * 8 + bit_in_byte;
                if (result < self.n_bits) return result;
            }
        }
        return null;
    }

    pub fn firstClear(self: *const Bitset) ?i32 {
        const byte_count: usize = @intCast(@divTrunc(self.n_bits + 7, 8));
        for (self.bits[0..byte_count], 0..) |b, bi| {
            const inv = ~b;
            if (inv != 0) {
                const bit_in_byte: i32 = @intCast(@ctz(inv));
                const result: i32 = @as(i32, @intCast(bi)) * 8 + bit_in_byte;
                if (result < self.n_bits) return result;
            }
        }
        return null;
    }

    pub fn allSet(self: *const Bitset) bool {
        return self.popcount() == self.n_bits;
    }

    pub fn noneSet(self: *const Bitset) bool {
        return self.popcount() == 0;
    }

    pub fn bitAnd(self: *Bitset, other: *const Bitset) void {
        const byte_count: usize = @intCast(@divTrunc(@min(self.n_bits, other.n_bits) + 7, 8));
        for (self.bits[0..byte_count], other.bits[0..byte_count]) |*a, b| {
            a.* &= b;
        }
    }

    pub fn bitOr(self: *Bitset, other: *const Bitset) void {
        const byte_count: usize = @intCast(@divTrunc(@min(self.n_bits, other.n_bits) + 7, 8));
        for (self.bits[0..byte_count], other.bits[0..byte_count]) |*a, b| {
            a.* |= b;
        }
    }

    pub fn bitXor(self: *Bitset, other: *const Bitset) void {
        const byte_count: usize = @intCast(@divTrunc(@min(self.n_bits, other.n_bits) + 7, 8));
        for (self.bits[0..byte_count], other.bits[0..byte_count]) |*a, b| {
            a.* ^= b;
        }
    }

    pub fn bitNot(self: *Bitset) void {
        const full_bytes: usize = @intCast(@divTrunc(self.n_bits, 8));
        const remainder: u3 = @intCast(@mod(self.n_bits, 8));
        for (self.bits[0..full_bytes]) |*b| {
            b.* = ~b.*;
        }
        if (remainder > 0) {
            const mask: u8 = (@as(u8, 1) << remainder) - 1;
            self.bits[full_bytes] = (~self.bits[full_bytes]) & mask;
        }
    }
};
