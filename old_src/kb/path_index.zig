// ============================================================
// src/kb/path_index.zig
// ============================================================

const std = @import("std");

pub const PathIndex = struct {
    keys: []i32,
    values: []i32,
    occupied: []bool,
    cap: i32,
    count: i32,

    pub fn init(keys: []i32, values: []i32, occupied: []bool) PathIndex {
        @memset(occupied, false);
        @memset(keys, 0);
        @memset(values, -1);
        return .{ .keys = keys, .values = values, .occupied = occupied, .cap = @intCast(keys.len), .count = 0 };
    }

    pub fn insert(self: *PathIndex, path: []const u8, kb_id: i32) bool {
        if (self.count >= @divTrunc(self.cap * 7, 10)) return false;
        const h = hash(path);
        var idx = @mod(h, self.cap);
        if (idx < 0) idx += self.cap;
        var probe: i32 = 0;
        while (probe < self.cap) : (probe += 1) {
            const i: usize = @intCast(idx);
            if (!self.occupied[i]) {
                self.keys[i] = h;
                self.values[i] = kb_id;
                self.occupied[i] = true;
                self.count += 1;
                return true;
            }
            if (self.keys[i] == h) {
                self.values[i] = kb_id;
                return true;
            }
            idx = @mod(idx + 1, self.cap);
        }
        return false;
    }

    pub fn lookup(self: *const PathIndex, path: []const u8) ?i32 {
        const h = hash(path);
        var idx = @mod(h, self.cap);
        if (idx < 0) idx += self.cap;
        var probe: i32 = 0;
        while (probe < self.cap) : (probe += 1) {
            const i: usize = @intCast(idx);
            if (!self.occupied[i]) return null;
            if (self.keys[i] == h) return self.values[i];
            idx = @mod(idx + 1, self.cap);
        }
        return null;
    }

    pub fn remove(self: *PathIndex, path: []const u8) bool {
        const h = hash(path);
        var idx = @mod(h, self.cap);
        if (idx < 0) idx += self.cap;
        var probe: i32 = 0;
        while (probe < self.cap) : (probe += 1) {
            const i: usize = @intCast(idx);
            if (!self.occupied[i]) return false;
            if (self.keys[i] == h) {
                self.occupied[i] = false;
                self.count -= 1;
                rehashFrom(self, idx);
                return true;
            }
            idx = @mod(idx + 1, self.cap);
        }
        return false;
    }

    fn rehashFrom(self: *PathIndex, start: i32) void {
        var idx = @mod(start + 1, self.cap);
        while (true) {
            const i: usize = @intCast(idx);
            if (!self.occupied[i]) break;
            const k = self.keys[i];
            const v = self.values[i];
            self.occupied[i] = false;
            self.count -= 1;
            var ins = @mod(k, self.cap);
            if (ins < 0) ins += self.cap;
            var p: i32 = 0;
            while (p < self.cap) : (p += 1) {
                const ii: usize = @intCast(ins);
                if (!self.occupied[ii]) {
                    self.keys[ii] = k;
                    self.values[ii] = v;
                    self.occupied[ii] = true;
                    self.count += 1;
                    break;
                }
                ins = @mod(ins + 1, self.cap);
            }
            idx = @mod(idx + 1, self.cap);
        }
    }

    pub fn hash(path: []const u8) i32 {
        var h: u32 = 2166136261;
        for (path) |b| {
            h ^= @as(u32, b);
            h *%= 16777619;
        }
        return @bitCast(h);
    }
};
