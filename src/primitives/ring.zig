// ============================================================
// src/primitives/ring.zig
// ============================================================

const prim_types = @import("types.zig");
const VlpFact = prim_types.VlpFact;

pub const RingBuffer = struct {
    entries: []VlpFact,
    capacity: i32,
    count: i32,
    write_pos: i32,

    pub fn init(backing: []VlpFact) RingBuffer {
        for (backing) |*e| e.* = VlpFact{};
        return .{
            .entries = backing,
            .capacity = @intCast(backing.len),
            .count = 0,
            .write_pos = 0,
        };
    }

    pub fn write(self: *RingBuffer, fact: VlpFact) void {
        self.entries[@intCast(self.write_pos)] = fact;
        self.write_pos = @mod(self.write_pos + 1, self.capacity);
        if (self.count < self.capacity) {
            self.count += 1;
        }
    }

    pub fn read(self: *const RingBuffer, index: i32) ?VlpFact {
        if (index < 0 or index >= self.count) return null;
        const oldest_pos = if (self.count < self.capacity) 0 else self.write_pos;
        const pos = @mod(oldest_pos + index, self.capacity);
        return self.entries[@intCast(pos)];
    }

    pub fn newest(self: *const RingBuffer) ?VlpFact {
        if (self.count <= 0) return null;
        return self.read(self.count - 1);
    }

    pub fn oldest(self: *const RingBuffer) ?VlpFact {
        if (self.count <= 0) return null;
        return self.read(0);
    }

    pub fn size(self: *const RingBuffer) i32 {
        return self.count;
    }

    pub fn isFull(self: *const RingBuffer) bool {
        return self.count >= self.capacity;
    }

    pub fn isEmpty(self: *const RingBuffer) bool {
        return self.count <= 0;
    }

    pub fn clear(self: *RingBuffer) void {
        for (self.entries) |*e| e.* = VlpFact{};
        self.count = 0;
        self.write_pos = 0;
    }
};
