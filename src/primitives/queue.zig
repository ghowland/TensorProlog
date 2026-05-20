// ============================================================
// src/primitives/queue.zig
// ============================================================

const prim_types = @import("types.zig");
const VlpFact = prim_types.VlpFact;

pub const BoundedQueue = struct {
    entries: []VlpFact,
    capacity: i32,
    count: i32,
    head: i32,
    tail: i32,

    pub fn init(backing: []VlpFact) BoundedQueue {
        for (backing) |*e| e.* = VlpFact{};
        return .{
            .entries = backing,
            .capacity = @intCast(backing.len),
            .count = 0,
            .head = 0,
            .tail = 0,
        };
    }

    pub fn push(self: *BoundedQueue, fact: VlpFact) bool {
        if (self.count >= self.capacity) return false;
        self.entries[@intCast(self.tail)] = fact;
        self.tail = @mod(self.tail + 1, self.capacity);
        self.count += 1;
        return true;
    }

    pub fn pop(self: *BoundedQueue) ?VlpFact {
        if (self.count <= 0) return null;
        const val = self.entries[@intCast(self.head)];
        self.entries[@intCast(self.head)] = VlpFact{};
        self.head = @mod(self.head + 1, self.capacity);
        self.count -= 1;
        return val;
    }

    pub fn peek(self: *const BoundedQueue) ?VlpFact {
        if (self.count <= 0) return null;
        return self.entries[@intCast(self.head)];
    }

    pub fn peekAt(self: *const BoundedQueue, index: i32) ?VlpFact {
        if (index < 0 or index >= self.count) return null;
        const pos = @mod(self.head + index, self.capacity);
        return self.entries[@intCast(pos)];
    }

    pub fn size(self: *const BoundedQueue) i32 {
        return self.count;
    }

    pub fn isFull(self: *const BoundedQueue) bool {
        return self.count >= self.capacity;
    }

    pub fn isEmpty(self: *const BoundedQueue) bool {
        return self.count <= 0;
    }

    pub fn clear(self: *BoundedQueue) void {
        for (self.entries) |*e| e.* = VlpFact{};
        self.count = 0;
        self.head = 0;
        self.tail = 0;
    }
};
