// ============================================================
// src/primitives/stack.zig
// ============================================================

const prim_types = @import("types.zig");
const VlpFact = prim_types.VlpFact;

pub const BoundedStack = struct {
    entries: []VlpFact,
    capacity: i32,
    top: i32,

    pub fn init(backing: []VlpFact) BoundedStack {
        for (backing) |*e| e.* = VlpFact{};
        return .{
            .entries = backing,
            .capacity = @intCast(backing.len),
            .top = 0,
        };
    }

    pub fn push(self: *BoundedStack, fact: VlpFact) bool {
        if (self.top >= self.capacity) return false;
        self.entries[@intCast(self.top)] = fact;
        self.top += 1;
        return true;
    }

    pub fn pop(self: *BoundedStack) ?VlpFact {
        if (self.top <= 0) return null;
        self.top -= 1;
        const val = self.entries[@intCast(self.top)];
        self.entries[@intCast(self.top)] = VlpFact{};
        return val;
    }

    pub fn peek(self: *const BoundedStack) ?VlpFact {
        if (self.top <= 0) return null;
        return self.entries[@intCast(self.top - 1)];
    }

    pub fn peekAt(self: *const BoundedStack, index: i32) ?VlpFact {
        if (index < 0 or index >= self.top) return null;
        return self.entries[@intCast(index)];
    }

    pub fn size(self: *const BoundedStack) i32 {
        return self.top;
    }

    pub fn isFull(self: *const BoundedStack) bool {
        return self.top >= self.capacity;
    }

    pub fn isEmpty(self: *const BoundedStack) bool {
        return self.top <= 0;
    }

    pub fn clear(self: *BoundedStack) void {
        var i: i32 = 0;
        while (i < self.top) : (i += 1) {
            self.entries[@intCast(i)] = VlpFact{};
        }
        self.top = 0;
    }
};
