// ============================================================
// src/primitives/lock.zig
// ============================================================

pub const Lock = struct {
    held: bool,
    holder: i32,

    pub fn init() Lock {
        return .{ .held = false, .holder = -1 };
    }

    pub fn acquire(self: *Lock, holder_id: i32) bool {
        if (self.held) return false;
        self.held = true;
        self.holder = holder_id;
        return true;
    }

    pub fn release(self: *Lock, holder_id: i32) bool {
        if (!self.held) return false;
        if (self.holder != holder_id) return false;
        self.held = false;
        self.holder = -1;
        return true;
    }

    pub fn forceRelease(self: *Lock) void {
        self.held = false;
        self.holder = -1;
    }

    pub fn isHeld(self: *const Lock) bool {
        return self.held;
    }

    pub fn getHolder(self: *const Lock) i32 {
        return self.holder;
    }
};
