// ============================================================
// src/primitives/counter.zig
// ============================================================

pub const Counter = struct {
    value: i32,
    min_val: i32,
    max_val: i32,
    initial: i32,

    pub fn init(min_val: i32, max_val: i32, initial: i32) Counter {
        const clamped = clamp(initial, min_val, max_val);
        return .{ .value = clamped, .min_val = min_val, .max_val = max_val, .initial = clamped };
    }

    pub fn get(self: *const Counter) i32 {
        return self.value;
    }

    pub fn increment(self: *Counter, amount: i32) void {
        const raw: i64 = @as(i64, self.value) + @as(i64, amount);
        if (raw > @as(i64, self.max_val)) {
            self.value = self.max_val;
        } else if (raw < @as(i64, self.min_val)) {
            self.value = self.min_val;
        } else {
            self.value = @intCast(raw);
        }
    }

    pub fn decrement(self: *Counter, amount: i32) void {
        self.increment(-amount);
    }

    pub fn reset(self: *Counter) void {
        self.value = self.initial;
    }

    pub fn set(self: *Counter, val: i32) void {
        self.value = clamp(val, self.min_val, self.max_val);
    }

    pub fn atMin(self: *const Counter) bool {
        return self.value <= self.min_val;
    }

    pub fn atMax(self: *const Counter) bool {
        return self.value >= self.max_val;
    }

    pub fn remaining(self: *const Counter) i32 {
        return self.max_val - self.value;
    }
};

fn clamp(v: i32, lo: i32, hi: i32) i32 {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
