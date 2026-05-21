// ============================================================
// src/kb/text_store.zig
// ============================================================

pub const TextRef = struct {
    offset: i32,
    length: i16,
};

pub const TextStore = struct {
    data: []u8,
    len: i32,

    pub fn init(backing: []u8) TextStore {
        return .{ .data = backing, .len = 0 };
    }

    pub fn append(self: *TextStore, bytes: []const u8) ?TextRef {
        const needed: i32 = @intCast(bytes.len);
        if (self.len + needed > @as(i32, @intCast(self.data.len))) return null;
        const off = self.len;
        @memcpy(self.data[@intCast(off)..@intCast(off + needed)], bytes);
        self.len += needed;
        return .{ .offset = off, .length = @intCast(bytes.len) };
    }

    pub fn read(self: *const TextStore, ref: TextRef) ?[]const u8 {
        if (ref.offset < 0 or ref.length <= 0) return null;
        const end: i32 = ref.offset + @as(i32, ref.length);
        if (end > self.len) return null;
        return self.data[@intCast(ref.offset)..@intCast(end)];
    }
};
