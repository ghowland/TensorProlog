// ============================================================
// src/engine/scratchpad.zig
// ============================================================

const q16_mod = @import("../vdr/q16.zig");
const vdr_types = @import("../vdr/types.zig");

const Q16 = q16_mod.Q16;
const VlpStatus = vdr_types.VlpStatus;
const VlpGrantClass = vdr_types.VlpGrantClass;

pub const Scratchpad = struct {
    buf: []u8,
    len: i32,

    tokens: []i32,
    token_count: i32,

    pub fn init(buf: []u8, tokens: []i32) Scratchpad {
        return .{
            .buf = buf,
            .len = 0,
            .tokens = tokens,
            .token_count = 0,
        };
    }

    pub fn writeResult(self: *Scratchpad, data: []const u8) void {
        self.appendBytes(data);
        self.appendToken(-200);
    }

    pub fn writeError(self: *Scratchpad, status: VlpStatus) void {
        self.appendToken(-300);
        self.appendToken(@intFromEnum(status));
    }

    pub fn writeDenied(self: *Scratchpad, kb_id: i32) void {
        self.appendToken(-400);
        self.appendToken(kb_id);
    }

    pub fn writeGrantDenied(self: *Scratchpad, class: VlpGrantClass) void {
        self.appendToken(-500);
        self.appendToken(@intFromEnum(class));
    }

    pub fn writeValue(self: *Scratchpad, val: Q16) void {
        self.appendToken(-600);
        self.appendToken(val.v);
        self.appendToken(@as(i32, val.r0));
    }

    pub fn writeInt(self: *Scratchpad, val: i32) void {
        self.appendToken(-700);
        self.appendToken(val);
    }

    pub fn clear(self: *Scratchpad) void {
        self.len = 0;
        self.token_count = 0;
    }

    pub fn getTokens(self: *const Scratchpad) []const i32 {
        if (self.token_count <= 0) return &[_]i32{};
        return self.tokens[0..@intCast(self.token_count)];
    }

    pub fn getBytes(self: *const Scratchpad) []const u8 {
        if (self.len <= 0) return &[_]u8{};
        return self.buf[0..@intCast(self.len)];
    }

    pub fn isEmpty(self: *const Scratchpad) bool {
        return self.len <= 0 and self.token_count <= 0;
    }

    fn appendBytes(self: *Scratchpad, data: []const u8) void {
        for (data) |b| {
            if (self.len >= @as(i32, @intCast(self.buf.len))) return;
            self.buf[@intCast(self.len)] = b;
            self.len += 1;
        }
    }

    fn appendToken(self: *Scratchpad, tok: i32) void {
        if (self.token_count >= @as(i32, @intCast(self.tokens.len))) return;
        self.tokens[@intCast(self.token_count)] = tok;
        self.token_count += 1;
    }
};
