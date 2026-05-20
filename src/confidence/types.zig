// ============================================================
// src/confidence/types.zig
// ============================================================

const q16_mod = @import("../vdr/q16.zig");
const vdr_types = @import("../vdr/types.zig");

pub const Q16 = q16_mod.Q16;
pub const VlpSourceType = vdr_types.VlpSourceType;

pub const CONFIDENCE_TABLE: [11]Q16 = .{
    .{ .v = 65536, .r0 = 0 },
    .{ .v = 65536, .r0 = 0 },
    .{ .v = 64225, .r0 = 0 },
    .{ .v = 62259, .r0 = 0 },
    .{ .v = 62259, .r0 = 0 },
    .{ .v = 55705, .r0 = 0 },
    .{ .v = 52428, .r0 = 0 },
    .{ .v = 45875, .r0 = 0 },
    .{ .v = 32768, .r0 = 0 },
    .{ .v = 19660, .r0 = 0 },
    .{ .v = 0, .r0 = 0 },
};
