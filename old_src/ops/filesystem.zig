// ============================================================
// src/ops/filesystem.zig
// ============================================================

const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const kb_types = @import("../kb/types.zig");
const grant_mod = @import("../safety/grant.zig");
const q16 = @import("../vdr/q16.zig");

const Q16 = q16.Q16;
const KBStore = kb_store_mod.KBStore;
const VlpFact = kb_types.VlpFact;

pub fn fsRead(path: []const u8, output: []u8) struct { len: i32, status: VlpStatus } {
    const file = std.fs.cwd().openFile(path, .{}) catch return .{ .len = 0, .status = .err_grant_denied };
    defer file.close();
    const n = file.read(output) catch return .{ .len = 0, .status = .err_snapshot_failed };
    return .{ .len = @intCast(n), .status = .ok };
}

pub fn fsWrite(path: []const u8, data: []const u8) VlpStatus {
    const file = std.fs.cwd().createFile(path, .{}) catch return .err_grant_denied;
    defer file.close();
    file.writeAll(data) catch return .err_snapshot_failed;
    return .ok;
}

pub fn fsAppend(path: []const u8, data: []const u8) VlpStatus {
    const file = std.fs.cwd().openFile(path, .{ .mode = .write_only }) catch {
        return fsWrite(path, data);
    };
    defer file.close();
    file.seekFromEnd(0) catch return .err_snapshot_failed;
    file.writeAll(data) catch return .err_snapshot_failed;
    return .ok;
}

pub fn fsDelete(path: []const u8) VlpStatus {
    std.fs.cwd().deleteFile(path) catch return .err_grant_denied;
    return .ok;
}

pub fn fsStat(path: []const u8) struct { size: i64, exists: bool, status: VlpStatus } {
    const file = std.fs.cwd().openFile(path, .{}) catch return .{ .size = 0, .exists = false, .status = .ok };
    defer file.close();
    const stat = file.stat() catch return .{ .size = 0, .exists = true, .status = .err_snapshot_failed };
    return .{ .size = @intCast(stat.size), .exists = true, .status = .ok };
}

pub fn fsReadToKB(store: *KBStore, kb_id: i32, slot: i32, path: []const u8) VlpStatus {
    var buf: [65536]u8 = undefined;
    const result = fsRead(path, &buf);
    if (result.status != .ok) return result.status;
    const rl: usize = @intCast(result.len);
    const ref = store.text.append(buf[0..rl]);
    const fact = VlpFact{
        .tag = .text,
        .value = .{ .v = ref.offset, .r0 = @intCast(ref.length) },
        .provenance = opsProvenance(kb_id, slot),
    };
    _ = fact_mod.factAssert(store, kb_id, slot, &fact);
    return .ok;
}

fn opsProvenance(kb_id: i32, slot_id: i32) kb_types.VlpProvenance {
    return .{
        .source_type = .script,
        .source_kb_id = kb_id,
        .source_slot_id = slot_id,
        .confidence = .{ .v = 62259, .r0 = 0 },
        .timestamp = timestampNow(),
        .derivation_rule_id = -1,
    };
}

fn timestampNow() i32 {
    return @intCast(@divTrunc(std.time.milliTimestamp(), 1000));
}
