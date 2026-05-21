// ============================================================
// src/ops/execute.zig
// ============================================================

pub fn execRun(command: []const u8, args: []const []const u8, output: []u8) struct { len: i32, exit_code: i32, status: VlpStatus } {
    _ = command;
    _ = args;
    const placeholder = "exec stub";
    const pl = @min(placeholder.len, output.len);
    @memcpy(output[0..pl], placeholder[0..pl]);
    return .{ .len = @intCast(pl), .exit_code = 0, .status = .ok };
}

pub fn execRunToKB(store: *KBStore, kb_id: i32, slot: i32, command: []const u8, args: []const []const u8) VlpStatus {
    var buf: [65536]u8 = undefined;
    const result = execRun(command, args, &buf);
    if (result.status != .ok) return result.status;
    const rl: usize = @intCast(result.len);
    const ref = store.text.append(buf[0..rl]);
    const fact = VlpFact{
        .tag = .text,
        .value = .{ .v = ref.offset, .r0 = @intCast(ref.length) },
        .provenance = .{
            .source_type = .script,
            .source_kb_id = kb_id,
            .source_slot_id = slot,
            .confidence = .{ .v = 62259, .r0 = 0 },
            .timestamp = timestampNow(),
            .derivation_rule_id = -1,
        },
    };
    _ = fact_mod.factAssert(store, kb_id, slot, &fact);
    return .ok;
}
