// ============================================================
// src/gpu/transfer.zig
// ============================================================

const memory = @import("memory.zig");
const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const KBStore = kb_store_mod.KBStore;
const VlpFact = kb_types.VlpFact;
const VlpKB = kb_types.VlpKB;

pub fn hostToDevice(alloc: *memory.DeviceAllocation, offset: i64, data: []const u8) VlpStatus {
    const region = memory.getRegion(alloc, offset, @intCast(data.len)) orelse return .err_out_of_memory;
    @memcpy(region, data);
    return .ok;
}

pub fn deviceToHost(alloc: *const memory.DeviceAllocation, offset: i64, size: i64, output: []u8) VlpStatus {
    const region = memory.getRegionConst(alloc, offset, size) orelse return .err_out_of_memory;
    const copy_len = @min(region.len, output.len);
    @memcpy(output[0..copy_len], region[0..copy_len]);
    return .ok;
}

pub fn deviceToDevice(alloc: *memory.DeviceAllocation, dst_offset: i64, src_offset: i64, size: i64) VlpStatus {
    const s: usize = @intCast(size);
    const src_o: usize = @intCast(src_offset);
    const dst_o: usize = @intCast(dst_offset);
    if (src_o + s > alloc.host_buf.len or dst_o + s > alloc.host_buf.len) return .err_out_of_memory;

    if (dst_o < src_o) {
        for (0..s) |i| {
            alloc.host_buf[dst_o + i] = alloc.host_buf[src_o + i];
        }
    } else if (dst_o > src_o) {
        var i = s;
        while (i > 0) {
            i -= 1;
            alloc.host_buf[dst_o + i] = alloc.host_buf[src_o + i];
        }
    }

    return .ok;
}

pub fn transferQ16Array(alloc: *memory.DeviceAllocation, offset: i64, data: []const Q16) VlpStatus {
    const byte_size: i64 = @intCast(data.len * 8);
    const region = memory.getRegion(alloc, offset, byte_size) orelse return .err_out_of_memory;
    const src_bytes = std.mem.sliceAsBytes(data);
    @memcpy(region[0..src_bytes.len], src_bytes);
    return .ok;
}

pub fn transferQ16ArrayBack(alloc: *const memory.DeviceAllocation, offset: i64, output: []Q16) VlpStatus {
    const byte_size: i64 = @intCast(output.len * 8);
    const region = memory.getRegionConst(alloc, offset, byte_size) orelse return .err_out_of_memory;
    const dst_bytes = std.mem.sliceAsBytes(output);
    @memcpy(dst_bytes, region[0..dst_bytes.len]);
    return .ok;
}

pub fn mirrorKBStore(alloc: *memory.DeviceAllocation, store: *const KBStore) VlpStatus {
    const kb_region = memory.getRegion(alloc, alloc.layout.kb_store_base, alloc.layout.kb_store_size) orelse return .err_out_of_memory;
    const kb_count: usize = @intCast(store.count());
    const kb_bytes = std.mem.sliceAsBytes(store.kbs[0..kb_count]);
    const copy_len = @min(kb_bytes.len, kb_region.len);
    @memcpy(kb_region[0..copy_len], kb_bytes[0..copy_len]);

    const fact_region = memory.getRegion(alloc, alloc.layout.fact_store_base, alloc.layout.fact_store_size) orelse return .err_out_of_memory;
    const fact_count: usize = @intCast(store.factCount());
    const fact_bytes = std.mem.sliceAsBytes(store.facts[0..fact_count]);
    const fact_copy = @min(fact_bytes.len, fact_region.len);
    @memcpy(fact_region[0..fact_copy], fact_bytes[0..fact_copy]);

    const text_region = memory.getRegion(alloc, alloc.layout.text_store_base, alloc.layout.text_store_size) orelse return .err_out_of_memory;
    const text_len: usize = @intCast(store.text.len());
    const text_copy = @min(text_len, text_region.len);
    if (text_copy > 0) {
        const text_data = store.text.readAll();
        @memcpy(text_region[0..text_copy], text_data[0..text_copy]);
    }

    return .ok;
}

pub fn deviceFactWrite(alloc: *memory.DeviceAllocation, kb_id: i32, slot_id: i32, fact: *const VlpFact, facts_per_kb: i32) VlpStatus {
    const fact_offset: i64 = alloc.layout.fact_store_base + @as(i64, @intCast(kb_id)) * @as(i64, @intCast(facts_per_kb)) * 40 + @as(i64, @intCast(slot_id)) * 40;
    const region = memory.getRegion(alloc, fact_offset, 40) orelse return .err_out_of_memory;
    const fact_bytes = std.mem.asBytes(fact);
    @memcpy(region[0..fact_bytes.len], fact_bytes);
    return .ok;
}

pub fn deviceFactRead(alloc: *const memory.DeviceAllocation, kb_id: i32, slot_id: i32, fact: *VlpFact, facts_per_kb: i32) VlpStatus {
    const fact_offset: i64 = alloc.layout.fact_store_base + @as(i64, @intCast(kb_id)) * @as(i64, @intCast(facts_per_kb)) * 40 + @as(i64, @intCast(slot_id)) * 40;
    const region = memory.getRegionConst(alloc, fact_offset, 40) orelse return .err_out_of_memory;
    const fact_bytes = std.mem.asBytes(fact);
    @memcpy(fact_bytes, region[0..fact_bytes.len]);
    return .ok;
}

pub fn memorySummary(layout: *const memory.DeviceMemoryLayout) MemorySummary {
    return .{
        .model_weights_mb = @intCast(@divTrunc(layout.model_weights_size, 1048576)),
        .kb_store_mb = @intCast(@divTrunc(layout.kb_store_size, 1048576)),
        .fact_store_mb = @intCast(@divTrunc(layout.fact_store_size, 1048576)),
        .text_store_mb = @intCast(@divTrunc(layout.text_store_size, 1048576)),
        .live_state_mb = @intCast(@divTrunc(layout.live_state_size, 1048576)),
        .scratch_mb = @intCast(@divTrunc(layout.scratch_size, 1048576)),
        .audit_mb = @intCast(@divTrunc(layout.audit_size, 1048576)),
        .total_mb = @intCast(@divTrunc(layout.total_bytes, 1048576)),
    };
}

pub const MemorySummary = struct {
    model_weights_mb: i32,
    kb_store_mb: i32,
    fact_store_mb: i32,
    text_store_mb: i32,
    live_state_mb: i32,
    scratch_mb: i32,
    audit_mb: i32,
    total_mb: i32,
};
