// ============================================================
// src/gpu/kb_device.zig
// ============================================================

const std = @import("std");
const q16_mod = @import("../vdr/q16.zig");
const kb_types_dev = @import("../kb/types.zig");
const kb_store_mod_dev = @import("../kb/store.zig");
const fact_mod_dev = @import("../kb/fact.zig");
const gpu_memory = @import("memory.zig");

const Q16 = q16_mod.Q16;
const VlpFact = kb_types_dev.VlpFact;
const VlpKB = kb_types_dev.VlpKB;
const KBStore = kb_store_mod_dev.KBStore;
const DeviceAllocation = gpu_memory.DeviceAllocation;
const DeviceMemoryLayout = gpu_memory.DeviceMemoryLayout;

pub const DeviceKBStore = struct {
    alloc: *DeviceAllocation,
    facts_per_kb: i32,
    kb_count: i32,

    pub fn init(alloc: *DeviceAllocation, facts_per_kb: i32) DeviceKBStore {
        return .{
            .alloc = alloc,
            .facts_per_kb = facts_per_kb,
            .kb_count = 0,
        };
    }

    pub fn factWrite(self: *DeviceKBStore, kb_id: i32, slot_id: i32, fact: *const VlpFact) VlpStatus {
        const offset = self.factOffset(kb_id, slot_id);
        const region = gpu_memory.getRegion(self.alloc, offset, 40) orelse return .err_out_of_memory;
        const fact_bytes = std.mem.asBytes(fact);
        @memcpy(region[0..fact_bytes.len], fact_bytes);
        return .ok;
    }

    pub fn factRead(self: *const DeviceKBStore, kb_id: i32, slot_id: i32, fact: *VlpFact) VlpStatus {
        const offset = self.factOffset(kb_id, slot_id);
        const region = gpu_memory.getRegionConst(self.alloc, offset, 40) orelse return .err_out_of_memory;
        const fact_bytes = std.mem.asBytes(fact);
        @memcpy(fact_bytes, region[0..fact_bytes.len]);
        return .ok;
    }

    pub fn kbWrite(self: *DeviceKBStore, kb_id: i32, kb: *const VlpKB) VlpStatus {
        const offset = self.kbOffset(kb_id);
        const region = gpu_memory.getRegion(self.alloc, offset, 256) orelse return .err_out_of_memory;
        const kb_bytes = std.mem.asBytes(kb);
        @memcpy(region[0..kb_bytes.len], kb_bytes);
        return .ok;
    }

    pub fn kbRead(self: *const DeviceKBStore, kb_id: i32, kb: *VlpKB) VlpStatus {
        const offset = self.kbOffset(kb_id);
        const region = gpu_memory.getRegionConst(self.alloc, offset, 256) orelse return .err_out_of_memory;
        const kb_bytes = std.mem.asBytes(kb);
        @memcpy(kb_bytes, region[0..kb_bytes.len]);
        return .ok;
    }

    pub fn mirrorFromHost(self: *DeviceKBStore, store: *const KBStore) VlpStatus {
        const kc: usize = @intCast(store.count());
        for (0..kc) |i| {
            const ki: i32 = @intCast(i);
            const kb = store.getKB(ki) orelse continue;
            _ = self.kbWrite(ki, kb);

            const fc: usize = @intCast(kb.facts_count);
            for (0..fc) |f| {
                const fi: i32 = @intCast(f);
                const fact = fact_mod_dev.factQuery(store, ki, fi) orelse continue;
                _ = self.factWrite(ki, fi, &fact);
            }
        }
        self.kb_count = @intCast(kc);
        return .ok;
    }

    pub fn syncToHost(self: *const DeviceKBStore, store: *KBStore) VlpStatus {
        const kc: usize = @intCast(self.kb_count);
        for (0..kc) |i| {
            const ki: i32 = @intCast(i);
            var kb: VlpKB = undefined;
            const read_status = self.kbRead(ki, &kb);
            if (read_status != .ok) continue;

            const fc: usize = @intCast(kb.facts_count);
            for (0..fc) |f| {
                const fi: i32 = @intCast(f);
                var fact: VlpFact = undefined;
                const fact_status = self.factRead(ki, fi, &fact);
                if (fact_status != .ok) continue;
                _ = fact_mod_dev.factAssert(store, ki, fi, &fact);
            }
        }
        return .ok;
    }

    fn factOffset(self: *const DeviceKBStore, kb_id: i32, slot_id: i32) i64 {
        return self.alloc.layout.fact_store_base +
            @as(i64, @intCast(kb_id)) * @as(i64, @intCast(self.facts_per_kb)) * 40 +
            @as(i64, @intCast(slot_id)) * 40;
    }

    fn kbOffset(self: *const DeviceKBStore, kb_id: i32) i64 {
        _ = self;
        return self.alloc.layout.kb_store_base + @as(i64, @intCast(kb_id)) * 256;
    }
};

const VlpStatus = types_op.VlpStatus;
