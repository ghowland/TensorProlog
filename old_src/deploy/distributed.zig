// ============================================================
// src/deploy/distributed.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;

pub const Comm = struct {
    rank: i32,
    size: i32,
    initialized: bool,
};

pub fn commCreate(n_ranks: i32, rank: i32) Comm {
    return .{
        .rank = rank,
        .size = n_ranks,
        .initialized = true,
    };
}

pub fn commDestroy(comm: *Comm) void {
    comm.initialized = false;
}

pub fn allReduceSum(local: []Q16, result: []Q16, n: i32, comm: *const Comm) VlpStatus {
    _ = comm;
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        result[i] = local[i];
    }
    return .ok;
}

pub fn allReduceMax(local: []const Q16, result: []Q16, n: i32, comm: *const Comm) VlpStatus {
    _ = comm;
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        result[i] = local[i];
    }
    return .ok;
}

pub fn allReduceMin(local: []const Q16, result: []Q16, n: i32, comm: *const Comm) VlpStatus {
    _ = comm;
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        result[i] = local[i];
    }
    return .ok;
}

pub fn broadcast(buf: []Q16, n: i32, root: i32, comm: *const Comm) VlpStatus {
    _ = root;
    _ = comm;
    _ = n;
    _ = buf;
    return .ok;
}

pub fn allGather(send: []const Q16, recv: []Q16, send_count: i32, comm: *const Comm) VlpStatus {
    _ = comm;
    const sc: usize = @intCast(send_count);
    @memcpy(recv[0..sc], send[0..sc]);
    return .ok;
}

pub fn reduceScatter(send: []const Q16, recv: []Q16, recv_count: i32, comm: *const Comm) VlpStatus {
    _ = comm;
    const rc: usize = @intCast(recv_count);
    @memcpy(recv[0..rc], send[0..rc]);
    return .ok;
}

pub fn kbSync(local_facts: []const Q16, synced_facts: []Q16, n: i32, comm: *const Comm) VlpStatus {
    _ = comm;
    const sz: usize = @intCast(n);
    @memcpy(synced_facts[0..sz], local_facts[0..sz]);
    return .ok;
}

pub fn snapshotBroadcast(snapshot_data: []const u8, output: []u8, root: i32, comm: *const Comm) VlpStatus {
    _ = root;
    _ = comm;
    const n = @min(snapshot_data.len, output.len);
    @memcpy(output[0..n], snapshot_data[0..n]);
    return .ok;
}
