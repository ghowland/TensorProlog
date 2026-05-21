const std = @import("std");
const gpu = std.gpu;

const ScratchBuf = extern struct { data: [1024]i32 };
const ParamsBuf = extern struct { data: [64]i32 };

extern var scratch_a: ScratchBuf addrspace(.storage_buffer);
extern var scratch_b: ScratchBuf addrspace(.storage_buffer);
extern var params: ParamsBuf addrspace(.uniform);

export fn main() callconv(.spirv_kernel) void {
    const idx: i32 = @intCast(gpu.global_invocation_id[0]);
    const n = params.data[0];
    if (idx >= n) return;

    scratch_b.data[@intCast(idx)] = scratch_a.data[@intCast(idx)] + 1;
}
