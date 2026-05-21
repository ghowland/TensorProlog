// ============================================================
// src/deploy/model_parallel.zig
// ============================================================

const gpu_memory = @import("../gpu/memory.zig");
const gpu_transfer = @import("../gpu/transfer.zig");
const gemm_kernel = @import("../gpu/kernels/gemm.zig");

pub const ModelParallelConfig = struct {
    n_devices: i32,
    n_layers: i32,
    d_model: i32,
    n_heads: i32,
    d_head: i32,
    vocab_size: i32,
};

pub const DeviceShard = struct {
    device_id: i32,
    layer_start: i32,
    layer_end: i32,
    n_layers: i32,
};

pub const ModelParallel = struct {
    config: ModelParallelConfig,
    shards: [8]DeviceShard,
    n_shards: i32,
    hidden_buf: [4096]Q16,
    inter_buf: [4096]Q16,

    pub fn init(config: ModelParallelConfig) ModelParallel {
        var mp = ModelParallel{
            .config = config,
            .shards = undefined,
            .n_shards = config.n_devices,
            .hidden_buf = undefined,
            .inter_buf = undefined,
        };

        const layers_per_device = @divTrunc(config.n_layers, config.n_devices);
        var layer_pos: i32 = 0;

        for (0..@as(usize, @intCast(config.n_devices))) |i| {
            const remaining = config.n_layers - layer_pos;
            const this_shard = if (i == @as(usize, @intCast(config.n_devices)) - 1) remaining else layers_per_device;
            mp.shards[i] = .{
                .device_id = @intCast(i),
                .layer_start = layer_pos,
                .layer_end = layer_pos + this_shard,
                .n_layers = this_shard,
            };
            layer_pos += this_shard;
        }

        return mp;
    }

    pub fn pipelineForward(self: *ModelParallel, input: []const Q16, output: []Q16) VlpStatus {
        const dm: usize = @intCast(self.config.d_model);
        const use_dm = @min(dm, self.hidden_buf.len);

        @memcpy(self.hidden_buf[0..use_dm], input[0..use_dm]);

        const ns: usize = @intCast(self.n_shards);
        for (0..ns) |s| {
            const nl: usize = @intCast(self.shards[s].n_layers);
            for (0..nl) |_| {
                @memcpy(self.inter_buf[0..use_dm], self.hidden_buf[0..use_dm]);
                @memcpy(self.hidden_buf[0..use_dm], self.inter_buf[0..use_dm]);
            }
        }

        @memcpy(output[0..use_dm], self.hidden_buf[0..use_dm]);
        return .ok;
    }

    pub fn getShardInfo(self: *const ModelParallel, device_id: i32) ?DeviceShard {
        const ns: usize = @intCast(self.n_shards);
        for (0..ns) |i| {
            if (self.shards[i].device_id == device_id) return self.shards[i];
        }
        return null;
    }
};
