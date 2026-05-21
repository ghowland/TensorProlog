// ============================================================
// src/deploy/multi_device.zig
// ============================================================

const gpu_device = @import("../gpu/device.zig");
const gpu_memory_md = @import("../gpu/memory.zig");
const gpu_transfer_md = @import("../gpu/transfer.zig");

pub const MultiDeviceConfig = struct {
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
    alloc: ?*gpu_memory_md.DeviceAllocation,
};

pub const MultiDevice = struct {
    config: MultiDeviceConfig,
    shards: [8]DeviceShard,
    n_shards: i32,
    hidden_transfer_buf: [8192]q16_mod.Q16,

    pub fn init(config: MultiDeviceConfig) MultiDevice {
        var md = MultiDevice{
            .config = config,
            .shards = undefined,
            .n_shards = @min(config.n_devices, 8),
            .hidden_transfer_buf = undefined,
        };

        const layers_per_device = @divTrunc(config.n_layers, md.n_shards);
        var layer_pos: i32 = 0;
        const ns: usize = @intCast(md.n_shards);

        for (0..ns) |i| {
            const is_last = (i == ns - 1);
            const this_layers = if (is_last) config.n_layers - layer_pos else layers_per_device;
            md.shards[i] = .{
                .device_id = @intCast(i),
                .layer_start = layer_pos,
                .layer_end = layer_pos + this_layers,
                .n_layers = this_layers,
                .alloc = null,
            };
            layer_pos += this_layers;
        }

        return md;
    }

    pub fn getShardForLayer(self: *const MultiDevice, layer: i32) ?usize {
        const ns: usize = @intCast(self.n_shards);
        for (0..ns) |i| {
            if (layer >= self.shards[i].layer_start and layer < self.shards[i].layer_end) return i;
        }
        return null;
    }

    pub fn pipelineForward(self: *MultiDevice, input: []const q16_mod.Q16, output: []q16_mod.Q16) VlpStatus {
        const dm: usize = @intCast(self.config.d_model);
        const use_dm = @min(dm, self.hidden_transfer_buf.len);

        @memcpy(self.hidden_transfer_buf[0..use_dm], input[0..use_dm]);

        const ns: usize = @intCast(self.n_shards);
        for (0..ns) |s| {
            const nl: usize = @intCast(self.shards[s].n_layers);
            for (0..nl) |_| {
                // stub: each layer would run forward pass on shard's device
            }
        }

        @memcpy(output[0..use_dm], self.hidden_transfer_buf[0..use_dm]);
        return .ok;
    }

    pub fn totalLayers(self: *const MultiDevice) i32 {
        var total: i32 = 0;
        const ns: usize = @intCast(self.n_shards);
        for (0..ns) |i| {
            total += self.shards[i].n_layers;
        }
        return total;
    }

    pub fn shardSummary(self: *const MultiDevice, output: []u8) i32 {
        var pos: usize = 0;
        const ns: usize = @intCast(self.n_shards);
        for (0..ns) |i| {
            pos += copyStrMD(output[pos..], "device ");
            pos += i32ToAsciiMD(self.shards[i].device_id, output[pos..]);
            pos += copyStrMD(output[pos..], ": layers ");
            pos += i32ToAsciiMD(self.shards[i].layer_start, output[pos..]);
            pos += copyStrMD(output[pos..], "-");
            pos += i32ToAsciiMD(self.shards[i].layer_end, output[pos..]);
            pos += copyStrMD(output[pos..], " (");
            pos += i32ToAsciiMD(self.shards[i].n_layers, output[pos..]);
            pos += copyStrMD(output[pos..], ")\n");
        }
        return @intCast(pos);
    }
};

fn copyStrMD(dest: []u8, src: []const u8) usize {
    const n = @min(src.len, dest.len);
    @memcpy(dest[0..n], src[0..n]);
    return n;
}

fn i32ToAsciiMD(val: i32, output: []u8) usize {
    if (output.len == 0) return 0;
    if (val == 0) {
        output[0] = '0';
        return 1;
    }
    var v: i64 = @intCast(val);
    var pos: usize = 0;
    if (v < 0) {
        output[pos] = '-';
        pos += 1;
        v = -v;
    }
    var buf_md: [12]u8 = undefined;
    var len: usize = 0;
    while (v > 0) {
        buf_md[len] = @intCast(@as(u8, @intCast(@mod(v, 10))) + '0');
        len += 1;
        v = @divTrunc(v, 10);
    }
    for (0..len) |i| {
        if (pos >= output.len) break;
        output[pos] = buf_md[len - 1 - i];
        pos += 1;
    }
    return pos;
}
