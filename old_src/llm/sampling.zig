// ============================================================
// src/llm/sampling.zig
// ============================================================

const q16 = @import("../vdr/q16.zig");
const Q16 = q16.Q16;

pub const SamplingConfig = struct {
    temperature: Q16 = .{ .v = Q16.D, .r0 = 0 },
    top_k: i32 = 50,
    top_p: Q16 = .{ .v = 58982, .r0 = 0 },
    greedy: bool = false,
};

pub const RNG = struct {
    state: i64,

    pub fn init(seed: i64) RNG {
        var s = seed;
        if (s == 0) s = 1;
        return .{ .state = s };
    }

    pub fn next(self: *RNG) i32 {
        self.state = self.state *% 6364136223846793005 +% 1442695040888963407;
        const shifted: i64 = @shrExact(self.state & 0x7FFFFFFF00000000, 32);
        return @intCast(shifted & 0x7FFFFFFF);
    }

    pub fn nextBounded(self: *RNG, bound: i32) i32 {
        if (bound <= 0) return 0;
        const raw = self.next();
        const b: i64 = @intCast(bound);
        const r: i64 = @intCast(raw);
        return @intCast(@mod(r, b));
    }
};

pub fn sampleGreedy(probs: []const Q16) i32 {
    var max_val: i32 = probs[0].v;
    var max_idx: i32 = 0;

    for (probs[1..], 1..) |p, i| {
        if (p.v > max_val) {
            max_val = p.v;
            max_idx = @intCast(i);
        }
    }

    return max_idx;
}

pub fn sampleTopK(
    probs: []const Q16,
    k: i32,
    rng: *RNG,
) i32 {
    const n = probs.len;
    const ku: usize = @intCast(@min(k, @as(i32, @intCast(n))));

    var indices: [65536]i32 = undefined;
    for (0..n) |i| {
        indices[i] = @intCast(i);
    }

    for (0..ku) |i| {
        var best = i;
        for (i + 1..n) |j| {
            const bi: usize = @intCast(indices[best]);
            const ji: usize = @intCast(indices[j]);
            if (probs[ji].v > probs[bi].v) {
                best = j;
            }
        }
        const tmp = indices[i];
        indices[i] = indices[best];
        indices[best] = tmp;
    }

    var top_sum: i64 = 0;
    for (0..ku) |i| {
        const idx: usize = @intCast(indices[i]);
        top_sum += @intCast(probs[idx].v);
    }

    if (top_sum == 0) {
        return indices[0];
    }

    var threshold = rng.next();
    threshold = @intCast(@mod(@as(i64, @intCast(threshold)), top_sum));

    var cumulative: i64 = 0;
    for (0..ku) |i| {
        const idx: usize = @intCast(indices[i]);
        cumulative += @intCast(probs[idx].v);
        if (cumulative > @as(i64, @intCast(threshold))) {
            return indices[i];
        }
    }

    return indices[ku - 1];
}

pub fn sampleTopP(
    probs: []const Q16,
    p_threshold: Q16,
    rng: *RNG,
) i32 {
    const n = probs.len;

    var indices: [65536]i32 = undefined;
    for (0..n) |i| {
        indices[i] = @intCast(i);
    }

    for (0..n) |i| {
        for (i + 1..n) |j| {
            const ii: usize = @intCast(indices[i]);
            const ji: usize = @intCast(indices[j]);
            if (probs[ji].v > probs[ii].v) {
                const tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }

    var total: i64 = 0;
    for (probs) |p| {
        total += @intCast(p.v);
    }

    if (total == 0) {
        return 0;
    }

    const threshold_val: i64 = @divTrunc(@as(i64, @intCast(p_threshold.v)) * total, @as(i64, Q16.D));

    var cumulative: i64 = 0;
    var prefix_len: usize = 0;
    for (0..n) |i| {
        const idx: usize = @intCast(indices[i]);
        cumulative += @intCast(probs[idx].v);
        prefix_len = i + 1;
        if (cumulative >= threshold_val) {
            break;
        }
    }

    if (prefix_len == 0) {
        prefix_len = 1;
    }

    var prefix_sum: i64 = 0;
    for (0..prefix_len) |i| {
        const idx: usize = @intCast(indices[i]);
        prefix_sum += @intCast(probs[idx].v);
    }

    if (prefix_sum == 0) {
        return indices[0];
    }

    var sample_point = rng.next();
    sample_point = @intCast(@mod(@as(i64, @intCast(sample_point)), prefix_sum));

    var running: i64 = 0;
    for (0..prefix_len) |i| {
        const idx: usize = @intCast(indices[i]);
        running += @intCast(probs[idx].v);
        if (running > @as(i64, @intCast(sample_point))) {
            return indices[i];
        }
    }

    return indices[prefix_len - 1];
}

pub fn applyTemperature(
    logits: []const Q16,
    temperature: Q16,
    output: []Q16,
) void {
    if (temperature.v == 0) {
        for (logits, 0..) |l, i| {
            output[i] = l;
        }
        return;
    }

    for (logits, 0..) |l, i| {
        const lv: i64 = @intCast(l.v);
        const tv: i64 = @intCast(temperature.v);
        const divided: i64 = @divTrunc(lv * @as(i64, Q16.D), tv);
        output[i] = .{
            .v = @intCast(divided),
            .r0 = @intCast(lv * @as(i64, Q16.D) - divided * tv),
        };
    }
}
