// ============================================================
// src/gpu/kernels/attention.zig
// ============================================================

const std = @import("std");
const types = @import("../../vdr/types.zig");
const q16 = @import("../../vdr/q16.zig");
const softmax_kernel = @import("softmax.zig");
const gemm_kernel = @import("gemm.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;

pub const AttentionKernelConfig = struct {
    n_heads: i32,
    d_head: i32,
    seq_len: i32,
    causal_mask: bool,
    softmax_type: SoftmaxType,
};

pub const SoftmaxType = enum(i8) {
    quadratic = 0,
    exp_fru = 1,
};

pub fn fusedAttentionForward(
    Q_buf: []const Q16,
    K_buf: []const Q16,
    V_buf: []const Q16,
    output: []Q16,
    attn_weights: ?[]Q16,
    config: AttentionKernelConfig,
) VlpStatus {
    const n_heads: usize = @intCast(config.n_heads);
    const d_head: usize = @intCast(config.d_head);
    const seq_len: usize = @intCast(config.seq_len);
    const d_model = n_heads * d_head;

    var scores_buf: [4096]Q16 = undefined;
    var weights_buf: [4096]Q16 = undefined;
    if (seq_len * seq_len > scores_buf.len) return .err_primitive_bounds;

    for (0..n_heads) |h| {
        const head_off = h * d_head;

        for (0..seq_len) |row| {
            for (0..seq_len) |col| {
                if (config.causal_mask and col > row) {
                    scores_buf[row * seq_len + col] = Q16.zero();
                    continue;
                }
                var acc: i64 = 0;
                for (0..d_head) |d| {
                    const qv: i64 = @intCast(Q_buf[row * d_model + head_off + d].v);
                    const kv: i64 = @intCast(K_buf[col * d_model + head_off + d].v);
                    acc += qv * kv;
                }
                const scaled = @divTrunc(acc, @as(i64, Q16.D));
                scores_buf[row * seq_len + col] = .{
                    .v = @intCast(scaled),
                    .r0 = @intCast(@mod(acc, @as(i64, Q16.D))),
                };
            }

            var active_len: usize = seq_len;
            if (config.causal_mask) active_len = row + 1;

            const row_scores = scores_buf[row * seq_len .. row * seq_len + active_len];
            var row_weights = weights_buf[row * seq_len .. row * seq_len + active_len];

            _ = softmax_kernel.q16Softmax(row_scores, row_weights, @intCast(active_len));

            if (config.causal_mask) {
                for (active_len..seq_len) |c| {
                    weights_buf[row * seq_len + c] = Q16.zero();
                }
            }

            for (0..d_head) |d| {
                var acc_v: i64 = 0;
                for (0..seq_len) |c| {
                    const wv: i64 = @intCast(weights_buf[row * seq_len + c].v);
                    const vv: i64 = @intCast(V_buf[c * d_model + head_off + d].v);
                    acc_v += wv * vv;
                }
                output[row * d_model + head_off + d] = .{
                    .v = @intCast(@divTrunc(acc_v, @as(i64, Q16.D))),
                    .r0 = @intCast(@mod(acc_v, @as(i64, Q16.D))),
                };
            }
        }

        if (attn_weights) |aw| {
            const head_weight_off = h * seq_len * seq_len;
            for (0..seq_len * seq_len) |i| {
                if (head_weight_off + i < aw.len) {
                    aw[head_weight_off + i] = weights_buf[i];
                }
            }
        }
    }

    return .ok;
}

pub fn verifySoftmaxSumAllHeads(
    attn_weights: []const Q16,
    seq_len: i32,
    n_heads: i32,
) i32 {
    const sl: usize = @intCast(seq_len);
    const nh: usize = @intCast(n_heads);
    var violations: i32 = 0;

    for (0..nh) |h| {
        const base = h * sl * sl;
        for (0..sl) |row| {
            var sum: i64 = 0;
            for (0..sl) |col| {
                sum += @intCast(attn_weights[base + row * sl + col].v);
            }
            if (sum != @as(i64, Q16.D)) violations += 1;
        }
    }

    return violations;
}

pub fn fusedAttentionWithKVCache(
    Q_buf: []const Q16,
    K_cache: []const Q16,
    V_cache: []const Q16,
    output: []Q16,
    config: AttentionKernelConfig,
    cache_len: i32,
) VlpStatus {
    const n_heads: usize = @intCast(config.n_heads);
    const d_head: usize = @intCast(config.d_head);
    const d_model = n_heads * d_head;
    const cl: usize = @intCast(cache_len);

    var scores: [4096]Q16 = undefined;
    var weights: [4096]Q16 = undefined;
    if (cl > scores.len) return .err_primitive_bounds;

    for (0..n_heads) |h| {
        const head_off = h * d_head;

        for (0..cl) |col| {
            var acc: i64 = 0;
            for (0..d_head) |d| {
                const qv: i64 = @intCast(Q_buf[head_off + d].v);
                const kv: i64 = @intCast(K_cache[col * d_model + head_off + d].v);
                acc += qv * kv;
            }
            scores[col] = .{
                .v = @intCast(@divTrunc(acc, @as(i64, Q16.D))),
                .r0 = 0,
            };
        }

        _ = softmax_kernel.q16Softmax(scores[0..cl], weights[0..cl], cache_len);

        for (0..d_head) |d| {
            var acc_v: i64 = 0;
            for (0..cl) |c| {
                const wv: i64 = @intCast(weights[c].v);
                const vv: i64 = @intCast(V_cache[c * d_model + head_off + d].v);
                acc_v += wv * vv;
            }
            output[head_off + d] = .{
                .v = @intCast(@divTrunc(acc_v, @as(i64, Q16.D))),
                .r0 = @intCast(@mod(acc_v, @as(i64, Q16.D))),
            };
        }
    }

    return .ok;
}
