// ============================================================
// src/llm/attention.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;

pub const AttentionConfig = struct {
    n_heads: i32,
    d_head: i32,
    seq_len: i32,
    causal_mask: bool,
};

pub fn attention(
    Q_buf: []const Q16,
    K_buf: []const Q16,
    V_buf: []const Q16,
    config: AttentionConfig,
    output: []Q16,
    scores_scratch: []Q16,
    weights_scratch: []Q16,
) void {
    const n_heads: usize = @intCast(config.n_heads);
    const d_head: usize = @intCast(config.d_head);
    const seq_len: usize = @intCast(config.seq_len);
    const d_model = n_heads * d_head;

    for (0..n_heads) |h| {
        const q_head_offset = h * d_head;
        const k_head_offset = h * d_head;
        const v_head_offset = h * d_head;
        const out_head_offset = h * d_head;

        for (0..seq_len) |row| {
            const q_row_start = row * d_model + q_head_offset;

            for (0..seq_len) |col| {
                if (config.causal_mask and col > row) {
                    scores_scratch[row * seq_len + col] = Q16.zero();
                    continue;
                }

                const k_col_start = col * d_model + k_head_offset;
                var acc: i64 = 0;
                for (0..d_head) |d| {
                    const qv: i64 = @intCast(Q_buf[q_row_start + d].v);
                    const kv: i64 = @intCast(K_buf[k_col_start + d].v);
                    acc += qv * kv;
                }
                const shifted: i64 = @divTrunc(acc, @as(i64, Q16.D));
                scores_scratch[row * seq_len + col] = .{
                    .v = @intCast(shifted),
                    .r0 = @intCast(@as(i64, @intCast(@mod(acc, @as(i64, Q16.D))))),
                };
            }

            var row_scores = scores_scratch[row * seq_len .. row * seq_len + seq_len];
            var row_weights = weights_scratch[row * seq_len .. row * seq_len + seq_len];

            var active_len: usize = seq_len;
            if (config.causal_mask) {
                active_len = row + 1;
            }

            Q16.softmax(row_scores[0..active_len], row_weights[0..active_len]);

            if (config.causal_mask) {
                for (active_len..seq_len) |c| {
                    row_weights[c] = Q16.zero();
                }
            }

            for (0..d_head) |d| {
                var acc_v: i64 = 0;
                for (0..seq_len) |c| {
                    const wv: i64 = @intCast(row_weights[c].v);
                    const vv: i64 = @intCast(V_buf[c * d_model + v_head_offset + d].v);
                    acc_v += wv * vv;
                }
                const out_shifted: i64 = @divTrunc(acc_v, @as(i64, Q16.D));
                output[row * d_model + out_head_offset + d] = .{
                    .v = @intCast(out_shifted),
                    .r0 = @intCast(@as(i64, @intCast(@mod(acc_v, @as(i64, Q16.D))))),
                };
            }
        }
    }
}

pub fn verifySoftmaxSum(
    weights: []const Q16,
    seq_len: i32,
    n_heads: i32,
) i32 {
    const s: usize = @intCast(seq_len);
    const h: usize = @intCast(n_heads);
    var violations: i32 = 0;

    for (0..h) |_| {
        for (0..s) |row| {
            var sum: i64 = 0;
            for (0..s) |col| {
                sum += @intCast(weights[row * s + col].v);
            }
            if (sum != @as(i64, Q16.D)) {
                violations += 1;
            }
        }
    }
    return violations;
}
