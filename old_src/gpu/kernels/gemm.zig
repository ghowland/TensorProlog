// ============================================================
// src/gpu/kernels/gemm.zig
// ============================================================

const std = @import("std");
const types = @import("../../vdr/types.zig");
const q16 = @import("../../vdr/q16.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;

pub const GemmConfig = struct {
    m: i32,
    n: i32,
    k: i32,
    trans_a: bool,
    trans_b: bool,
    alpha_v: i32,
    beta_v: i32,
};

pub fn q16Gemm(
    A: []const Q16,
    B: []const Q16,
    C: []Q16,
    config: GemmConfig,
) VlpStatus {
    const m: usize = @intCast(config.m);
    const n: usize = @intCast(config.n);
    const k: usize = @intCast(config.k);
    const alpha: i64 = @intCast(config.alpha_v);
    const beta: i64 = @intCast(config.beta_v);

    for (0..m) |row| {
        for (0..n) |col| {
            var acc: i64 = 0;
            for (0..k) |ki| {
                const a_idx = if (config.trans_a) ki * m + row else row * k + ki;
                const b_idx = if (config.trans_b) col * k + ki else ki * n + col;
                const av: i64 = @intCast(A[a_idx].v);
                const bv: i64 = @intCast(B[b_idx].v);
                acc += av * bv;
            }

            const scaled_acc = @divTrunc(acc * alpha, @as(i64, Q16.D) * @as(i64, Q16.D));

            const c_idx = row * n + col;
            const old_c: i64 = @intCast(C[c_idx].v);
            const beta_c = @divTrunc(old_c * beta, @as(i64, Q16.D));

            const result = scaled_acc + beta_c;
            C[c_idx] = .{
                .v = @intCast(result),
                .r0 = @intCast(@mod(acc, @as(i64, Q16.D))),
            };
        }
    }

    return .ok;
}

pub fn q16GemmBatched(
    A_batch: []const []const Q16,
    B_batch: []const []const Q16,
    C_batch: [][]Q16,
    config: GemmConfig,
    batch_count: i32,
) VlpStatus {
    const bc: usize = @intCast(batch_count);
    for (0..bc) |b| {
        const status = q16Gemm(A_batch[b], B_batch[b], C_batch[b], config);
        if (status != .ok) return status;
    }
    return .ok;
}

pub fn q16GemmStridedBatched(
    A: []const Q16,
    B: []const Q16,
    C: []Q16,
    config: GemmConfig,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
    batch_count: i32,
) VlpStatus {
    const bc: usize = @intCast(batch_count);
    const sa: usize = @intCast(stride_a);
    const sb: usize = @intCast(stride_b);
    const sc: usize = @intCast(stride_c);
    const m: usize = @intCast(config.m);
    const n: usize = @intCast(config.n);
    const k: usize = @intCast(config.k);

    for (0..bc) |b| {
        const a_off = b * sa;
        const b_off = b * sb;
        const c_off = b * sc;

        for (0..m) |row| {
            for (0..n) |col| {
                var acc: i64 = 0;
                for (0..k) |ki| {
                    const a_idx = a_off + if (config.trans_a) ki * m + row else row * k + ki;
                    const b_idx = b_off + if (config.trans_b) col * k + ki else ki * n + col;
                    const av: i64 = @intCast(A[a_idx].v);
                    const bv: i64 = @intCast(B[b_idx].v);
                    acc += av * bv;
                }

                const c_idx = c_off + row * n + col;
                C[c_idx] = .{
                    .v = @intCast(@divTrunc(acc, @as(i64, Q16.D))),
                    .r0 = @intCast(@mod(acc, @as(i64, Q16.D))),
                };
            }
        }
    }

    return .ok;
}

pub fn q16MatVecMul(A: []const Q16, x: []const Q16, y: []Q16, m: i32, n: i32) VlpStatus {
    const rows: usize = @intCast(m);
    const cols: usize = @intCast(n);

    for (0..rows) |row| {
        var acc: i64 = 0;
        for (0..cols) |col| {
            const av: i64 = @intCast(A[row * cols + col].v);
            const xv: i64 = @intCast(x[col].v);
            acc += av * xv;
        }
        y[row] = .{
            .v = @intCast(@divTrunc(acc, @as(i64, Q16.D))),
            .r0 = @intCast(@mod(acc, @as(i64, Q16.D))),
        };
    }

    return .ok;
}
