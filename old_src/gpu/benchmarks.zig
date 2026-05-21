// ============================================================
// src/gpu/benchmarks.zig
// ============================================================

const gemm_kernel = @import("kernels/gemm.zig");
const softmax_kernel = @import("kernels/softmax.zig");
const elementwise_kernel = @import("kernels/elementwise.zig");
const attention_kernel = @import("kernels/attention.zig");
const sort_kernel = @import("kernels/sort.zig");
const normalize_kernel = @import("kernels/normalize.zig");
const prolog_kernel = @import("kernels/prolog_kernel.zig");
const prolog_types = @import("../prolog/types.zig");

pub const BenchResult = struct {
    name: [32]u8,
    name_len: i32,
    total_ns: i64,
    per_iter_ns: i64,
    n_iters: i32,
    ops_per_second: i64,
};

fn nameBench(name: []const u8) [32]u8 {
    var buf: [32]u8 = undefined;
    const n = @min(name.len, buf.len);
    @memcpy(buf[0..n], name[0..n]);
    if (n < buf.len) @memset(buf[n..], 0);
    return buf;
}

pub fn benchForwardPass(seq_len: i32, d_model: i32, n_iters: i32) BenchResult {
    const sl: usize = @intCast(seq_len);
    const dm: usize = @intCast(d_model);
    const ni: usize = @intCast(n_iters);
    const total = sl * dm;

    var input: [4096]Q16 = undefined;
    var output: [4096]Q16 = undefined;
    var weights: [4096]Q16 = undefined;

    const use_total = @min(total, 4096);
    const use_dm = @min(dm, 64);

    for (0..use_total) |i| {
        input[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 7 + 3, 1000)), .r0 = 0 };
    }
    for (0..use_dm * use_dm) |i| {
        weights[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 13 + 5, 500)), .r0 = 0 };
    }

    const start = std.time.nanoTimestamp();

    for (0..ni) |_| {
        _ = gemm_kernel.q16MatVecMul(&weights, &input, &output, @intCast(use_dm), @intCast(use_dm));
    }

    const elapsed = std.time.nanoTimestamp() - start;
    const per_iter = @divTrunc(elapsed, @as(i64, @intCast(ni)));

    return .{
        .name = nameBench("forward_pass"),
        .name_len = 12,
        .total_ns = elapsed,
        .per_iter_ns = per_iter,
        .n_iters = n_iters,
        .ops_per_second = if (per_iter > 0) @divTrunc(1000000000, per_iter) else 0,
    };
}

pub fn benchSoftmax(n: i32, n_iters: i32) BenchResult {
    const sz: usize = @intCast(n);
    const ni: usize = @intCast(n_iters);
    const use_sz = @min(sz, 4096);

    var input: [4096]Q16 = undefined;
    var output: [4096]Q16 = undefined;

    for (0..use_sz) |i| {
        input[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 11 + 7, 2000)), .r0 = 0 };
    }

    const start = std.time.nanoTimestamp();

    for (0..ni) |_| {
        _ = softmax_kernel.q16Softmax(input[0..use_sz], output[0..use_sz], @intCast(use_sz));
    }

    const elapsed = std.time.nanoTimestamp() - start;
    const per_iter = @divTrunc(elapsed, @as(i64, @intCast(ni)));

    return .{
        .name = nameBench("softmax"),
        .name_len = 7,
        .total_ns = elapsed,
        .per_iter_ns = per_iter,
        .n_iters = n_iters,
        .ops_per_second = if (per_iter > 0) @divTrunc(1000000000, per_iter) else 0,
    };
}

pub fn benchAttention(seq_len: i32, n_heads: i32, d_head: i32, n_iters: i32) BenchResult {
    const sl: usize = @intCast(seq_len);
    const nh: usize = @intCast(n_heads);
    const dh: usize = @intCast(d_head);
    const ni: usize = @intCast(n_iters);
    const dm = nh * dh;
    const total = sl * dm;
    const use_total = @min(total, 4096);

    var Q_buf: [4096]Q16 = undefined;
    var K_buf: [4096]Q16 = undefined;
    var V_buf: [4096]Q16 = undefined;
    var output: [4096]Q16 = undefined;

    for (0..use_total) |i| {
        Q_buf[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 3 + 1, 500)), .r0 = 0 };
        K_buf[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 5 + 2, 500)), .r0 = 0 };
        V_buf[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 7 + 3, 500)), .r0 = 0 };
    }

    const start = std.time.nanoTimestamp();

    for (0..ni) |_| {
        _ = attention_kernel.fusedAttentionForward(
            Q_buf[0..use_total],
            K_buf[0..use_total],
            V_buf[0..use_total],
            output[0..use_total],
            null,
            .{
                .n_heads = n_heads,
                .d_head = d_head,
                .seq_len = seq_len,
                .causal_mask = true,
                .softmax_type = .quadratic,
            },
        );
    }

    const elapsed = std.time.nanoTimestamp() - start;
    const per_iter = @divTrunc(elapsed, @as(i64, @intCast(ni)));

    return .{
        .name = nameBench("attention"),
        .name_len = 9,
        .total_ns = elapsed,
        .per_iter_ns = per_iter,
        .n_iters = n_iters,
        .ops_per_second = if (per_iter > 0) @divTrunc(1000000000, per_iter) else 0,
    };
}

pub fn benchSort(n: i32, n_iters: i32) BenchResult {
    const sz: usize = @intCast(n);
    const ni: usize = @intCast(n_iters);
    const use_sz = @min(sz, 4096);

    var data: [4096]Q16 = undefined;
    var work: [4096]Q16 = undefined;

    for (0..use_sz) |i| {
        data[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 997 + 41, 65536)), .r0 = 0 };
    }

    const start = std.time.nanoTimestamp();

    for (0..ni) |_| {
        @memcpy(work[0..use_sz], data[0..use_sz]);
        _ = sort_kernel.q16Sort(work[0..use_sz], @intCast(use_sz));
    }

    const elapsed = std.time.nanoTimestamp() - start;
    const per_iter = @divTrunc(elapsed, @as(i64, @intCast(ni)));

    return .{
        .name = nameBench("sort"),
        .name_len = 4,
        .total_ns = elapsed,
        .per_iter_ns = per_iter,
        .n_iters = n_iters,
        .ops_per_second = if (per_iter > 0) @divTrunc(1000000000, per_iter) else 0,
    };
}

pub fn benchLayerNorm(n: i32, n_iters: i32) BenchResult {
    const sz: usize = @intCast(n);
    const ni: usize = @intCast(n_iters);
    const use_sz = @min(sz, 4096);

    var input: [4096]Q16 = undefined;
    var output: [4096]Q16 = undefined;
    var gamma: [4096]Q16 = undefined;
    var beta: [4096]Q16 = undefined;

    for (0..use_sz) |i| {
        input[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 17 + 11, 10000)), .r0 = 0 };
        gamma[i] = Q16.one();
        beta[i] = Q16.zero();
    }

    const start = std.time.nanoTimestamp();

    for (0..ni) |_| {
        _ = normalize_kernel.q16LayerNorm(input[0..use_sz], output[0..use_sz], gamma[0..use_sz], beta[0..use_sz], @intCast(use_sz));
    }

    const elapsed = std.time.nanoTimestamp() - start;
    const per_iter = @divTrunc(elapsed, @as(i64, @intCast(ni)));

    return .{
        .name = nameBench("layernorm"),
        .name_len = 9,
        .total_ns = elapsed,
        .per_iter_ns = per_iter,
        .n_iters = n_iters,
        .ops_per_second = if (per_iter > 0) @divTrunc(1000000000, per_iter) else 0,
    };
}

pub fn benchPrologUnify(n_candidates: i32, n_iters: i32) BenchResult {
    const nc: usize = @intCast(n_candidates);
    const ni: usize = @intCast(n_iters);
    const use_nc = @min(nc, 256);

    var query = prolog_types.VlpTerm{
        .term_type = .atom,
        .data = .{ .atom_id = 42 },
    };

    var candidates: [256]prolog_types.VlpTerm = undefined;
    for (0..use_nc) |i| {
        candidates[i] = .{
            .term_type = .atom,
            .data = .{ .atom_id = @intCast(40 + @mod(i, 5)) },
        };
    }

    const start = std.time.nanoTimestamp();

    var result: prolog_kernel.UnifyBatchResult = undefined;
    for (0..ni) |_| {
        _ = prolog_kernel.batchUnify(&query, candidates[0..use_nc], @intCast(use_nc), &result);
    }

    const elapsed = std.time.nanoTimestamp() - start;
    const per_iter = @divTrunc(elapsed, @as(i64, @intCast(ni)));

    return .{
        .name = nameBench("prolog_unify"),
        .name_len = 12,
        .total_ns = elapsed,
        .per_iter_ns = per_iter,
        .n_iters = n_iters,
        .ops_per_second = if (per_iter > 0) @divTrunc(1000000000, per_iter) else 0,
    };
}

pub fn runAllBenchmarks() [8]BenchResult {
    var results: [8]BenchResult = undefined;
    results[0] = benchForwardPass(4, 16, 1000);
    results[1] = benchSoftmax(64, 1000);
    results[2] = benchAttention(4, 1, 8, 1000);
    results[3] = benchSort(256, 1000);
    results[4] = benchLayerNorm(64, 1000);
    results[5] = benchPrologUnify(64, 1000);
    results[6] = benchElementwise(256, 1000);
    results[7] = benchGemm(8, 8, 8, 1000);
    return results;
}

fn benchElementwise(n: i32, n_iters: i32) BenchResult {
    const sz: usize = @intCast(n);
    const ni: usize = @intCast(n_iters);
    const use_sz = @min(sz, 4096);

    var a: [4096]Q16 = undefined;
    var b: [4096]Q16 = undefined;
    var c: [4096]Q16 = undefined;

    for (0..use_sz) |i| {
        a[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 3, 1000)), .r0 = 0 };
        b[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 7, 1000) + 1), .r0 = 0 };
    }

    const start = std.time.nanoTimestamp();

    for (0..ni) |_| {
        _ = elementwise_kernel.q16Add(a[0..use_sz], b[0..use_sz], c[0..use_sz], @intCast(use_sz));
    }

    const elapsed = std.time.nanoTimestamp() - start;
    const per_iter = @divTrunc(elapsed, @as(i64, @intCast(ni)));

    return .{
        .name = nameBench("elementwise"),
        .name_len = 11,
        .total_ns = elapsed,
        .per_iter_ns = per_iter,
        .n_iters = n_iters,
        .ops_per_second = if (per_iter > 0) @divTrunc(1000000000, per_iter) else 0,
    };
}

fn benchGemm(m: i32, n: i32, k: i32, n_iters: i32) BenchResult {
    const mi: usize = @intCast(m);
    const ni_dim: usize = @intCast(n);
    const ki: usize = @intCast(k);
    const iters: usize = @intCast(n_iters);

    var A: [4096]Q16 = undefined;
    var B: [4096]Q16 = undefined;
    var C: [4096]Q16 = undefined;

    for (0..mi * ki) |i| {
        A[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 11, 500)), .r0 = 0 };
    }
    for (0..ki * ni_dim) |i| {
        B[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 13, 500)), .r0 = 0 };
    }
    for (0..mi * ni_dim) |i| {
        C[i] = Q16.zero();
    }

    const start = std.time.nanoTimestamp();

    for (0..iters) |_| {
        _ = gemm_kernel.q16Gemm(&A, &B, &C, .{
            .m = m,
            .n = n,
            .k = k,
            .trans_a = false,
            .trans_b = false,
            .alpha_v = Q16.D,
            .beta_v = 0,
        });
    }

    const elapsed = std.time.nanoTimestamp() - start;
    const per_iter = @divTrunc(elapsed, @as(i64, @intCast(n_iters)));

    return .{
        .name = nameBench("gemm"),
        .name_len = 4,
        .total_ns = elapsed,
        .per_iter_ns = per_iter,
        .n_iters = n_iters,
        .ops_per_second = if (per_iter > 0) @divTrunc(1000000000, per_iter) else 0,
    };
}

// ============================================================
// src/gpu/determinism.zig
// ============================================================

pub const DeterminismVerifyResult = struct {
    kernel_name: [32]u8,
    kernel_name_len: i32,
    n_runs: i32,
    all_identical: bool,
    first_mismatch_run: i32,
    first_mismatch_byte: i32,
};

pub fn verifyDeterminismSoftmax(n: i32, n_runs: i32) DeterminismVerifyResult {
    const sz: usize = @intCast(n);
    const nr: usize = @intCast(n_runs);
    const use_sz = @min(sz, 4096);

    var input: [4096]Q16 = undefined;
    for (0..use_sz) |i| {
        input[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 37 + 13, 5000)), .r0 = 0 };
    }

    var reference: [4096]Q16 = undefined;
    _ = softmax_kernel.q16Softmax(input[0..use_sz], reference[0..use_sz], @intCast(use_sz));

    var run_output: [4096]Q16 = undefined;
    var result = DeterminismVerifyResult{
        .kernel_name = nameBench("softmax"),
        .kernel_name_len = 7,
        .n_runs = n_runs,
        .all_identical = true,
        .first_mismatch_run = -1,
        .first_mismatch_byte = -1,
    };

    for (0..nr) |run| {
        _ = softmax_kernel.q16Softmax(input[0..use_sz], run_output[0..use_sz], @intCast(use_sz));
        const ref_bytes = std.mem.sliceAsBytes(reference[0..use_sz]);
        const run_bytes = std.mem.sliceAsBytes(run_output[0..use_sz]);
        for (0..ref_bytes.len) |bi| {
            if (ref_bytes[bi] != run_bytes[bi]) {
                result.all_identical = false;
                result.first_mismatch_run = @intCast(run);
                result.first_mismatch_byte = @intCast(bi);
                return result;
            }
        }
    }

    return result;
}

pub fn verifyDeterminismGemm(m: i32, n: i32, k: i32, n_runs: i32) DeterminismVerifyResult {
    const mi: usize = @intCast(m);
    const ni: usize = @intCast(n);
    const ki: usize = @intCast(k);
    const nr: usize = @intCast(n_runs);

    var A: [4096]Q16 = undefined;
    var B: [4096]Q16 = undefined;
    var ref_C: [4096]Q16 = undefined;
    var run_C: [4096]Q16 = undefined;

    for (0..mi * ki) |i| {
        A[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 7, 1000)), .r0 = 0 };
    }
    for (0..ki * ni) |i| {
        B[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 11, 1000)), .r0 = 0 };
    }
    for (0..mi * ni) |i| {
        ref_C[i] = Q16.zero();
    }

    _ = gemm_kernel.q16Gemm(&A, &B, &ref_C, .{ .m = m, .n = n, .k = k, .trans_a = false, .trans_b = false, .alpha_v = Q16.D, .beta_v = 0 });

    var result = DeterminismVerifyResult{
        .kernel_name = nameBench("gemm"),
        .kernel_name_len = 4,
        .n_runs = n_runs,
        .all_identical = true,
        .first_mismatch_run = -1,
        .first_mismatch_byte = -1,
    };

    for (0..nr) |run| {
        for (0..mi * ni) |i| {
            run_C[i] = Q16.zero();
        }
        _ = gemm_kernel.q16Gemm(&A, &B, &run_C, .{ .m = m, .n = n, .k = k, .trans_a = false, .trans_b = false, .alpha_v = Q16.D, .beta_v = 0 });
        const ref_bytes = std.mem.sliceAsBytes(ref_C[0 .. mi * ni]);
        const run_bytes = std.mem.sliceAsBytes(run_C[0 .. mi * ni]);
        for (0..ref_bytes.len) |bi| {
            if (ref_bytes[bi] != run_bytes[bi]) {
                result.all_identical = false;
                result.first_mismatch_run = @intCast(run);
                result.first_mismatch_byte = @intCast(bi);
                return result;
            }
        }
    }

    return result;
}

pub fn verifyDeterminismAttention(seq_len: i32, n_heads: i32, d_head: i32, n_runs: i32) DeterminismVerifyResult {
    const sl: usize = @intCast(seq_len);
    const nh: usize = @intCast(n_heads);
    const dh: usize = @intCast(d_head);
    const dm = nh * dh;
    const total = @min(sl * dm, 4096);
    const nr: usize = @intCast(n_runs);

    var Q_buf: [4096]Q16 = undefined;
    var K_buf: [4096]Q16 = undefined;
    var V_buf: [4096]Q16 = undefined;
    var ref_out: [4096]Q16 = undefined;
    var run_out: [4096]Q16 = undefined;

    for (0..total) |i| {
        Q_buf[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 3 + 1, 500)), .r0 = 0 };
        K_buf[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 5 + 2, 500)), .r0 = 0 };
        V_buf[i] = .{ .v = @intCast(@mod(@as(i32, @intCast(i)) * 7 + 3, 500)), .r0 = 0 };
    }

    const cfg = attention_kernel.AttentionKernelConfig{ .n_heads = n_heads, .d_head = d_head, .seq_len = seq_len, .causal_mask = true, .softmax_type = .quadratic };

    _ = attention_kernel.fusedAttentionForward(Q_buf[0..total], K_buf[0..total], V_buf[0..total], ref_out[0..total], null, cfg);

    var result = DeterminismVerifyResult{
        .kernel_name = nameBench("attention"),
        .kernel_name_len = 9,
        .n_runs = n_runs,
        .all_identical = true,
        .first_mismatch_run = -1,
        .first_mismatch_byte = -1,
    };

    for (0..nr) |run| {
        _ = attention_kernel.fusedAttentionForward(Q_buf[0..total], K_buf[0..total], V_buf[0..total], run_out[0..total], null, cfg);
        const ref_bytes = std.mem.sliceAsBytes(ref_out[0..total]);
        const run_bytes = std.mem.sliceAsBytes(run_out[0..total]);
        for (0..ref_bytes.len) |bi| {
            if (ref_bytes[bi] != run_bytes[bi]) {
                result.all_identical = false;
                result.first_mismatch_run = @intCast(run);
                result.first_mismatch_byte = @intCast(bi);
                return result;
            }
        }
    }

    return result;
}

pub fn runFullDeterminismSuite(n_runs: i32) [3]DeterminismVerifyResult {
    return .{
        verifyDeterminismSoftmax(64, n_runs),
        verifyDeterminismGemm(8, 8, 8, n_runs),
        verifyDeterminismAttention(4, 1, 8, n_runs),
    };
}
