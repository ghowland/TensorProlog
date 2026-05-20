// ============================================================
// src/builtins/linalg.zig
// ============================================================

const std = @import("std");
const q16_mod = @import("../vdr/q16.zig");
const types = @import("../vdr/types.zig");
const dispatch_mod = @import("dispatch.zig");

const Q16 = q16_mod.Q16;
const VlpStatus = types.VlpStatus;
const BuiltinArgs = dispatch_mod.BuiltinArgs;
const BuiltinResult = dispatch_mod.BuiltinResult;

pub fn matVecMul(A: []const Q16, x: []const Q16, y: []Q16, m: i32, n: i32) void {
    const rows: usize = @intCast(m);
    const cols: usize = @intCast(n);
    for (0..rows) |r| {
        var acc: i64 = 0;
        for (0..cols) |c| {
            const av: i64 = @intCast(A[r * cols + c].v);
            const xv: i64 = @intCast(x[c].v);
            acc += av * xv;
        }
        y[r] = .{
            .v = @intCast(@divTrunc(acc, @as(i64, Q16.D))),
            .r0 = @intCast(@mod(acc, @as(i64, Q16.D))),
        };
    }
}

pub fn transpose(input: []const Q16, output: []Q16, m: i32, n: i32) void {
    const rows: usize = @intCast(m);
    const cols: usize = @intCast(n);
    for (0..rows) |r| {
        for (0..cols) |c| {
            output[c * rows + r] = input[r * cols + c];
        }
    }
}

pub fn gaussianElim(A: []Q16, b: []Q16, x: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);

    for (0..sz) |col| {
        var pivot_row = col;
        while (pivot_row < sz and A[pivot_row * sz + col].v == 0) {
            pivot_row += 1;
        }
        if (pivot_row >= sz) return .err_remainder_overflow;

        if (pivot_row != col) {
            for (0..sz) |j| {
                const tmp = A[col * sz + j];
                A[col * sz + j] = A[pivot_row * sz + j];
                A[pivot_row * sz + j] = tmp;
            }
            const tmp_b = b[col];
            b[col] = b[pivot_row];
            b[pivot_row] = tmp_b;
        }

        const pivot_v: i64 = @intCast(A[col * sz + col].v);
        if (pivot_v == 0) return .err_remainder_overflow;

        for (col + 1..sz) |row| {
            const row_v: i64 = @intCast(A[row * sz + col].v);
            if (row_v == 0) continue;

            for (0..sz) |j| {
                const a_col_j: i64 = @intCast(A[col * sz + j].v);
                const a_row_j: i64 = @intCast(A[row * sz + j].v);
                const new_val: i64 = a_row_j * pivot_v - row_v * a_col_j;
                A[row * sz + j] = .{
                    .v = @intCast(@divTrunc(new_val, @as(i64, Q16.D))),
                    .r0 = @intCast(@mod(new_val, @as(i64, Q16.D))),
                };
            }
            const b_col: i64 = @intCast(b[col].v);
            const b_row: i64 = @intCast(b[row].v);
            const new_b: i64 = b_row * pivot_v - row_v * b_col;
            b[row] = .{
                .v = @intCast(@divTrunc(new_b, @as(i64, Q16.D))),
                .r0 = @intCast(@mod(new_b, @as(i64, Q16.D))),
            };
        }
    }

    var i_rev: usize = sz;
    while (i_rev > 0) {
        i_rev -= 1;
        var sum: i64 = @intCast(b[i_rev].v);
        sum *= @as(i64, Q16.D);
        for (i_rev + 1..sz) |j| {
            const a_val: i64 = @intCast(A[i_rev * sz + j].v);
            const x_val: i64 = @intCast(x[j].v);
            sum -= a_val * x_val;
        }
        const diag: i64 = @intCast(A[i_rev * sz + i_rev].v);
        if (diag == 0) return .err_remainder_overflow;
        x[i_rev] = .{
            .v = @intCast(@divTrunc(sum, diag)),
            .r0 = @intCast(@mod(sum, diag)),
        };
    }

    return .ok;
}

pub fn inverse(A: []const Q16, A_inv: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    var aug: [4096]Q16 = undefined;
    if (sz * sz * 2 > aug.len) return .err_primitive_bounds;

    for (0..sz) |r| {
        for (0..sz) |c| {
            aug[r * (sz * 2) + c] = A[r * sz + c];
        }
        for (0..sz) |c| {
            aug[r * (sz * 2) + sz + c] = if (r == c) Q16.one() else Q16.zero();
        }
    }

    const w = sz * 2;

    for (0..sz) |col| {
        var pivot_row = col;
        var max_abs: i32 = absVal(aug[col * w + col].v);
        for (col + 1..sz) |r| {
            const a = absVal(aug[r * w + col].v);
            if (a > max_abs) {
                max_abs = a;
                pivot_row = r;
            }
        }
        if (max_abs == 0) return .err_remainder_overflow;

        if (pivot_row != col) {
            for (0..w) |j| {
                const tmp = aug[col * w + j];
                aug[col * w + j] = aug[pivot_row * w + j];
                aug[pivot_row * w + j] = tmp;
            }
        }

        const pivot_v: i64 = @intCast(aug[col * w + col].v);

        for (0..sz) |row| {
            if (row == col) continue;
            const row_v: i64 = @intCast(aug[row * w + col].v);
            if (row_v == 0) continue;
            for (0..w) |j| {
                const a_col_j: i64 = @intCast(aug[col * w + j].v);
                const a_row_j: i64 = @intCast(aug[row * w + j].v);
                const new_val: i64 = a_row_j * pivot_v - row_v * a_col_j;
                aug[row * w + j] = .{
                    .v = @intCast(@divTrunc(new_val, @as(i64, Q16.D))),
                    .r0 = @intCast(@mod(new_val, @as(i64, Q16.D))),
                };
            }
        }
    }

    for (0..sz) |r| {
        const diag: i64 = @intCast(aug[r * w + r].v);
        if (diag == 0) return .err_remainder_overflow;
        for (0..sz) |c| {
            const val: i64 = @intCast(aug[r * w + sz + c].v);
            const scaled: i64 = val * @as(i64, Q16.D);
            A_inv[r * sz + c] = .{
                .v = @intCast(@divTrunc(scaled, diag)),
                .r0 = @intCast(@mod(scaled, diag)),
            };
        }
    }

    return .ok;
}

pub fn determinant(A: []const Q16, n: i32) Q16 {
    const sz: usize = @intCast(n);
    var work: [4096]Q16 = undefined;
    if (sz * sz > work.len) return Q16.zero();

    @memcpy(work[0 .. sz * sz], A[0 .. sz * sz]);

    var det_num: i64 = @as(i64, Q16.D);
    var det_den: i64 = 1;
    var sign: i64 = 1;

    for (0..sz) |col| {
        var pivot_row = col;
        while (pivot_row < sz and work[pivot_row * sz + col].v == 0) {
            pivot_row += 1;
        }
        if (pivot_row >= sz) return Q16.zero();

        if (pivot_row != col) {
            for (0..sz) |j| {
                const tmp = work[col * sz + j];
                work[col * sz + j] = work[pivot_row * sz + j];
                work[pivot_row * sz + j] = tmp;
            }
            sign = -sign;
        }

        const pivot_v: i64 = @intCast(work[col * sz + col].v);
        det_num = @divTrunc(det_num * pivot_v, @as(i64, Q16.D));

        for (col + 1..sz) |row| {
            const row_v: i64 = @intCast(work[row * sz + col].v);
            if (row_v == 0) continue;
            for (col + 1..sz) |j| {
                const a_col_j: i64 = @intCast(work[col * sz + j].v);
                const a_row_j: i64 = @intCast(work[row * sz + j].v);
                const new_val: i64 = a_row_j * pivot_v - row_v * a_col_j;
                work[row * sz + j] = .{
                    .v = @intCast(@divTrunc(new_val, @as(i64, Q16.D))),
                    .r0 = 0,
                };
            }
            work[row * sz + col] = Q16.zero();
            det_den *= pivot_v;
        }
    }

    const result: i64 = @divTrunc(det_num * sign * @as(i64, Q16.D), @max(det_den, 1));
    return .{ .v = @intCast(result), .r0 = 0 };
}

pub fn gramSchmidt(vectors: []const Q16, orthogonal: []Q16, n_vectors: i32, dim: i32) void {
    const nv: usize = @intCast(n_vectors);
    const d: usize = @intCast(dim);

    for (0..d) |j| {
        orthogonal[j] = vectors[j];
    }

    for (1..nv) |i| {
        for (0..d) |j| {
            orthogonal[i * d + j] = vectors[i * d + j];
        }

        for (0..i) |k| {
            var dot_vu: i64 = 0;
            var dot_uu: i64 = 0;
            for (0..d) |j| {
                const vj: i64 = @intCast(vectors[i * d + j].v);
                const uj: i64 = @intCast(orthogonal[k * d + j].v);
                dot_vu += vj * uj;
                dot_uu += uj * uj;
            }

            if (dot_uu == 0) continue;

            for (0..d) |j| {
                const uj: i64 = @intCast(orthogonal[k * d + j].v);
                const proj: i64 = @divTrunc(dot_vu * uj, dot_uu);
                const cur: i64 = @intCast(orthogonal[i * d + j].v);
                orthogonal[i * d + j] = .{
                    .v = @intCast(cur - proj),
                    .r0 = 0,
                };
            }
        }
    }
}

pub fn eigenvalues(A: []const Q16, eig_real: []Q16, eig_imag: []Q16, n: i32) void {
    const sz: usize = @intCast(n);

    if (sz == 1) {
        eig_real[0] = A[0];
        eig_imag[0] = Q16.zero();
        return;
    }

    if (sz == 2) {
        const a: i64 = @intCast(A[0].v);
        const b: i64 = @intCast(A[1].v);
        const c: i64 = @intCast(A[2].v);
        const d: i64 = @intCast(A[3].v);

        const trace: i64 = a + d;
        const det_val: i64 = @divTrunc(a * d - b * c, @as(i64, Q16.D));
        const disc: i64 = @divTrunc(trace * trace, @as(i64, Q16.D)) - 4 * det_val;

        if (disc >= 0) {
            const sqrt_disc = intSqrt(disc * @as(i64, Q16.D));
            eig_real[0] = .{ .v = @intCast(@divTrunc(trace + sqrt_disc, 2)), .r0 = 0 };
            eig_real[1] = .{ .v = @intCast(@divTrunc(trace - sqrt_disc, 2)), .r0 = 0 };
            eig_imag[0] = Q16.zero();
            eig_imag[1] = Q16.zero();
        } else {
            const sqrt_neg_disc = intSqrt(-disc * @as(i64, Q16.D));
            eig_real[0] = .{ .v = @intCast(@divTrunc(trace, 2)), .r0 = 0 };
            eig_real[1] = eig_real[0];
            eig_imag[0] = .{ .v = @intCast(@divTrunc(sqrt_neg_disc, 2)), .r0 = 0 };
            eig_imag[1] = .{ .v = @intCast(-@divTrunc(sqrt_neg_disc, 2)), .r0 = 0 };
        }
        return;
    }

    for (0..sz) |i| {
        eig_real[i] = A[i * sz + i];
        eig_imag[i] = Q16.zero();
    }
}

pub fn svd(A: []const Q16, U: []Q16, S: []Q16, Vt: []Q16, m: i32, n: i32) void {
    const rows: usize = @intCast(m);
    const cols: usize = @intCast(n);
    const k = @min(rows, cols);

    var AtA: [4096]Q16 = undefined;
    if (cols * cols > AtA.len) return;

    for (0..cols) |i| {
        for (0..cols) |j| {
            var acc: i64 = 0;
            for (0..rows) |r| {
                const ai: i64 = @intCast(A[r * cols + i].v);
                const aj: i64 = @intCast(A[r * cols + j].v);
                acc += ai * aj;
            }
            AtA[i * cols + j] = .{
                .v = @intCast(@divTrunc(acc, @as(i64, Q16.D))),
                .r0 = 0,
            };
        }
    }

    var eig_r: [64]Q16 = undefined;
    var eig_i: [64]Q16 = undefined;
    eigenvalues(&AtA, &eig_r, &eig_i, n);

    for (0..k) |i| {
        const ev: i64 = @intCast(eig_r[i].v);
        if (ev > 0) {
            S[i] = .{ .v = @intCast(intSqrt(ev * @as(i64, Q16.D))), .r0 = 0 };
        } else {
            S[i] = Q16.zero();
        }
    }

    for (0..rows) |r| {
        for (0..k) |c| {
            U[r * k + c] = if (r == c) Q16.one() else Q16.zero();
        }
    }
    for (0..k) |r| {
        for (0..cols) |c| {
            Vt[r * cols + c] = if (r == c) Q16.one() else Q16.zero();
        }
    }
}

fn absVal(v: i32) i32 {
    return if (v < 0) -v else v;
}

pub fn intSqrt(val: i64) i64 {
    if (val <= 0) return 0;
    var x: i64 = val;
    var y: i64 = (x + 1) / 2;
    while (y < x) {
        x = y;
        y = (x + @divTrunc(val, x)) / 2;
    }
    return x;
}

pub fn builtinMatVecMul(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinTranspose(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinGaussianElim(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinInverse(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinDeterminant(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinGramSchmidt(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinEigenvalues(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinSvd(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
