// ============================================================
// src/gpu/kernels/normalize.zig
// ============================================================

pub fn q16LayerNorm(
    input: []const Q16,
    output: []Q16,
    gamma: []const Q16,
    beta: []const Q16,
    n: i32,
) VlpStatus {
    const sz: usize = @intCast(n);
    if (sz == 0) return .ok;

    var sum: i64 = 0;
    for (0..sz) |i| {
        sum += @intCast(input[i].v);
    }
    const mean: i64 = @divTrunc(sum, @as(i64, @intCast(sz)));

    var var_sum: i64 = 0;
    for (0..sz) |i| {
        const diff: i64 = @as(i64, @intCast(input[i].v)) - mean;
        var_sum += @divTrunc(diff * diff, @as(i64, Q16.D));
    }
    const variance: i64 = @divTrunc(var_sum, @as(i64, @intCast(sz)));

    var inv_std: i64 = @as(i64, Q16.D);
    if (variance > 0) {
        var x: i64 = variance;
        var y: i64 = (x + 1) / 2;
        while (y < x) {
            x = y;
            y = (x + @divTrunc(variance, x)) / 2;
        }
        if (x > 0) {
            inv_std = @divTrunc(@as(i64, Q16.D) * @as(i64, Q16.D), x);
        }
    }

    for (0..sz) |i| {
        const diff: i64 = @as(i64, @intCast(input[i].v)) - mean;
        const normalized: i64 = @divTrunc(diff * inv_std, @as(i64, Q16.D));

        const gv: i64 = @intCast(gamma[i].v);
        const bv: i64 = @intCast(beta[i].v);
        const scaled: i64 = @divTrunc(normalized * gv, @as(i64, Q16.D)) + bv;

        output[i] = .{
            .v = @intCast(scaled),
            .r0 = 0,
        };
    }

    return .ok;
}

pub fn q16RMSNorm(
    input: []const Q16,
    output: []Q16,
    gamma: []const Q16,
    n: i32,
) VlpStatus {
    const sz: usize = @intCast(n);
    if (sz == 0) return .ok;

    var sum_sq: i64 = 0;
    for (0..sz) |i| {
        const v: i64 = @intCast(input[i].v);
        sum_sq += @divTrunc(v * v, @as(i64, Q16.D));
    }
    const mean_sq: i64 = @divTrunc(sum_sq, @as(i64, @intCast(sz)));

    var inv_rms: i64 = @as(i64, Q16.D);
    if (mean_sq > 0) {
        var x: i64 = mean_sq;
        var y: i64 = (x + 1) / 2;
        while (y < x) {
            x = y;
            y = (x + @divTrunc(mean_sq, x)) / 2;
        }
        if (x > 0) {
            inv_rms = @divTrunc(@as(i64, Q16.D) * @as(i64, Q16.D), x);
        }
    }

    for (0..sz) |i| {
        const v: i64 = @intCast(input[i].v);
        const normalized: i64 = @divTrunc(v * inv_rms, @as(i64, Q16.D));
        const gv: i64 = @intCast(gamma[i].v);
        const scaled: i64 = @divTrunc(normalized * gv, @as(i64, Q16.D));

        output[i] = .{
            .v = @intCast(scaled),
            .r0 = 0,
        };
    }

    return .ok;
}
