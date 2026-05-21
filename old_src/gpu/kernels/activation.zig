// ============================================================
// src/gpu/kernels/activation.zig
// ============================================================

pub fn q16ReLU(input: []const Q16, output: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        output[i] = if (input[i].v > 0) input[i] else Q16.zero();
    }
    return .ok;
}

pub fn q16GELU(input: []const Q16, output: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        const v: i64 = @intCast(input[i].v);
        if (v >= @as(i64, Q16.D) * 3) {
            output[i] = input[i];
        } else if (v <= -@as(i64, Q16.D) * 3) {
            output[i] = Q16.zero();
        } else {
            const abs_v = if (v < 0) -v else v;
            const sigmoid_approx: i64 = @divTrunc(@as(i64, Q16.D) * (v + @as(i64, Q16.D) * 3), @as(i64, Q16.D) * 6);
            const clamped = @min(@max(sigmoid_approx, 0), @as(i64, Q16.D));
            output[i] = .{
                .v = @intCast(@divTrunc(v * clamped, @as(i64, Q16.D))),
                .r0 = 0,
            };
            _ = abs_v;
        }
    }
    return .ok;
}

pub fn q16SiLU(input: []const Q16, output: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        const v: i64 = @intCast(input[i].v);
        const sigmoid: i64 = if (v >= 0)
            @divTrunc(@as(i64, Q16.D) * @as(i64, Q16.D), @as(i64, Q16.D) + @divTrunc(@as(i64, Q16.D) * @as(i64, Q16.D), @max(@as(i64, Q16.D) + v, 1)))
        else blk: {
            const exp_approx = @max(@as(i64, Q16.D) + v, 0);
            break :blk @divTrunc(exp_approx * @as(i64, Q16.D), @max(@as(i64, Q16.D) + exp_approx, 1));
        };
        output[i] = .{
            .v = @intCast(@divTrunc(v * sigmoid, @as(i64, Q16.D))),
            .r0 = 0,
        };
    }
    return .ok;
}
