
// ============================================================
// src/gpu/kernels/softmax.zig
// ============================================================

pub fn q16Softmax(input: []const Q16, output: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    if (sz == 0) return .ok;

    var min_v: i32 = input[0].v;
    for (input[1..sz]) |v| {
        if (v.v < min_v) min_v = v.v;
    }

    var shifted: [4096]i64 = undefined;
    if (sz > shifted.len) return .err_primitive_bounds;

    for (0..sz) |i| {
        shifted[i] = @intCast(input[i].v - min_v);
    }

    var sum_sq: i64 = 0;
    for (0..sz) |i| {
        sum_sq += shifted[i] * shifted[i];
    }

    if (sum_sq == 0) {
        const equal_share: i32 = @intCast(@divTrunc(@as(i64, Q16.D), @as(i64, @intCast(sz))));
        var running_sum: i64 = 0;
        for (0..sz - 1) |i| {
            output[i] = .{ .v = equal_share, .r0 = 0 };
            running_sum += @intCast(equal_share);
        }
        output[sz - 1] = .{ .v = @intCast(@as(i64, Q16.D) - running_sum), .r0 = 0 };
        return .ok;
    }

    var running_sum: i64 = 0;
    for (0..sz - 1) |i| {
        const weight: i64 = @divTrunc(shifted[i] * shifted[i] * @as(i64, Q16.D), sum_sq);
        output[i] = .{
            .v = @intCast(weight),
            .r0 = @intCast(shifted[i] * shifted[i] * @as(i64, Q16.D) - weight * sum_sq),
        };
        running_sum += weight;
    }
    output[sz - 1] = .{
        .v = @intCast(@as(i64, Q16.D) - running_sum),
        .r0 = 0,
    };

    return .ok;
}

pub fn q16SoftmaxBatched(input: []const Q16, output: []Q16, n: i32, batch_count: i32) VlpStatus {
    const sz: usize = @intCast(n);
    const bc: usize = @intCast(batch_count);

    for (0..bc) |b| {
        const offset = b * sz;
        const status = q16Softmax(input[offset .. offset + sz], output[offset .. offset + sz], n);
        if (status != .ok) return status;
    }

    return .ok;
}

pub fn verifySoftmaxSum(weights: []const Q16, n: i32, batch_count: i32) i32 {
    const sz: usize = @intCast(n);
    const bc: usize = @intCast(batch_count);
    var violations: i32 = 0;

    for (0..bc) |b| {
        var sum: i64 = 0;
        for (0..sz) |i| {
            sum += @intCast(weights[b * sz + i].v);
        }
        if (sum != @as(i64, Q16.D)) {
            violations += 1;
        }
    }

    return violations;
}
