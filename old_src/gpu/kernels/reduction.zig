// ============================================================
// src/gpu/kernels/reduction.zig
// ============================================================

pub fn q16ReduceSum(data: []const Q16, n: i32) Q16 {
    const sz: usize = @intCast(n);
    var acc: i64 = 0;
    for (0..sz) |i| {
        acc += @intCast(data[i].v);
    }
    return .{ .v = @intCast(acc), .r0 = 0 };
}

pub fn q16ReduceMax(data: []const Q16, n: i32) Q16 {
    const sz: usize = @intCast(n);
    if (sz == 0) return Q16.zero();
    var max_val = data[0];
    for (1..sz) |i| {
        if (Q16.compare(data[i], max_val) > 0) max_val = data[i];
    }
    return max_val;
}

pub fn q16ReduceMin(data: []const Q16, n: i32) Q16 {
    const sz: usize = @intCast(n);
    if (sz == 0) return Q16.zero();
    var min_val = data[0];
    for (1..sz) |i| {
        if (Q16.compare(data[i], min_val) < 0) min_val = data[i];
    }
    return min_val;
}

pub fn q16ReduceArgMax(data: []const Q16, n: i32) i32 {
    const sz: usize = @intCast(n);
    if (sz == 0) return -1;
    var max_idx: usize = 0;
    for (1..sz) |i| {
        if (Q16.compare(data[i], data[max_idx]) > 0) max_idx = i;
    }
    return @intCast(max_idx);
}

pub fn q16ReduceArgMin(data: []const Q16, n: i32) i32 {
    const sz: usize = @intCast(n);
    if (sz == 0) return -1;
    var min_idx: usize = 0;
    for (1..sz) |i| {
        if (Q16.compare(data[i], data[min_idx]) < 0) min_idx = i;
    }
    return @intCast(min_idx);
}

pub fn i32ReduceSum(data: []const i32, n: i32) i64 {
    const sz: usize = @intCast(n);
    var acc: i64 = 0;
    for (0..sz) |i| {
        acc += @intCast(data[i]);
    }
    return acc;
}

pub fn allReduceSum(data: []Q16, n: i32) VlpStatus {
    _ = data;
    _ = n;
    return .ok;
}

pub fn allReduceMax(data: []Q16, n: i32) VlpStatus {
    _ = data;
    _ = n;
    return .ok;
}
