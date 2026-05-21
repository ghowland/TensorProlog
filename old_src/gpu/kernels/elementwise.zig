// ============================================================
// src/gpu/kernels/elementwise.zig
// ============================================================

pub fn q16Add(a: []const Q16, b: []const Q16, out: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        out[i] = Q16.add(a[i], b[i]);
    }
    return .ok;
}

pub fn q16Sub(a: []const Q16, b: []const Q16, out: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        out[i] = Q16.sub(a[i], b[i]);
    }
    return .ok;
}

pub fn q16Mul(a: []const Q16, b: []const Q16, out: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        out[i] = Q16.mul(a[i], b[i]);
    }
    return .ok;
}

pub fn q16Div(a: []const Q16, b: []const Q16, out: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        if (b[i].v == 0) {
            out[i] = Q16.zero();
        } else {
            out[i] = Q16.div(a[i], b[i]);
        }
    }
    return .ok;
}

pub fn q16Scale(input: []const Q16, scalar: Q16, output: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        output[i] = Q16.mul(input[i], scalar);
    }
    return .ok;
}

pub fn q16Dot(a: []const Q16, b: []const Q16, n: i32) Q16 {
    const sz: usize = @intCast(n);
    var acc: i64 = 0;
    for (0..sz) |i| {
        const av: i64 = @intCast(a[i].v);
        const bv: i64 = @intCast(b[i].v);
        acc += av * bv;
    }
    return .{
        .v = @intCast(@divTrunc(acc, @as(i64, Q16.D))),
        .r0 = @intCast(@mod(acc, @as(i64, Q16.D))),
    };
}

pub fn q16Compare(a: []const Q16, b: []const Q16, result: []i32, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        result[i] = Q16.compare(a[i], b[i]);
    }
    return .ok;
}

pub fn q16Negate(input: []const Q16, output: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        output[i] = Q16.negate(input[i]);
    }
    return .ok;
}

pub fn q16Abs(input: []const Q16, output: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        output[i] = Q16.abs_val(input[i]);
    }
    return .ok;
}

pub fn q16Min(a: []const Q16, b: []const Q16, out: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        out[i] = Q16.min_val(a[i], b[i]);
    }
    return .ok;
}

pub fn q16Max(a: []const Q16, b: []const Q16, out: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        out[i] = Q16.max_val(a[i], b[i]);
    }
    return .ok;
}

pub fn q16Clamp(input: []const Q16, lo: Q16, hi: Q16, output: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        if (Q16.compare(input[i], lo) < 0) {
            output[i] = lo;
        } else if (Q16.compare(input[i], hi) > 0) {
            output[i] = hi;
        } else {
            output[i] = input[i];
        }
    }
    return .ok;
}

pub fn q16Fill(output: []Q16, value: Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        output[i] = value;
    }
    return .ok;
}

pub fn q16Copy(src: []const Q16, dst: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    @memcpy(dst[0..sz], src[0..sz]);
    return .ok;
}

pub fn q16Sum(data: []const Q16, n: i32) Q16 {
    const sz: usize = @intCast(n);
    var acc: i64 = 0;
    for (0..sz) |i| {
        acc += @intCast(data[i].v);
    }
    return .{
        .v = @intCast(acc),
        .r0 = 0,
    };
}

pub fn q16RemainderMagnitude(data: []const Q16, magnitudes: []i32, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        const r: i32 = data[i].r0;
        magnitudes[i] = if (r < 0) -r else r;
    }
    return .ok;
}
