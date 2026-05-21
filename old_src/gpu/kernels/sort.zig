// ============================================================
// src/gpu/kernels/sort.zig
// ============================================================

pub const SortEntry = struct {
    value: Q16,
    index: i32,
};

pub fn q16Sort(data: []Q16, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    if (sz <= 1) return .ok;
    mergeSortQ16(data, 0, sz);
    return .ok;
}

fn mergeSortQ16(data: []Q16, start: usize, end: usize) void {
    if (end - start <= 1) return;
    if (end - start == 2) {
        if (Q16.compare(data[start], data[start + 1]) > 0) {
            const tmp = data[start];
            data[start] = data[start + 1];
            data[start + 1] = tmp;
        }
        return;
    }
    const mid = start + (end - start) / 2;
    mergeSortQ16(data, start, mid);
    mergeSortQ16(data, mid, end);
    mergeQ16(data, start, mid, end);
}

fn mergeQ16(data: []Q16, start: usize, mid: usize, end: usize) void {
    var scratch: [4096]Q16 = undefined;
    const left_len = mid - start;
    if (left_len > scratch.len) return;
    @memcpy(scratch[0..left_len], data[start..mid]);
    var i: usize = 0;
    var j: usize = mid;
    var k: usize = start;
    while (i < left_len and j < end) {
        if (Q16.compare(scratch[i], data[j]) <= 0) {
            data[k] = scratch[i];
            i += 1;
        } else {
            data[k] = data[j];
            j += 1;
        }
        k += 1;
    }
    while (i < left_len) {
        data[k] = scratch[i];
        i += 1;
        k += 1;
    }
}

pub fn q16ArgSort(data: []const Q16, indices: []i32, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    for (0..sz) |i| {
        indices[i] = @intCast(i);
    }
    for (0..sz) |i| {
        var best = i;
        for (i + 1..sz) |j| {
            const bi: usize = @intCast(indices[best]);
            const ji: usize = @intCast(indices[j]);
            if (Q16.compare(data[ji], data[bi]) > 0) {
                best = j;
            }
        }
        if (best != i) {
            const tmp = indices[i];
            indices[i] = indices[best];
            indices[best] = tmp;
        }
    }
    return .ok;
}

pub fn q16TopK(data: []const Q16, k: i32, out_values: []Q16, out_indices: []i32, n: i32) VlpStatus {
    const sz: usize = @intCast(n);
    const ku: usize = @intCast(@min(k, n));

    var indices: [4096]i32 = undefined;
    if (sz > indices.len) return .err_primitive_bounds;

    for (0..sz) |i| {
        indices[i] = @intCast(i);
    }

    for (0..ku) |i| {
        var best = i;
        for (i + 1..sz) |j| {
            const bi: usize = @intCast(indices[best]);
            const ji: usize = @intCast(indices[j]);
            if (Q16.compare(data[ji], data[bi]) > 0) {
                best = j;
            }
        }
        if (best != i) {
            const tmp = indices[i];
            indices[i] = indices[best];
            indices[best] = tmp;
        }
    }

    for (0..ku) |i| {
        out_indices[i] = indices[i];
        out_values[i] = data[@intCast(indices[i])];
    }

    return .ok;
}
