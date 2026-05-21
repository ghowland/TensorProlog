// ============================================================
// src/builtins/collections.zig
// ============================================================

const std = @import("std");
const q16_mod = @import("../vdr/q16.zig");
const dispatch_mod = @import("dispatch.zig");

const Q16 = q16_mod.Q16;
const BuiltinArgs = dispatch_mod.BuiltinArgs;
const BuiltinResult = dispatch_mod.BuiltinResult;

pub const UnaryOp = enum(i8) {
    negate = 0,
    abs_val = 1,
    square = 2,
    double = 3,
    halve = 4,
};

pub const BinaryOp = enum(i8) {
    add = 0,
    sub = 1,
    mul = 2,
    min_val = 3,
    max_val = 4,
};

pub const Group = struct {
    key: i32,
    start: i32,
    count: i32,
};

pub const Pair = struct {
    a: Q16,
    b: Q16,
};

pub const IndexedValue = struct {
    index: i32,
    value: Q16,
};

fn applyUnary(op: UnaryOp, val: Q16) Q16 {
    return switch (op) {
        .negate => Q16.negate(val),
        .abs_val => Q16.abs_val(val),
        .square => Q16.mul(val, val),
        .double => Q16.add(val, val),
        .halve => .{
            .v = @divTrunc(val.v, 2),
            .r0 = @intCast(@mod(val.v, 2) * @divTrunc(Q16.D, 2) + @as(i32, val.r0)),
        },
    };
}

fn applyBinary(op: BinaryOp, a: Q16, b: Q16) Q16 {
    return switch (op) {
        .add => Q16.add(a, b),
        .sub => Q16.sub(a, b),
        .mul => Q16.mul(a, b),
        .min_val => Q16.min_val(a, b),
        .max_val => Q16.max_val(a, b),
    };
}

pub fn collSort(data: []Q16) void {
    mergeSortInPlace(data, 0, data.len);
}

fn mergeSortInPlace(data: []Q16, start: usize, end: usize) void {
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
    mergeSortInPlace(data, start, mid);
    mergeSortInPlace(data, mid, end);

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

pub fn collSortBy(data: []Q16, keys: []const Q16) void {
    const n = @min(data.len, keys.len);
    for (0..n) |i| {
        var min_idx = i;
        for (i + 1..n) |j| {
            if (Q16.compare(keys[j], keys[min_idx]) < 0) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            const tmp_d = data[i];
            data[i] = data[min_idx];
            data[min_idx] = tmp_d;
        }
    }
}

pub fn collFilter(data: []const Q16, mask: []const bool, output: []Q16) i32 {
    var out_pos: usize = 0;
    const n = @min(data.len, mask.len);

    for (0..n) |i| {
        if (mask[i]) {
            if (out_pos >= output.len) break;
            output[out_pos] = data[i];
            out_pos += 1;
        }
    }

    return @intCast(out_pos);
}

pub fn collMap(data: []const Q16, op: UnaryOp, output: []Q16) void {
    const n = @min(data.len, output.len);
    for (0..n) |i| {
        output[i] = applyUnary(op, data[i]);
    }
}

pub fn collReduce(data: []const Q16, op: BinaryOp, initial: Q16) Q16 {
    var acc = initial;
    for (data) |val| {
        acc = applyBinary(op, acc, val);
    }
    return acc;
}

pub fn collGroupBy(data: []const Q16, keys: []const i32, groups: []Group) i32 {
    var group_count: usize = 0;
    const n = @min(data.len, keys.len);

    for (0..n) |i| {
        var found = false;
        for (0..group_count) |g| {
            if (groups[g].key == keys[i]) {
                groups[g].count += 1;
                found = true;
                break;
            }
        }
        if (!found and group_count < groups.len) {
            groups[group_count] = .{
                .key = keys[i],
                .start = @intCast(i),
                .count = 1,
            };
            group_count += 1;
        }
    }

    return @intCast(group_count);
}

pub fn collFrequencies(data: []const Q16, values: []Q16, counts: []i32) i32 {
    var unique_count: usize = 0;

    for (data) |val| {
        var found = false;
        for (0..unique_count) |u| {
            if (Q16.eql(values[u], val)) {
                counts[u] += 1;
                found = true;
                break;
            }
        }
        if (!found and unique_count < values.len) {
            values[unique_count] = val;
            counts[unique_count] = 1;
            unique_count += 1;
        }
    }

    return @intCast(unique_count);
}

pub fn collDistinct(data: []const Q16, output: []Q16) i32 {
    var count: usize = 0;

    for (data) |val| {
        var found = false;
        for (0..count) |u| {
            if (Q16.eql(output[u], val)) {
                found = true;
                break;
            }
        }
        if (!found and count < output.len) {
            output[count] = val;
            count += 1;
        }
    }

    return @intCast(count);
}

pub fn collFlatten(arrays: []const []const Q16, output: []Q16) i32 {
    var pos: usize = 0;
    for (arrays) |arr| {
        for (arr) |val| {
            if (pos >= output.len) return @intCast(pos);
            output[pos] = val;
            pos += 1;
        }
    }
    return @intCast(pos);
}

pub fn collChunk(data: []const Q16, chunk_size: i32, chunks: [][]const Q16) i32 {
    if (chunk_size <= 0) return 0;
    const cs: usize = @intCast(chunk_size);
    var count: usize = 0;
    var pos: usize = 0;

    while (pos < data.len and count < chunks.len) {
        const end = @min(pos + cs, data.len);
        chunks[count] = data[pos..end];
        count += 1;
        pos = end;
    }

    return @intCast(count);
}

pub fn collZip(a: []const Q16, b: []const Q16, output: []Pair) i32 {
    const n = @min(@min(a.len, b.len), output.len);
    for (0..n) |i| {
        output[i] = .{ .a = a[i], .b = b[i] };
    }
    return @intCast(n);
}

pub fn collUnzip(pairs: []const Pair, a: []Q16, b: []Q16) void {
    const n = @min(@min(pairs.len, a.len), b.len);
    for (0..n) |i| {
        a[i] = pairs[i].a;
        b[i] = pairs[i].b;
    }
}

pub fn collReverse(data: []Q16) void {
    if (data.len < 2) return;
    var lo: usize = 0;
    var hi: usize = data.len - 1;
    while (lo < hi) {
        const tmp = data[lo];
        data[lo] = data[hi];
        data[hi] = tmp;
        lo += 1;
        hi -= 1;
    }
}

pub fn collRotate(data: []Q16, amount: i32) void {
    if (data.len == 0) return;
    const n: i32 = @intCast(data.len);
    var a = @mod(amount, n);
    if (a < 0) a += n;
    if (a == 0) return;

    const au: usize = @intCast(a);
    collReverse(data[0..au]);
    collReverse(data[au..]);
    collReverse(data);
}

pub fn collTakeFirst(data: []const Q16, count: i32, output: []Q16) i32 {
    const c: usize = @intCast(@max(count, 0));
    const n = @min(@min(c, data.len), output.len);
    @memcpy(output[0..n], data[0..n]);
    return @intCast(n);
}

pub fn collTakeLast(data: []const Q16, count: i32, output: []Q16) i32 {
    const c: usize = @intCast(@max(count, 0));
    const take = @min(c, data.len);
    const start = data.len - take;
    const n = @min(take, output.len);
    @memcpy(output[0..n], data[start .. start + n]);
    return @intCast(n);
}

pub fn collDropFirst(data: []const Q16, count: i32, output: []Q16) i32 {
    const c: usize = @intCast(@max(count, 0));
    const start = @min(c, data.len);
    const remaining = data.len - start;
    const n = @min(remaining, output.len);
    @memcpy(output[0..n], data[start .. start + n]);
    return @intCast(n);
}

pub fn collDropLast(data: []const Q16, count: i32, output: []Q16) i32 {
    const c: usize = @intCast(@max(count, 0));
    const drop = @min(c, data.len);
    const remaining = data.len - drop;
    const n = @min(remaining, output.len);
    @memcpy(output[0..n], data[0..n]);
    return @intCast(n);
}

pub fn collPartition(data: []const Q16, pred: []const bool, true_out: []Q16, false_out: []Q16) struct { n_true: i32, n_false: i32 } {
    var t: usize = 0;
    var f: usize = 0;
    const n = @min(data.len, pred.len);

    for (0..n) |i| {
        if (pred[i]) {
            if (t < true_out.len) {
                true_out[t] = data[i];
                t += 1;
            }
        } else {
            if (f < false_out.len) {
                false_out[f] = data[i];
                f += 1;
            }
        }
    }

    return .{ .n_true = @intCast(t), .n_false = @intCast(f) };
}

pub fn collInterleave(a: []const Q16, b: []const Q16, output: []Q16) i32 {
    var pos: usize = 0;
    const n = @min(a.len, b.len);

    for (0..n) |i| {
        if (pos + 1 >= output.len) break;
        output[pos] = a[i];
        pos += 1;
        output[pos] = b[i];
        pos += 1;
    }

    if (a.len > n) {
        for (n..a.len) |i| {
            if (pos >= output.len) break;
            output[pos] = a[i];
            pos += 1;
        }
    }

    if (b.len > n) {
        for (n..b.len) |i| {
            if (pos >= output.len) break;
            output[pos] = b[i];
            pos += 1;
        }
    }

    return @intCast(pos);
}

pub fn collEnumerate(data: []const Q16, output: []IndexedValue) void {
    const n = @min(data.len, output.len);
    for (0..n) |i| {
        output[i] = .{ .index = @intCast(i), .value = data[i] };
    }
}

pub fn collMinBy(data: []const Q16, keys: []const Q16) Q16 {
    if (data.len == 0) return Q16.zero();
    const n = @min(data.len, keys.len);
    var min_idx: usize = 0;

    for (1..n) |i| {
        if (Q16.compare(keys[i], keys[min_idx]) < 0) {
            min_idx = i;
        }
    }

    return data[min_idx];
}

pub fn collMaxBy(data: []const Q16, keys: []const Q16) Q16 {
    if (data.len == 0) return Q16.zero();
    const n = @min(data.len, keys.len);
    var max_idx: usize = 0;

    for (1..n) |i| {
        if (Q16.compare(keys[i], keys[max_idx]) > 0) {
            max_idx = i;
        }
    }

    return data[max_idx];
}

pub fn collScan(data: []const Q16, op: BinaryOp, output: []Q16) void {
    if (data.len == 0) return;
    const n = @min(data.len, output.len);
    output[0] = data[0];

    for (1..n) |i| {
        output[i] = applyBinary(op, output[i - 1], data[i]);
    }
}

pub fn collAll(preds: []const bool) bool {
    for (preds) |p| {
        if (!p) return false;
    }
    return true;
}

pub fn collAny(preds: []const bool) bool {
    for (preds) |p| {
        if (p) return true;
    }
    return false;
}

pub fn collNone(preds: []const bool) bool {
    for (preds) |p| {
        if (p) return false;
    }
    return true;
}

pub fn collCount(preds: []const bool) i32 {
    var c: i32 = 0;
    for (preds) |p| {
        if (p) c += 1;
    }
    return c;
}

pub fn collFindFirst(data: []const Q16, target: Q16) ?i32 {
    for (data, 0..) |val, i| {
        if (Q16.eql(val, target)) return @intCast(i);
    }
    return null;
}

pub fn collFindLast(data: []const Q16, target: Q16) ?i32 {
    var last: ?i32 = null;
    for (data, 0..) |val, i| {
        if (Q16.eql(val, target)) last = @intCast(i);
    }
    return last;
}

pub fn collFindAll(data: []const Q16, target: Q16, indices: []i32) i32 {
    var count: usize = 0;
    for (data, 0..) |val, i| {
        if (Q16.eql(val, target)) {
            if (count >= indices.len) break;
            indices[count] = @intCast(i);
            count += 1;
        }
    }
    return @intCast(count);
}

pub fn collBinarySearch(sorted: []const Q16, target: Q16) ?i32 {
    if (sorted.len == 0) return null;

    var lo: usize = 0;
    var hi: usize = sorted.len;

    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const cmp = Q16.compare(sorted[mid], target);
        if (cmp == 0) return @intCast(mid);
        if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    return null;
}

pub fn collMerge(a: []const Q16, b: []const Q16, output: []Q16) i32 {
    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    while (i < a.len and j < b.len and k < output.len) {
        if (Q16.compare(a[i], b[j]) <= 0) {
            output[k] = a[i];
            i += 1;
        } else {
            output[k] = b[j];
            j += 1;
        }
        k += 1;
    }

    while (i < a.len and k < output.len) {
        output[k] = a[i];
        i += 1;
        k += 1;
    }

    while (j < b.len and k < output.len) {
        output[k] = b[j];
        j += 1;
        k += 1;
    }

    return @intCast(k);
}

pub fn collDeduplicate(sorted: []Q16) i32 {
    if (sorted.len <= 1) return @intCast(sorted.len);

    var write: usize = 1;
    for (1..sorted.len) |i| {
        if (!Q16.eql(sorted[i], sorted[write - 1])) {
            sorted[write] = sorted[i];
            write += 1;
        }
    }

    return @intCast(write);
}

pub fn collWindow(data: []const Q16, window_size: i32, windows: [][]const Q16) i32 {
    if (window_size <= 0) return 0;
    const ws: usize = @intCast(window_size);
    if (ws > data.len) return 0;

    var count: usize = 0;
    const n_windows = data.len - ws + 1;
    const limit = @min(n_windows, windows.len);

    for (0..limit) |i| {
        windows[i] = data[i .. i + ws];
        count += 1;
    }

    return @intCast(count);
}

pub fn collCartesianProduct(a: []const Q16, b: []const Q16, output: []Pair) i32 {
    var count: usize = 0;
    for (a) |va| {
        for (b) |vb| {
            if (count >= output.len) return @intCast(count);
            output[count] = .{ .a = va, .b = vb };
            count += 1;
        }
    }
    return @intCast(count);
}
