// ============================================================
// src/builtins/sets.zig
// ============================================================

const q16_set = @import("../vdr/q16.zig");
const Q16S = q16_set.Q16;

fn sortedInsert(set: []Q16S, set_len: usize, element: Q16S) usize {
    var pos: usize = 0;
    while (pos < set_len and Q16S.compare(set[pos], element) < 0) {
        pos += 1;
    }
    if (pos < set_len and Q16S.eql(set[pos], element)) {
        return set_len;
    }

    var i = set_len;
    while (i > pos) {
        set[i] = set[i - 1];
        i -= 1;
    }
    set[pos] = element;
    return set_len + 1;
}

fn sortedContains(set: []const Q16S, element: Q16S) bool {
    var lo: usize = 0;
    var hi: usize = set.len;

    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const cmp = Q16S.compare(set[mid], element);
        if (cmp == 0) return true;
        if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return false;
}

pub fn setUnion(a: []const Q16S, b: []const Q16S, out: []Q16S) i32 {
    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    while (i < a.len and j < b.len and k < out.len) {
        const cmp = Q16S.compare(a[i], b[j]);
        if (cmp < 0) {
            out[k] = a[i];
            i += 1;
            k += 1;
        } else if (cmp > 0) {
            out[k] = b[j];
            j += 1;
            k += 1;
        } else {
            out[k] = a[i];
            i += 1;
            j += 1;
            k += 1;
        }
    }

    while (i < a.len and k < out.len) {
        out[k] = a[i];
        i += 1;
        k += 1;
    }

    while (j < b.len and k < out.len) {
        out[k] = b[j];
        j += 1;
        k += 1;
    }

    return @intCast(k);
}

pub fn setIntersection(a: []const Q16S, b: []const Q16S, out: []Q16S) i32 {
    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    while (i < a.len and j < b.len and k < out.len) {
        const cmp = Q16S.compare(a[i], b[j]);
        if (cmp < 0) {
            i += 1;
        } else if (cmp > 0) {
            j += 1;
        } else {
            out[k] = a[i];
            i += 1;
            j += 1;
            k += 1;
        }
    }

    return @intCast(k);
}

pub fn setDifference(a: []const Q16S, b: []const Q16S, out: []Q16S) i32 {
    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    while (i < a.len and k < out.len) {
        if (j >= b.len) {
            out[k] = a[i];
            i += 1;
            k += 1;
            continue;
        }

        const cmp = Q16S.compare(a[i], b[j]);
        if (cmp < 0) {
            out[k] = a[i];
            i += 1;
            k += 1;
        } else if (cmp > 0) {
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }

    return @intCast(k);
}

pub fn setSymmetricDiff(a: []const Q16S, b: []const Q16S, out: []Q16S) i32 {
    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    while (i < a.len and j < b.len and k < out.len) {
        const cmp = Q16S.compare(a[i], b[j]);
        if (cmp < 0) {
            out[k] = a[i];
            i += 1;
            k += 1;
        } else if (cmp > 0) {
            out[k] = b[j];
            j += 1;
            k += 1;
        } else {
            i += 1;
            j += 1;
        }
    }

    while (i < a.len and k < out.len) {
        out[k] = a[i];
        i += 1;
        k += 1;
    }

    while (j < b.len and k < out.len) {
        out[k] = b[j];
        j += 1;
        k += 1;
    }

    return @intCast(k);
}

pub fn setIsSubset(a: []const Q16S, b: []const Q16S) bool {
    var i: usize = 0;
    var j: usize = 0;

    while (i < a.len) {
        if (j >= b.len) return false;

        const cmp = Q16S.compare(a[i], b[j]);
        if (cmp < 0) return false;
        if (cmp == 0) {
            i += 1;
            j += 1;
        } else {
            j += 1;
        }
    }

    return true;
}

pub fn setIsSuperset(a: []const Q16S, b: []const Q16S) bool {
    return setIsSubset(b, a);
}

pub fn setIsDisjoint(a: []const Q16S, b: []const Q16S) bool {
    var i: usize = 0;
    var j: usize = 0;

    while (i < a.len and j < b.len) {
        const cmp = Q16S.compare(a[i], b[j]);
        if (cmp == 0) return false;
        if (cmp < 0) {
            i += 1;
        } else {
            j += 1;
        }
    }

    return true;
}

pub fn setContains(set: []const Q16S, element: Q16S) bool {
    return sortedContains(set, element);
}

pub fn setAdd(set: []Q16S, set_len: *i32, element: Q16S, capacity: i32) bool {
    const sl: usize = @intCast(set_len.*);
    if (sl >= @as(usize, @intCast(capacity))) return false;
    if (sortedContains(set[0..sl], element)) return false;

    const new_len = sortedInsert(set, sl, element);
    set_len.* = @intCast(new_len);
    return true;
}

pub fn setRemove(set: []Q16S, set_len: *i32, element: Q16S) bool {
    const sl: usize = @intCast(set_len.*);

    for (0..sl) |i| {
        if (Q16S.eql(set[i], element)) {
            for (i..sl - 1) |j| {
                set[j] = set[j + 1];
            }
            set_len.* -= 1;
            return true;
        }
    }

    return false;
}

pub fn setEqual(a: []const Q16S, b: []const Q16S) bool {
    if (a.len != b.len) return false;
    for (a, b) |va, vb| {
        if (!Q16S.eql(va, vb)) return false;
    }
    return true;
}

pub fn setPowerSet(set: []const Q16S, output: [][]Q16S) i32 {
    if (set.len > 20) return 0;

    const n_subsets: usize = @as(usize, 1) << @intCast(set.len);
    var count: usize = 0;

    for (0..n_subsets) |mask| {
        if (count >= output.len) break;

        var subset_len: usize = 0;
        for (0..set.len) |bit| {
            if (mask & (@as(usize, 1) << @intCast(bit)) != 0) {
                if (subset_len < output[count].len) {
                    output[count][subset_len] = set[bit];
                    subset_len += 1;
                }
            }
        }

        count += 1;
    }

    return @intCast(count);
}

pub fn setFromArray(data: []const Q16S, out: []Q16S) i32 {
    if (data.len == 0) return 0;

    var scratch: [4096]Q16S = undefined;
    const n = @min(data.len, scratch.len);
    @memcpy(scratch[0..n], data[0..n]);

    const coll = @import("collections.zig");
    coll.collSort(scratch[0..n]);

    var write: usize = 0;
    if (write < out.len) {
        out[0] = scratch[0];
        write = 1;
    }

    for (1..n) |i| {
        if (!Q16S.eql(scratch[i], scratch[i - 1])) {
            if (write >= out.len) break;
            out[write] = scratch[i];
            write += 1;
        }
    }

    return @intCast(write);
}
