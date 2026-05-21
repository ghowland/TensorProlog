// ============================================================
// src/builtins/mappings.zig
// ============================================================

const std = @import("std");
const q16_mod = @import("../vdr/q16.zig");
const types = @import("../vdr/types.zig");
const kb_types = @import("../kb/types.zig");
const dispatch_mod = @import("dispatch.zig");

const Q16 = q16_mod.Q16;
const VlpFact = kb_types.VlpFact;
const VlpFactTag = types.VlpFactTag;
const VlpStatus = types.VlpStatus;
const BuiltinArgs = dispatch_mod.BuiltinArgs;
const BuiltinResult = dispatch_mod.BuiltinResult;

pub const MapEntry = struct {
    key: i32,
    value: VlpFact,
    occupied: bool,
};

pub const VlpMap = struct {
    entries: []MapEntry,
    count: i32,
    capacity: i32,

    pub fn init(entries: []MapEntry) VlpMap {
        for (entries) |*e| {
            e.occupied = false;
        }
        return .{
            .entries = entries,
            .count = 0,
            .capacity = @intCast(entries.len),
        };
    }

    fn probe(self: *const VlpMap, key: i32) ?usize {
        const cap: usize = @intCast(self.capacity);
        if (cap == 0) return null;
        const hash: usize = @intCast(@as(u32, @bitCast(key)) % @as(u32, @intCast(cap)));
        var idx = hash;
        var probes: usize = 0;
        while (probes < cap) {
            if (!self.entries[idx].occupied) return null;
            if (self.entries[idx].key == key) return idx;
            idx = (idx + 1) % cap;
            probes += 1;
        }
        return null;
    }

    fn probeInsert(self: *VlpMap, key: i32) ?usize {
        const cap: usize = @intCast(self.capacity);
        if (cap == 0) return null;
        const hash: usize = @intCast(@as(u32, @bitCast(key)) % @as(u32, @intCast(cap)));
        var idx = hash;
        var probes: usize = 0;
        while (probes < cap) {
            if (!self.entries[idx].occupied) return idx;
            if (self.entries[idx].key == key) return idx;
            idx = (idx + 1) % cap;
            probes += 1;
        }
        return null;
    }
};

pub fn mapGet(map: *const VlpMap, key: i32) ?VlpFact {
    const slot = map.probe(key) orelse return null;
    return map.entries[slot].value;
}

pub fn mapSet(map: *VlpMap, key: i32, value: VlpFact) VlpStatus {
    const slot = map.probeInsert(key) orelse return .err_kb_full;
    const was_occupied = map.entries[slot].occupied;
    map.entries[slot] = .{ .key = key, .value = value, .occupied = true };
    if (!was_occupied) map.count += 1;
    return .ok;
}

pub fn mapDelete(map: *VlpMap, key: i32) bool {
    const slot = map.probe(key) orelse return false;
    map.entries[slot].occupied = false;
    map.count -= 1;
    const cap: usize = @intCast(map.capacity);
    var idx = (slot + 1) % cap;
    while (map.entries[idx].occupied) {
        const entry = map.entries[idx];
        map.entries[idx].occupied = false;
        map.count -= 1;
        _ = mapSet(map, entry.key, entry.value);
        idx = (idx + 1) % cap;
    }
    return true;
}

pub fn mapContainsKey(map: *const VlpMap, key: i32) bool {
    return map.probe(key) != null;
}

pub fn mapKeys(map: *const VlpMap, out: []i32) i32 {
    var count: usize = 0;
    const cap: usize = @intCast(map.capacity);
    for (0..cap) |i| {
        if (map.entries[i].occupied) {
            if (count >= out.len) break;
            out[count] = map.entries[i].key;
            count += 1;
        }
    }
    return @intCast(count);
}

pub fn mapValues(map: *const VlpMap, out: []VlpFact) i32 {
    var count: usize = 0;
    const cap: usize = @intCast(map.capacity);
    for (0..cap) |i| {
        if (map.entries[i].occupied) {
            if (count >= out.len) break;
            out[count] = map.entries[i].value;
            count += 1;
        }
    }
    return @intCast(count);
}

pub fn mapSize(map: *const VlpMap) i32 {
    return map.count;
}

pub fn mapMerge(a: *const VlpMap, b: *const VlpMap, out: *VlpMap, policy: types.VlpMergePolicy) VlpStatus {
    const cap_a: usize = @intCast(a.capacity);
    for (0..cap_a) |i| {
        if (a.entries[i].occupied) {
            _ = mapSet(out, a.entries[i].key, a.entries[i].value);
        }
    }
    const cap_b: usize = @intCast(b.capacity);
    for (0..cap_b) |i| {
        if (b.entries[i].occupied) {
            const existing = mapGet(out, b.entries[i].key);
            if (existing != null) {
                switch (policy) {
                    .ours => {},
                    .theirs => _ = mapSet(out, b.entries[i].key, b.entries[i].value),
                    .fail_on_conflict => return .err_kb_full,
                }
            } else {
                _ = mapSet(out, b.entries[i].key, b.entries[i].value);
            }
        }
    }
    return .ok;
}

pub fn mapFilterKeys(map: *const VlpMap, keep_keys: []const i32, out: *VlpMap) void {
    const cap: usize = @intCast(map.capacity);
    for (0..cap) |i| {
        if (map.entries[i].occupied) {
            for (keep_keys) |k| {
                if (map.entries[i].key == k) {
                    _ = mapSet(out, map.entries[i].key, map.entries[i].value);
                    break;
                }
            }
        }
    }
}

pub fn mapFilterValues(map: *const VlpMap, pred: []const bool, out: *VlpMap) void {
    var vi: usize = 0;
    const cap: usize = @intCast(map.capacity);
    for (0..cap) |i| {
        if (map.entries[i].occupied) {
            if (vi < pred.len and pred[vi]) {
                _ = mapSet(out, map.entries[i].key, map.entries[i].value);
            }
            vi += 1;
        }
    }
}

pub fn mapMapValues(map: *const VlpMap, scale: Q16, out: *VlpMap) void {
    const cap: usize = @intCast(map.capacity);
    for (0..cap) |i| {
        if (map.entries[i].occupied) {
            var new_val = map.entries[i].value;
            new_val.value = Q16.mul(new_val.value, scale);
            _ = mapSet(out, map.entries[i].key, new_val);
        }
    }
}

pub fn mapInvert(map: *const VlpMap, out: *VlpMap) VlpStatus {
    const cap: usize = @intCast(map.capacity);
    for (0..cap) |i| {
        if (map.entries[i].occupied) {
            const new_key = map.entries[i].value.value.v;
            var new_val = map.entries[i].value;
            new_val.value = .{ .v = map.entries[i].key, .r0 = 0 };
            const status = mapSet(out, new_key, new_val);
            if (status != .ok) return status;
        }
    }
    return .ok;
}

pub fn mapClear(map: *VlpMap) void {
    const cap: usize = @intCast(map.capacity);
    for (0..cap) |i| {
        map.entries[i].occupied = false;
    }
    map.count = 0;
}

pub fn mapEqual(a: *const VlpMap, b: *const VlpMap) bool {
    if (a.count != b.count) return false;
    const cap: usize = @intCast(a.capacity);
    for (0..cap) |i| {
        if (a.entries[i].occupied) {
            const bv = mapGet(b, a.entries[i].key) orelse return false;
            if (!Q16.eql(a.entries[i].value.value, bv.value)) return false;
        }
    }
    return true;
}

pub fn mapFromArrays(keys: []const i32, values: []const VlpFact, out: *VlpMap) VlpStatus {
    const n = @min(keys.len, values.len);
    for (0..n) |i| {
        const status = mapSet(out, keys[i], values[i]);
        if (status != .ok) return status;
    }
    return .ok;
}

pub fn builtinMapGet(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapSet(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapDelete(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapContainsKey(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapKeys(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapValues(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapSize(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapMerge(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapFilterKeys(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapFilterValues(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapMapValues(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapInvert(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapClear(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapEqual(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinMapFromArrays(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
