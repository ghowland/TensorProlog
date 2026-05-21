// ============================================================
// src/primitives/lru.zig
// ============================================================

const prim_types = @import("types.zig");
const VlpFact = prim_types.VlpFact;

pub const LRUEntry = struct {
    key: i32 = -1,
    value: VlpFact = .{},
    prev: i32 = -1,
    next: i32 = -1,
    occupied: bool = false,
};

pub const LRU = struct {
    entries: []LRUEntry,
    capacity: i32,
    count: i32,
    head: i32,
    tail: i32,
    index: []i32,
    index_cap: i32,

    pub fn init(entries: []LRUEntry, index: []i32) LRU {
        for (entries) |*e| e.* = LRUEntry{};
        @memset(index, -1);
        return .{
            .entries = entries,
            .capacity = @intCast(entries.len),
            .count = 0,
            .head = -1,
            .tail = -1,
            .index = index,
            .index_cap = @intCast(index.len),
        };
    }

    pub fn get(self: *LRU, key: i32) ?VlpFact {
        const idx = self.findEntry(key);
        if (idx < 0) return null;
        self.moveToHead(idx);
        return self.entries[@intCast(idx)].value;
    }

    pub fn put(self: *LRU, key: i32, value: VlpFact) void {
        const existing = self.findEntry(key);
        if (existing >= 0) {
            self.entries[@intCast(existing)].value = value;
            self.moveToHead(existing);
            return;
        }

        if (self.count >= self.capacity) {
            _ = self.evictTail();
        }

        const slot = self.findFreeSlot();
        if (slot < 0) return;

        self.entries[@intCast(slot)] = .{
            .key = key,
            .value = value,
            .prev = -1,
            .next = -1,
            .occupied = true,
        };
        self.indexInsert(key, slot);
        self.pushHead(slot);
        self.count += 1;
    }

    pub fn evict(self: *LRU) ?VlpFact {
        return self.evictTail();
    }

    pub fn size(self: *const LRU) i32 {
        return self.count;
    }

    pub fn clear(self: *LRU) void {
        for (self.entries) |*e| e.* = LRUEntry{};
        @memset(self.index, -1);
        self.count = 0;
        self.head = -1;
        self.tail = -1;
    }

    pub fn contains(self: *const LRU, key: i32) bool {
        return self.findEntryConst(key) >= 0;
    }

    fn findEntry(self: *const LRU, key: i32) i32 {
        return self.findEntryConst(key);
    }

    fn findEntryConst(self: *const LRU, key: i32) i32 {
        const h = indexHash(key, self.index_cap);
        var probe: i32 = 0;
        while (probe < self.index_cap) : (probe += 1) {
            const idx = @mod(h + probe, self.index_cap);
            const slot = self.index[@intCast(idx)];
            if (slot < 0) return -1;
            if (self.entries[@intCast(slot)].occupied and self.entries[@intCast(slot)].key == key) return slot;
        }
        return -1;
    }

    fn findFreeSlot(self: *const LRU) i32 {
        var i: i32 = 0;
        while (i < self.capacity) : (i += 1) {
            if (!self.entries[@intCast(i)].occupied) return i;
        }
        return -1;
    }

    fn evictTail(self: *LRU) ?VlpFact {
        if (self.tail < 0) return null;
        const t = self.tail;
        const entry = &self.entries[@intCast(t)];
        const val = entry.value;
        const key = entry.key;

        self.removeFromList(t);
        self.indexRemove(key);
        entry.* = LRUEntry{};
        self.count -= 1;
        return val;
    }

    fn moveToHead(self: *LRU, idx: i32) void {
        if (idx == self.head) return;
        self.removeFromList(idx);
        self.pushHead(idx);
    }

    fn pushHead(self: *LRU, idx: i32) void {
        self.entries[@intCast(idx)].prev = -1;
        self.entries[@intCast(idx)].next = self.head;
        if (self.head >= 0) {
            self.entries[@intCast(self.head)].prev = idx;
        }
        self.head = idx;
        if (self.tail < 0) self.tail = idx;
    }

    fn removeFromList(self: *LRU, idx: i32) void {
        const e = &self.entries[@intCast(idx)];
        if (e.prev >= 0) {
            self.entries[@intCast(e.prev)].next = e.next;
        } else {
            self.head = e.next;
        }
        if (e.next >= 0) {
            self.entries[@intCast(e.next)].prev = e.prev;
        } else {
            self.tail = e.prev;
        }
        e.prev = -1;
        e.next = -1;
    }

    fn indexInsert(self: *LRU, key: i32, slot: i32) void {
        const h = indexHash(key, self.index_cap);
        var probe: i32 = 0;
        while (probe < self.index_cap) : (probe += 1) {
            const idx = @mod(h + probe, self.index_cap);
            const i: usize = @intCast(idx);
            if (self.index[i] < 0) {
                self.index[i] = slot;
                return;
            }
        }
    }

    fn indexRemove(self: *LRU, key: i32) void {
        const h = indexHash(key, self.index_cap);
        var probe: i32 = 0;
        while (probe < self.index_cap) : (probe += 1) {
            const idx = @mod(h + probe, self.index_cap);
            const i: usize = @intCast(idx);
            const slot = self.index[i];
            if (slot < 0) return;
            if (self.entries[@intCast(slot)].key == key) {
                self.index[i] = -1;
                rehashFrom(self, idx);
                return;
            }
        }
    }

    fn rehashFrom(self: *LRU, start: i32) void {
        var idx = @mod(start + 1, self.index_cap);
        while (true) {
            const i: usize = @intCast(idx);
            if (self.index[i] < 0) break;
            const slot = self.index[i];
            self.index[i] = -1;
            const key = self.entries[@intCast(slot)].key;
            const h = indexHash(key, self.index_cap);
            var p: i32 = 0;
            while (p < self.index_cap) : (p += 1) {
                const ni = @mod(h + p, self.index_cap);
                const nii: usize = @intCast(ni);
                if (self.index[nii] < 0) {
                    self.index[nii] = slot;
                    break;
                }
            }
            idx = @mod(idx + 1, self.index_cap);
        }
    }
};

fn indexHash(key: i32, cap: i32) i32 {
    const k: u32 = @bitCast(key);
    const h: u32 = k *% 2654435761;
    return @intCast(@mod(@as(i32, @bitCast(h)), cap) + (if (@as(i32, @bitCast(h)) < 0) @as(i32, cap) else @as(i32, 0)));
}
