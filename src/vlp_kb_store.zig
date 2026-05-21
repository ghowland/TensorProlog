// ============================================================
// vlp_kb_store.zig
// KB store engine — host manages structure, GPU does bulk ops.
// Path index, COW, text store all host-side.
// Fact scans dispatch to GPU when above threshold.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const gpu = @import("vlp_gpu_params.zig");

// ============================================================
// Configuration
// ============================================================

pub const KbCreateConfig = struct {
    name: []const u8,
    path: []const u8,
    parent_id: i32,
    max_facts: i32,
    max_rules: i32,
    visibility: i8,
    owner: []const u8,
};

pub const ScopedSearchConfig = struct {
    start_kb_id: i32,
    tag: types.FactTag,
    max_depth: i32 = 100,
    max_results: i32 = 100,
};

// ============================================================
// Search / scan results
// ============================================================

pub const SearchResult = struct {
    facts: []types.Fact,
    kb_ids: []i32,
    slot_ids: []i32,
    count: i32,

    pub fn empty() SearchResult {
        return .{ .facts = &.{}, .kb_ids = &.{}, .slot_ids = &.{}, .count = 0 };
    }
};

// ============================================================
// Path index — host-side hash map: dotted path → kb_id
// ============================================================

const PATH_INDEX_LOAD_FACTOR_MAX: i32 = 70; // percent

pub const PathEntry = struct {
    path_hash: u32,
    kb_id: i32,
    occupied: bool,
};

pub const PathIndex = struct {
    entries: []PathEntry,
    capacity: i32,
    count: i32,

    pub fn init(allocator: std.mem.Allocator, capacity: i32) PathIndex {
        const cap: usize = @intCast(capacity);
        const entries = allocator.alloc(PathEntry, cap) catch return .{
            .entries = &.{},
            .capacity = 0,
            .count = 0,
        };
        for (entries) |*e| {
            e.* = .{ .path_hash = 0, .kb_id = -1, .occupied = false };
        }
        return .{ .entries = entries, .capacity = capacity, .count = entries.len };
    }

    pub fn deinit(self: *PathIndex, allocator: std.mem.Allocator) void {
        if (self.entries.len > 0) allocator.free(self.entries);
        self.entries = &.{};
        self.count = 0;
    }

    pub fn resolve(self: *PathIndex, path: []const u8) ?i32 {
        if (self.capacity == 0) return null;
        const h = hashPath(path);
        var idx: u32 = h % @as(u32, @intCast(self.capacity));
        var probes: i32 = 0;
        while (probes < self.capacity) : (probes += 1) {
            const e = &self.entries[@intCast(idx)];
            if (!e.occupied) return null;
            if (e.path_hash == h) return e.kb_id;
            idx = (idx + 1) % @as(u32, @intCast(self.capacity));
        }
        return null;
    }

    pub fn register(self: *PathIndex, path: []const u8, kb_id: i32) bool {
        if (self.capacity == 0) return false;
        if (self.count * 100 >= self.capacity * PATH_INDEX_LOAD_FACTOR_MAX) return false;
        const h = hashPath(path);
        var idx: u32 = h % @as(u32, @intCast(self.capacity));
        var probes: i32 = 0;
        while (probes < self.capacity) : (probes += 1) {
            const e = &self.entries[@intCast(idx)];
            if (!e.occupied) {
                e.path_hash = h;
                e.kb_id = kb_id;
                e.occupied = true;
                self.count += 1;
                return true;
            }
            if (e.path_hash == h) {
                // Update existing
                e.kb_id = kb_id;
                return true;
            }
            idx = (idx + 1) % @as(u32, @intCast(self.capacity));
        }
        return false;
    }

    pub fn remove(self: *PathIndex, path: []const u8) bool {
        if (self.capacity == 0) return false;
        const h = hashPath(path);
        var idx: u32 = h % @as(u32, @intCast(self.capacity));
        var probes: i32 = 0;
        while (probes < self.capacity) : (probes += 1) {
            const e = &self.entries[@intCast(idx)];
            if (!e.occupied) return false;
            if (e.path_hash == h) {
                e.occupied = false;
                e.kb_id = -1;
                self.count -= 1;
                // Rehash subsequent entries (linear probe deletion)
                rehashFrom(self, (idx + 1) % @as(u32, @intCast(self.capacity)));
                return true;
            }
            idx = (idx + 1) % @as(u32, @intCast(self.capacity));
        }
        return false;
    }

    fn rehashFrom(self: *PathIndex, start: u32) void {
        var idx = start;
        while (true) {
            const e = &self.entries[@intCast(idx)];
            if (!e.occupied) break;
            const saved_hash = e.path_hash;
            const saved_id = e.kb_id;
            e.occupied = false;
            self.count -= 1;
            // Re-insert
            _ = self.register(&.{}, saved_id); // hash already known
            // Shortcut: direct insert using saved_hash
            var ins_idx: u32 = saved_hash % @as(u32, @intCast(self.capacity));
            while (self.entries[@intCast(ins_idx)].occupied) {
                ins_idx = (ins_idx + 1) % @as(u32, @intCast(self.capacity));
            }
            self.entries[@intCast(ins_idx)] = .{
                .path_hash = saved_hash,
                .kb_id = saved_id,
                .occupied = true,
            };
            self.count += 1;
            idx = (idx + 1) % @as(u32, @intCast(self.capacity));
        }
    }
};

fn hashPath(path: []const u8) u32 {
    // FNV-1a 32-bit
    var h: u32 = 2166136261;
    for (path) |b| {
        h ^= @as(u32, b);
        h *%= 16777619;
    }
    return h;
}

// ============================================================
// COW page table — host-side, per clone session
// ============================================================

pub const COW_PAGE_SIZE: i32 = 4096; // bytes — ~100 facts per page

pub const CowPageTable = struct {
    parent_session_id: i32,
    clone_session_id: i32,
    n_pages: i32,
    dirty_bits: []u8, // one bit per page, packed
    private_base_offset: i64, // clone's private region in fact store
    parent_base_offset: i64, // parent's region

    pub fn init(allocator: std.mem.Allocator, parent_session: i32, clone_session: i32, region_size: i64, parent_offset: i64, private_offset: i64) CowPageTable {
        const pages: i32 = @intCast(@divTrunc(region_size + COW_PAGE_SIZE - 1, COW_PAGE_SIZE));
        const dirty_bytes: usize = @intCast(@divTrunc(pages + 7, 8));
        const bits = allocator.alloc(u8, dirty_bytes) catch return .{
            .parent_session_id = parent_session,
            .clone_session_id = clone_session,
            .n_pages = 0,
            .dirty_bits = &.{},
            .private_base_offset = private_offset,
            .parent_base_offset = parent_offset,
        };
        @memset(bits, 0);
        return .{
            .parent_session_id = parent_session,
            .clone_session_id = clone_session,
            .n_pages = pages,
            .dirty_bits = bits,
            .private_base_offset = private_offset,
            .parent_base_offset = parent_offset,
        };
    }

    pub fn deinit(self: *CowPageTable, allocator: std.mem.Allocator) void {
        if (self.dirty_bits.len > 0) allocator.free(self.dirty_bits);
        self.dirty_bits = &.{};
        self.n_pages = 0;
    }

    pub fn isDirty(self: *CowPageTable, page: i32) bool {
        if (page < 0 or page >= self.n_pages) return false;
        const byte_idx: usize = @intCast(@divTrunc(page, 8));
        const bit_idx: u3 = @intCast(@mod(page, 8));
        return (self.dirty_bits[byte_idx] & (@as(u8, 1) << bit_idx)) != 0;
    }

    pub fn markDirty(self: *CowPageTable, page: i32) void {
        if (page < 0 or page >= self.n_pages) return;
        const byte_idx: usize = @intCast(@divTrunc(page, 8));
        const bit_idx: u3 = @intCast(@mod(page, 8));
        self.dirty_bits[byte_idx] |= @as(u8, 1) << bit_idx;
    }

    pub fn dirtyPageCount(self: *CowPageTable) i32 {
        var count: i32 = 0;
        var page: i32 = 0;
        while (page < self.n_pages) : (page += 1) {
            if (self.isDirty(page)) count += 1;
        }
        return count;
    }

    pub fn pageOffsetInParent(self: *CowPageTable, page: i32) i64 {
        return self.parent_base_offset + @as(i64, page) * COW_PAGE_SIZE;
    }

    pub fn pageOffsetInPrivate(self: *CowPageTable, page: i32) i64 {
        return self.private_base_offset + @as(i64, page) * COW_PAGE_SIZE;
    }

    pub fn pageForFactOffset(self: *CowPageTable, fact_offset: i64) i32 {
        _ = self;
        return @intCast(@divTrunc(fact_offset, COW_PAGE_SIZE));
    }
};

// ============================================================
// KB Store
// ============================================================

pub const KbStore = struct {
    bridge: *bridge_mod.Bridge,
    allocator: std.mem.Allocator,
    path_index: PathIndex,

    // Allocation cursors
    next_kb_id: i32,
    next_fact_offset: i64,
    next_rule_offset: i32,
    next_term_offset: i64,
    next_text_offset: i64,
    next_grammar_offset: i32,

    // Host-side KB cache — frequently accessed KBs kept in host memory
    // to avoid GPU readback for structural operations
    kb_cache: []types.Kb,
    kb_cache_valid: []bool,
    kb_cache_capacity: i32,

    // COW tables — one per active clone
    cow_tables: std.ArrayList(CowPageTable),

    // Scratch buffers for search results (reused across calls)
    search_fact_buf: []types.Fact,
    search_kb_id_buf: []i32,
    search_slot_id_buf: []i32,
    search_buf_capacity: i32,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(bridge: *bridge_mod.Bridge, allocator: std.mem.Allocator, max_kbs: i32) KbStore {
    const path_cap = max_kbs * 2; // 50% load factor headroom
    const cache_cap: usize = @intCast(@min(max_kbs, 1024)); // cache up to 1024 KBs
    const search_cap: i32 = 1024;

    const kb_cache = allocator.alloc(types.Kb, cache_cap) catch &.{};
    const kb_valid = allocator.alloc(bool, cache_cap) catch &.{};
    if (kb_valid.len > 0) @memset(kb_valid, false);

    const s_facts = allocator.alloc(types.Fact, @intCast(search_cap)) catch &.{};
    const s_kbs = allocator.alloc(i32, @intCast(search_cap)) catch &.{};
    const s_slots = allocator.alloc(i32, @intCast(search_cap)) catch &.{};

    return .{
        .bridge = bridge,
        .allocator = allocator,
        .path_index = PathIndex.init(allocator, path_cap),
        .next_kb_id = 0,
        .next_fact_offset = 0,
        .next_rule_offset = 0,
        .next_term_offset = 0,
        .next_text_offset = 0,
        .next_grammar_offset = 0,
        .kb_cache = kb_cache,
        .kb_cache_valid = kb_valid,
        .kb_cache_capacity = @intCast(cache_cap),
        .cow_tables = std.ArrayList(CowPageTable).init(allocator),
        .search_fact_buf = s_facts,
        .search_kb_id_buf = s_kbs,
        .search_slot_id_buf = s_slots,
        .search_buf_capacity = search_cap,
    };
}

pub fn deinit(self: *KbStore) void {
    for (self.cow_tables.items) |*cow| {
        cow.deinit(self.allocator);
    }
    self.cow_tables.deinit();
    self.path_index.deinit(self.allocator);
    if (self.kb_cache.len > 0) self.allocator.free(self.kb_cache);
    if (self.kb_cache_valid.len > 0) self.allocator.free(self.kb_cache_valid);
    if (self.search_fact_buf.len > 0) self.allocator.free(self.search_fact_buf);
    if (self.search_kb_id_buf.len > 0) self.allocator.free(self.search_kb_id_buf);
    if (self.search_slot_id_buf.len > 0) self.allocator.free(self.search_slot_id_buf);
}

// ============================================================
// KB CRUD — host-side structural operations
// ============================================================

pub fn createKb(self: *KbStore, config: *const KbCreateConfig) i32 {
    const kb_id = self.next_kb_id;
    self.next_kb_id += 1;

    // Store name and path in text store
    const name_off = self.textAppend(config.name);
    const path_off = self.textAppend(config.path);
    const owner_off = self.textAppend(config.owner);

    // Allocate fact region
    const facts_off = self.next_fact_offset;
    self.next_fact_offset += @as(i64, config.max_facts);

    // Allocate rule region
    const rules_off = self.next_rule_offset;
    self.next_rule_offset += config.max_rules;

    // Build KB struct
    var kb = std.mem.zeroes(types.Kb);
    kb.name_offset = name_off;
    kb.name_length = @intCast(@min(config.name.len, 32767));
    kb.path_offset = path_off;
    kb.path_length = @intCast(@min(config.path.len, 32767));
    kb.id = kb_id;
    kb.facts_offset = @intCast(facts_off);
    kb.facts_count = 0;
    kb.facts_capacity = config.max_facts;
    kb.rules_offset = rules_off;
    kb.rules_count = 0;
    kb.rules_capacity = config.max_rules;
    kb.parent_id = config.parent_id;
    kb.visibility = config.visibility;
    kb.frozen = 0;
    kb.owner_offset = owner_off;
    kb.owner_length = @intCast(@min(config.owner.len, 32767));
    kb.created_at = currentTimestamp();
    kb.last_modified = kb.created_at;

    // Initialize all remaining offsets to -1
    kb.constraints_offset = -1;
    kb.connections_offset = -1;
    kb.grammars_offset = -1;
    kb.iose_offset = -1;
    kb.working_data_offset = -1;
    kb.lru_table_offset = -1;
    kb.counter_table_offset = -1;
    kb.lock_table_offset = -1;
    kb.queue_table_offset = -1;
    kb.stack_table_offset = -1;
    kb.ring_table_offset = -1;
    kb.bitset_table_offset = -1;
    kb.children_offset = -1;
    kb.mounts_offset = -1;

    // Write KB struct to device
    const kb_device_offset = @as(i64, kb_id) * types.KB_STRUCT_SIZE;
    const kb_bytes: []const u8 = @as([*]const u8, @ptrCast(&kb))[0..@sizeOf(types.Kb)];
    _ = self.bridge.uploadToBuffer(.kb_store, kb_device_offset, kb_bytes);

    // Initialize fact slots to empty on device
    var empty_fact = types.Fact.empty();
    const empty_bytes: []const u8 = @as([*]const u8, @ptrCast(&empty_fact))[0..@sizeOf(types.Fact)];
    // Batch fill — use buffer fill for the region
    _ = self.bridge.fillBuffer(.fact_store, facts_off * @sizeOf(types.Fact), @as(i64, config.max_facts) * @sizeOf(types.Fact), 0);
    _ = empty_bytes; // fact fill handled by fillBuffer with 0 (tag=0=value, but we want 255=empty)
    // TODO: dispatch a fill kernel that writes FactTag.empty to each slot

    // Update host cache
    self.cacheKb(kb_id, &kb);

    // Register path
    _ = self.path_index.register(config.path, kb_id);

    // Add as child of parent
    if (config.parent_id >= 0) {
        _ = self.addChild(config.parent_id, kb_id);
    }

    return kb_id;
}

pub fn getKb(self: *KbStore, kb_id: i32) ?types.Kb {
    // Check host cache first
    if (kb_id >= 0 and kb_id < self.kb_cache_capacity) {
        const idx: usize = @intCast(kb_id);
        if (idx < self.kb_cache_valid.len and self.kb_cache_valid[idx]) {
            return self.kb_cache[idx];
        }
    }
    // Read from device
    var kb: types.Kb = undefined;
    const offset = @as(i64, kb_id) * types.KB_STRUCT_SIZE;
    const dest: []u8 = @as([*]u8, @ptrCast(&kb))[0..@sizeOf(types.Kb)];
    const status = self.bridge.downloadFromBuffer(.kb_store, offset, dest);
    if (status.isErr()) return null;
    if (kb.id != kb_id) return null; // invalid / uninitialized
    self.cacheKb(kb_id, &kb);
    return kb;
}

pub fn freezeKb(self: *KbStore, kb_id: i32) types.Status {
    var kb = self.getKb(kb_id) orelse return types.Status.err(.kb, .kb_not_found, kb_id);
    kb.frozen = 1;
    kb.last_modified = currentTimestamp();
    return self.writeKbToDevice(kb_id, &kb);
}

pub fn setVisibility(self: *KbStore, kb_id: i32, visibility: i8) types.Status {
    var kb = self.getKb(kb_id) orelse return types.Status.err(.kb, .kb_not_found, kb_id);
    kb.visibility = visibility;
    kb.last_modified = currentTimestamp();
    return self.writeKbToDevice(kb_id, &kb);
}

// ============================================================
// Path resolution — host-side
// ============================================================

pub fn pathResolve(self: *KbStore, path: []const u8) ?i32 {
    return self.path_index.resolve(path);
}

pub fn pathRegister(self: *KbStore, path: []const u8, kb_id: i32) types.Status {
    if (self.path_index.register(path, kb_id)) return types.Status.ok();
    return types.Status.err(.kb, .kb_full, 0);
}

pub fn pathRemove(self: *KbStore, path: []const u8) types.Status {
    if (self.path_index.remove(path)) return types.Status.ok();
    return types.Status.err(.kb, .kb_not_found, 0);
}

// ============================================================
// Fact operations — single item (host path for small ops)
// ============================================================

pub fn factWrite(self: *KbStore, kb_id: i32, slot_id: i32, fact: *const types.Fact) types.Status {
    var kb = self.getKb(kb_id) orelse return types.Status.err(.kb, .kb_not_found, kb_id);
    if (kb.isFrozen()) return types.Status.err(.kb, .kb_frozen, kb_id);
    if (slot_id < 0 or slot_id >= kb.facts_capacity) return types.Status.err(.kb, .slot_out_of_range, slot_id);

    // COW check — if this is a clone, handle page copy
    if (self.findCow(kb_id)) |cow| {
        const page = cow.pageForFactOffset(@as(i64, kb.facts_offset + slot_id) * @sizeOf(types.Fact));
        if (!cow.isDirty(page)) {
            _ = self.cowCopyPage(cow, page);
        }
    }

    const offset = (@as(i64, kb.facts_offset) + @as(i64, slot_id)) * @sizeOf(types.Fact);
    const bytes: []const u8 = @as([*]const u8, @ptrCast(fact))[0..@sizeOf(types.Fact)];
    const status = self.bridge.uploadToBuffer(.fact_store, offset, bytes);
    if (status.isErr()) return status;

    // Update count if this is a new slot
    if (slot_id >= kb.facts_count) {
        kb.facts_count = slot_id + 1;
    }
    kb.last_modified = currentTimestamp();
    return self.writeKbToDevice(kb_id, &kb);
}

pub fn factRead(self: *KbStore, kb_id: i32, slot_id: i32) ?types.Fact {
    const kb = self.getKb(kb_id) orelse return null;
    if (slot_id < 0 or slot_id >= kb.facts_count) return null;

    var fact: types.Fact = undefined;
    const offset = (@as(i64, kb.facts_offset) + @as(i64, slot_id)) * @sizeOf(types.Fact);
    const dest: []u8 = @as([*]u8, @ptrCast(&fact))[0..@sizeOf(types.Fact)];
    const status = self.bridge.downloadFromBuffer(.fact_store, offset, dest);
    if (status.isErr()) return null;
    if (fact.isEmpty()) return null;
    return fact;
}

pub fn factRetract(self: *KbStore, kb_id: i32, slot_id: i32) types.Status {
    var empty = types.Fact.empty();
    return self.factWrite(kb_id, slot_id, &empty);
}

// ============================================================
// Batch fact operations — GPU for large batches, host for small
// ============================================================

pub fn factWriteBatch(self: *KbStore, kb_id: i32, slot_ids: []const i32, facts: []const types.Fact) types.Status {
    if (slot_ids.len != facts.len) return types.Status.err(.kb, .slot_out_of_range, -1);
    const n: i32 = @intCast(slot_ids.len);

    if (!self.bridge.shouldUseGpu(.fact_scan, n)) {
        // Host path — write one by one
        for (slot_ids, facts) |sid, *f| {
            const s = self.factWrite(kb_id, sid, f);
            if (s.isErr()) return s;
        }
        return types.Status.ok();
    }

    // GPU path
    const kb = self.getKb(kb_id) orelse return types.Status.err(.kb, .kb_not_found, kb_id);

    // Upload facts to scratch_a
    const fact_bytes: []const u8 = @as([*]const u8, @ptrCast(facts.ptr))[0 .. facts.len * @sizeOf(types.Fact)];
    var status = self.bridge.uploadToBuffer(.scratch_a, 0, fact_bytes);
    if (status.isErr()) return status;

    // Upload slot_ids to scratch_b
    const slot_bytes: []const u8 = @as([*]const u8, @ptrCast(slot_ids.ptr))[0 .. slot_ids.len * 4];
    status = self.bridge.uploadToBuffer(.scratch_b, 0, slot_bytes);
    if (status.isErr()) return status;

    // Dispatch
    var params = gpu.FactWriteBatchParams{
        .n_facts = n,
        .base_offset = kb.facts_offset,
        .fact_store_capacity = @intCast(self.bridge.layout.fact_capacity),
    };

    _ = self.bridge.resetStatusBuffer();
    status = self.bridge.dispatch(&.{
        .pipeline = .fact_write_batch,
        .group_count_x = @divTrunc(n + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&params),
        .params_size = @sizeOf(gpu.FactWriteBatchParams),
    });

    return status;
}

pub fn factReadBatch(self: *KbStore, kb_id: i32, slot_ids: []const i32, out: []types.Fact) types.Status {
    if (slot_ids.len != out.len) return types.Status.err(.kb, .slot_out_of_range, -1);
    const n: i32 = @intCast(slot_ids.len);

    if (!self.bridge.shouldUseGpu(.fact_scan, n)) {
        for (slot_ids, 0..) |sid, i| {
            out[i] = self.factRead(kb_id, sid) orelse types.Fact.empty();
        }
        return types.Status.ok();
    }

    const kb = self.getKb(kb_id) orelse return types.Status.err(.kb, .kb_not_found, kb_id);

    // Upload absolute offsets to scratch_a
    var offsets = self.allocator.alloc(i32, @intCast(n)) catch return types.Status.err(.device, .device_out_of_memory, 0);
    defer self.allocator.free(offsets);
    for (slot_ids, 0..) |sid, i| {
        offsets[i] = kb.facts_offset + sid;
    }

    const off_bytes: []const u8 = @as([*]const u8, @ptrCast(offsets.ptr))[0 .. offsets.len * 4];
    var status = self.bridge.uploadToBuffer(.scratch_a, 0, off_bytes);
    if (status.isErr()) return status;

    var params = gpu.FactReadBatchParams{ .n_reads = n };
    status = self.bridge.dispatch(&.{
        .pipeline = .fact_read_batch,
        .group_count_x = @divTrunc(n + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&params),
        .params_size = @sizeOf(gpu.FactReadBatchParams),
    });
    if (status.isErr()) return status;

    // Download results from scratch_b
    return self.bridge.readScratchSlice(types.Fact, .scratch_b, 0, n, out);
}

// ============================================================
// Scan / Search — GPU for large, host for small
// ============================================================

pub fn factScanByTag(self: *KbStore, kb_id: i32, tag: types.FactTag, max_results: i32) SearchResult {
    const kb = self.getKb(kb_id) orelse return SearchResult.empty();
    if (kb.facts_count == 0) return SearchResult.empty();

    const cap = @min(max_results, self.search_buf_capacity);

    if (!self.bridge.shouldUseGpu(.fact_scan, kb.facts_count)) {
        // Host path — read facts one by one
        var count: i32 = 0;
        var slot: i32 = 0;
        while (slot < kb.facts_count and count < cap) : (slot += 1) {
            if (self.factRead(kb_id, slot)) |fact| {
                if (fact.tag == tag) {
                    self.search_fact_buf[@intCast(count)] = fact;
                    self.search_kb_id_buf[@intCast(count)] = kb_id;
                    self.search_slot_id_buf[@intCast(count)] = slot;
                    count += 1;
                }
            }
        }
        return .{
            .facts = self.search_fact_buf[0..@intCast(count)],
            .kb_ids = self.search_kb_id_buf[0..@intCast(count)],
            .slot_ids = self.search_slot_id_buf[0..@intCast(count)],
            .count = count,
        };
    }

    // GPU path
    _ = self.bridge.resetResultCounts();

    var params = gpu.FactScanByTagParams{
        .base_offset = kb.facts_offset,
        .scan_length = kb.facts_count,
        .target_tag = @intFromEnum(tag),
        .max_results = cap,
    };

    const status = self.bridge.dispatch(&.{
        .pipeline = .fact_scan_by_tag,
        .group_count_x = @divTrunc(kb.facts_count + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&params),
        .params_size = @sizeOf(gpu.FactScanByTagParams),
    });
    if (status.isErr()) return SearchResult.empty();

    const count = self.bridge.readResultCount(0);
    if (count <= 0) return SearchResult.empty();

    const actual = @min(count, cap);

    // Read matching slot indices from scratch_a
    _ = self.bridge.readScratchSlice(i32, .scratch_a, 0, actual, self.search_slot_id_buf[0..@intCast(actual)]);

    // Read the actual facts
    for (self.search_slot_id_buf[0..@intCast(actual)], 0..) |sid, i| {
        self.search_fact_buf[i] = self.factRead(kb_id, sid) orelse types.Fact.empty();
        self.search_kb_id_buf[i] = kb_id;
    }

    return .{
        .facts = self.search_fact_buf[0..@intCast(actual)],
        .kb_ids = self.search_kb_id_buf[0..@intCast(actual)],
        .slot_ids = self.search_slot_id_buf[0..@intCast(actual)],
        .count = actual,
    };
}

pub fn scopedSearch(self: *KbStore, config: *const ScopedSearchConfig) SearchResult {
    // Build chain: walk from start_kb_id up parent chain, host-side
    var chain: [128]ChainEntry = undefined;
    var chain_len: i32 = 0;
    var current_id = config.start_kb_id;
    var depth: i32 = 0;

    while (current_id >= 0 and depth < config.max_depth and chain_len < 128) {
        if (self.getKb(current_id)) |kb| {
            chain[@intCast(chain_len)] = .{
                .kb_id = current_id,
                .facts_offset = kb.facts_offset,
                .facts_count = kb.facts_count,
            };
            chain_len += 1;
            current_id = kb.parent_id;
        } else break;
        depth += 1;
    }

    if (chain_len == 0) return SearchResult.empty();

    // For small total fact count, scan on host
    var total_facts: i32 = 0;
    for (chain[0..@intCast(chain_len)]) |entry| {
        total_facts += entry.facts_count;
    }

    // Search each KB in chain order (deepest first = lexical scoping)
    var result_count: i32 = 0;
    const cap = @min(config.max_results, self.search_buf_capacity);

    for (chain[0..@intCast(chain_len)]) |entry| {
        if (result_count >= cap) break;
        const sub = self.factScanByTag(entry.kb_id, config.tag, cap - result_count);
        for (0..@intCast(sub.count)) |i| {
            if (result_count >= cap) break;
            const ri: usize = @intCast(result_count);
            self.search_fact_buf[ri] = sub.facts[i];
            self.search_kb_id_buf[ri] = sub.kb_ids[i];
            self.search_slot_id_buf[ri] = sub.slot_ids[i];
            result_count += 1;
        }
    }

    return .{
        .facts = self.search_fact_buf[0..@intCast(result_count)],
        .kb_ids = self.search_kb_id_buf[0..@intCast(result_count)],
        .slot_ids = self.search_slot_id_buf[0..@intCast(result_count)],
        .count = result_count,
    };
}

pub const ChainEntry = struct {
    kb_id: i32,
    facts_offset: i32,
    facts_count: i32,
};

pub fn buildChain(self: *KbStore, start_kb_id: i32, max_depth: i32) []ChainEntry {
    // Exposed for Prolog engine use
    var chain = self.allocator.alloc(ChainEntry, @intCast(@min(max_depth, 128))) catch return &.{};
    var len: usize = 0;
    var current = start_kb_id;
    var depth: i32 = 0;
    while (current >= 0 and depth < max_depth and len < chain.len) {
        if (self.getKb(current)) |kb| {
            chain[len] = .{
                .kb_id = current,
                .facts_offset = kb.facts_offset,
                .facts_count = kb.facts_count,
            };
            len += 1;
            current = kb.parent_id;
        } else break;
        depth += 1;
    }
    return chain[0..len];
}

// ============================================================
// Text store — host-side append-only
// ============================================================

pub fn textAppend(self: *KbStore, data: []const u8) i32 {
    const offset: i32 = @intCast(self.next_text_offset);
    _ = self.bridge.uploadToBuffer(.text_store, self.next_text_offset, data);
    self.next_text_offset += @intCast(data.len);
    return offset;
}

pub fn textRead(self: *KbStore, offset: i32, length: i16, buf: []u8) types.Status {
    const len: usize = @intCast(@min(length, @as(i16, @intCast(buf.len))));
    return self.bridge.downloadFromBuffer(.text_store, @as(i64, offset), buf[0..len]);
}

// ============================================================
// Children / Mounts — host-side
// ============================================================

pub fn addChild(self: *KbStore, parent_id: i32, child_id: i32) types.Status {
    _ = child_id;
    var kb = self.getKb(parent_id) orelse return types.Status.err(.kb, .kb_not_found, parent_id);
    // For now, children stored as a convention in fact store
    // with TAG_REFERENCE pointing to child kb_id.
    // More sophisticated: dedicated children array in live_state.
    kb.children_count += 1;
    kb.last_modified = currentTimestamp();
    return self.writeKbToDevice(parent_id, &kb);
}

pub fn removeChild(self: *KbStore, parent_id: i32, child_id: i32) types.Status {
    var kb = self.getKb(parent_id) orelse return types.Status.err(.kb, .kb_not_found, parent_id);
    _ = child_id;
    if (kb.children_count > 0) kb.children_count -= 1;
    kb.last_modified = currentTimestamp();
    return self.writeKbToDevice(parent_id, &kb);
}

pub fn addMount(self: *KbStore, kb_id: i32, source_kb_id: i32, mount_name: []const u8) types.Status {
    var kb = self.getKb(kb_id) orelse return types.Status.err(.kb, .kb_not_found, kb_id);
    _ = source_kb_id;
    _ = mount_name;
    kb.mounts_count += 1;
    kb.last_modified = currentTimestamp();
    return self.writeKbToDevice(kb_id, &kb);
}

pub fn removeMount(self: *KbStore, kb_id: i32, mount_name: []const u8) types.Status {
    var kb = self.getKb(kb_id) orelse return types.Status.err(.kb, .kb_not_found, kb_id);
    _ = mount_name;
    if (kb.mounts_count > 0) kb.mounts_count -= 1;
    kb.last_modified = currentTimestamp();
    return self.writeKbToDevice(kb_id, &kb);
}

// ============================================================
// COW helpers
// ============================================================

pub fn cowInit(self: *KbStore, parent_session_id: i32, clone_session_id: i32, region_size: i64, parent_offset: i64, private_offset: i64) types.Status {
    const cow = CowPageTable.init(self.allocator, parent_session_id, clone_session_id, region_size, parent_offset, private_offset);
    self.cow_tables.append(cow) catch return types.Status.err(.session, .clone_failed, 0);
    return types.Status.ok();
}

pub fn cowDestroy(self: *KbStore, clone_session_id: i32) void {
    var i: usize = 0;
    while (i < self.cow_tables.items.len) {
        if (self.cow_tables.items[i].clone_session_id == clone_session_id) {
            self.cow_tables.items[i].deinit(self.allocator);
            _ = self.cow_tables.swapRemove(i);
        } else {
            i += 1;
        }
    }
}

pub fn cowResolve(self: *KbStore, clone_session_id: i32) types.Status {
    if (self.findCowMut(clone_session_id)) |cow| {
        var page: i32 = 0;
        while (page < cow.n_pages) : (page += 1) {
            if (!cow.isDirty(page)) {
                _ = self.cowCopyPage(cow, page);
            }
        }
        return types.Status.ok();
    }
    return types.Status.err(.session, .clone_failed, clone_session_id);
}

fn findCow(self: *KbStore, kb_id: i32) ?*CowPageTable {
    // Find COW table that covers this KB's region
    // Simplified: match by checking if kb's fact region overlaps COW range
    _ = kb_id;
    for (self.cow_tables.items) |*cow| {
        return cow; // simplified — real impl checks offset ranges
    }
    return null;
}

fn findCowMut(self: *KbStore, clone_session_id: i32) ?*CowPageTable {
    for (self.cow_tables.items) |*cow| {
        if (cow.clone_session_id == clone_session_id) return cow;
    }
    return null;
}

fn cowCopyPage(self: *KbStore, cow: *CowPageTable, page: i32) types.Status {
    const src_off = cow.pageOffsetInParent(page);
    const dst_off = cow.pageOffsetInPrivate(page);
    const status = self.bridge.copyBufferToBuffer(.fact_store, src_off, .fact_store, dst_off, COW_PAGE_SIZE);
    if (status.isOk()) cow.markDirty(page);
    return status;
}

// ============================================================
// Internal helpers
// ============================================================

fn cacheKb(self: *KbStore, kb_id: i32, kb: *const types.Kb) void {
    if (kb_id >= 0 and kb_id < self.kb_cache_capacity) {
        const idx: usize = @intCast(kb_id);
        if (idx < self.kb_cache.len) {
            self.kb_cache[idx] = kb.*;
            self.kb_cache_valid[idx] = true;
        }
    }
}

fn invalidateCache(self: *KbStore, kb_id: i32) void {
    if (kb_id >= 0 and kb_id < self.kb_cache_capacity) {
        const idx: usize = @intCast(kb_id);
        if (idx < self.kb_cache_valid.len) {
            self.kb_cache_valid[idx] = false;
        }
    }
}

fn writeKbToDevice(self: *KbStore, kb_id: i32, kb: *const types.Kb) types.Status {
    const offset = @as(i64, kb_id) * types.KB_STRUCT_SIZE;
    const bytes: []const u8 = @as([*]const u8, @ptrCast(kb))[0..@sizeOf(types.Kb)];
    const status = self.bridge.uploadToBuffer(.kb_store, offset, bytes);
    if (status.isOk()) self.cacheKb(kb_id, kb);
    return status;
}

fn currentTimestamp() i32 {
    // Host-side wall clock, seconds since epoch, truncated to i32
    const ts = std.time.timestamp();
    return @intCast(@min(ts, std.math.maxInt(i32)));
}
