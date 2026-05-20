
// ============================================================
// src/gpu/memory.zig
// ============================================================

const kb_types = @import("../kb/types.zig");

pub const MemcpyKind = enum(i8) {
    host_to_device = 0,
    device_to_host = 1,
    device_to_device = 2,
};

pub const DeviceMemoryLayout = struct {
    model_weights_base: i64,
    model_weights_size: i64,

    kb_store_base: i64,
    kb_store_size: i64,
    kb_capacity: i32,

    fact_store_base: i64,
    fact_store_size: i64,
    fact_capacity: i64,

    rule_store_base: i64,
    rule_store_size: i64,

    term_store_base: i64,
    term_store_size: i64,

    text_store_base: i64,
    text_store_size: i64,

    grammar_store_base: i64,
    grammar_store_size: i64,

    live_state_base: i64,
    live_state_size: i64,

    scratch_base: i64,
    scratch_size: i64,

    audit_base: i64,
    audit_size: i64,
    audit_capacity: i32,

    grant_store_base: i64,
    grant_store_size: i64,

    session_table_base: i64,
    session_table_size: i64,
    session_capacity: i32,

    path_index_base: i64,
    path_index_size: i64,

    total_bytes: i64,
};

pub const MemoryConfig = struct {
    model_params: i64,
    qbasis_bytes: i32,
    max_kbs: i32,
    max_facts: i64,
    max_rules: i32,
    max_terms: i64,
    text_store_bytes: i64,
    max_grammars: i32,
    max_sessions: i32,
    live_state_per_session: i32,
    scratch_per_stream: i32,
    n_streams: i32,
    audit_capacity: i32,
    max_grants: i32,
    path_index_entries: i32,
};

pub fn defaultMemoryConfig() MemoryConfig {
    return .{
        .model_params = 0,
        .qbasis_bytes = 8,
        .max_kbs = 100000,
        .max_facts = 10000000,
        .max_rules = 100000,
        .max_terms = 1000000,
        .text_store_bytes = 104857600,
        .max_grammars = 10000,
        .max_sessions = 10000,
        .live_state_per_session = 51200,
        .scratch_per_stream = 10485760,
        .n_streams = 100,
        .audit_capacity = 1000000,
        .max_grants = 100000,
        .path_index_entries = 200000,
    };
}

pub fn computeLayout(config: MemoryConfig) DeviceMemoryLayout {
    var layout: DeviceMemoryLayout = undefined;
    var offset: i64 = 0;

    layout.model_weights_base = offset;
    layout.model_weights_size = config.model_params * @as(i64, config.qbasis_bytes);
    offset += layout.model_weights_size;
    offset = alignUp(offset, 256);

    layout.kb_store_base = offset;
    layout.kb_capacity = config.max_kbs;
    layout.kb_store_size = @as(i64, config.max_kbs) * 256;
    offset += layout.kb_store_size;
    offset = alignUp(offset, 256);

    layout.fact_store_base = offset;
    layout.fact_capacity = config.max_facts;
    layout.fact_store_size = config.max_facts * 40;
    offset += layout.fact_store_size;
    offset = alignUp(offset, 256);

    layout.rule_store_base = offset;
    layout.rule_store_size = @as(i64, config.max_rules) * 44;
    offset += layout.rule_store_size;
    offset = alignUp(offset, 256);

    layout.term_store_base = offset;
    layout.term_store_size = config.max_terms * 24;
    offset += layout.term_store_size;
    offset = alignUp(offset, 256);

    layout.text_store_base = offset;
    layout.text_store_size = config.text_store_bytes;
    offset += layout.text_store_size;
    offset = alignUp(offset, 256);

    layout.grammar_store_base = offset;
    layout.grammar_store_size = @as(i64, config.max_grammars) * 28;
    offset += layout.grammar_store_size;
    offset = alignUp(offset, 256);

    layout.live_state_base = offset;
    layout.live_state_size = @as(i64, config.max_sessions) * @as(i64, config.live_state_per_session);
    offset += layout.live_state_size;
    offset = alignUp(offset, 256);

    layout.scratch_base = offset;
    layout.scratch_size = @as(i64, config.n_streams) * @as(i64, config.scratch_per_stream);
    offset += layout.scratch_size;
    offset = alignUp(offset, 256);

    layout.audit_base = offset;
    layout.audit_capacity = config.audit_capacity;
    layout.audit_size = @as(i64, config.audit_capacity) * 28;
    offset += layout.audit_size;
    offset = alignUp(offset, 256);

    layout.grant_store_base = offset;
    layout.grant_store_size = @as(i64, config.max_grants) * 48;
    offset += layout.grant_store_size;
    offset = alignUp(offset, 256);

    layout.session_table_base = offset;
    layout.session_capacity = config.max_sessions;
    layout.session_table_size = @as(i64, config.max_sessions) * 128;
    offset += layout.session_table_size;
    offset = alignUp(offset, 256);

    layout.path_index_base = offset;
    layout.path_index_size = @as(i64, config.path_index_entries) * 8;
    offset += layout.path_index_size;
    offset = alignUp(offset, 256);

    layout.total_bytes = offset;

    return layout;
}

fn alignUp(val: i64, alignment: i64) i64 {
    return ((val + alignment - 1) / alignment) * alignment;
}

pub const DeviceAllocation = struct {
    host_buf: []u8,
    layout: DeviceMemoryLayout,
    allocated: bool,
};

pub fn allocateDevice(layout: DeviceMemoryLayout, backing: []u8) DeviceAllocation {
    const needed: usize = @intCast(layout.total_bytes);
    if (backing.len < needed) {
        return .{ .host_buf = backing, .layout = layout, .allocated = false };
    }

    @memset(backing[0..needed], 0);

    return .{
        .host_buf = backing[0..needed],
        .layout = layout,
        .allocated = true,
    };
}

pub fn freeDevice(alloc: *DeviceAllocation) void {
    alloc.allocated = false;
}

pub fn getRegion(alloc: *const DeviceAllocation, base: i64, size: i64) ?[]u8 {
    if (!alloc.allocated) return null;
    const b: usize = @intCast(base);
    const s: usize = @intCast(size);
    if (b + s > alloc.host_buf.len) return null;
    return alloc.host_buf[b .. b + s];
}

pub fn getRegionConst(alloc: *const DeviceAllocation, base: i64, size: i64) ?[]const u8 {
    if (!alloc.allocated) return null;
    const b: usize = @intCast(base);
    const s: usize = @intCast(size);
    if (b + s > alloc.host_buf.len) return null;
    return alloc.host_buf[b .. b + s];
}
