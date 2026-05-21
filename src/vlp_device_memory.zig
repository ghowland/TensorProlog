// ============================================================
// vlp_device_memory.zig
// Device memory layout and capacity planning.
// ============================================================

const types = @import("vlp_types.zig");

pub const DeviceMemoryLayout = struct {
    // Model weights
    model_weights_base: i64,
    model_weights_size: i64,

    // KB store — array of Kb structs (256 bytes each)
    kb_store_base: i64,
    kb_store_size: i64,
    kb_count: i32,
    kb_capacity: i32,

    // Fact store — array of Fact structs (40 bytes each)
    fact_store_base: i64,
    fact_store_size: i64,
    fact_capacity: i64,

    // Rule store — array of Rule structs
    rule_store_base: i64,
    rule_store_size: i64,
    rule_capacity: i32,

    // Term store — array of Term structs (24 bytes each)
    term_store_base: i64,
    term_store_size: i64,
    term_capacity: i64,

    // Text store — raw bytes, append-only
    text_store_base: i64,
    text_store_size: i64,
    text_used: i64,

    // Grammar store
    grammar_store_base: i64,
    grammar_store_size: i64,
    grammar_capacity: i32,

    // Live state — per-session bounded primitives
    live_state_base: i64,
    live_state_size: i64,

    // Scratch — per-stream temporary buffers
    scratch_base: i64,
    scratch_size: i64,

    // KV cache
    kv_cache_base: i64,
    kv_cache_size: i64,

    // Audit ring buffer
    audit_base: i64,
    audit_size: i64,
    audit_capacity: i32,
    audit_head: i32,

    // Grant store
    grant_store_base: i64,
    grant_store_size: i64,
    grant_capacity: i32,

    // Session table
    session_table_base: i64,
    session_table_size: i64,
    session_capacity: i32,

    // Status + result count buffers (for GPU kernel output)
    status_buffer_base: i64,
    status_buffer_size: i64,
    result_counts_base: i64,
    result_counts_size: i64,

    // Params uniform buffer
    params_buffer_base: i64,
    params_buffer_size: i64,
};

pub const SizingConfig = struct {
    // Model
    model_params: i64,
    qbasis: types.QBasis,

    // Stores
    max_total_kbs: i32,
    max_total_facts: i64,
    max_total_rules: i32,
    max_total_terms: i64,
    text_store_bytes: i64,
    max_grammars: i32,

    // Sessions
    max_concurrent_sessions: i32,
    live_state_per_session: i64,

    // Scratch
    scratch_per_stream: i64,
    n_scratch_streams: i32,

    // KV cache
    kv_max_seq_len: i32,
    kv_n_layers: i32,
    kv_n_heads: i32,
    kv_d_head: i32,

    // Safety
    audit_ring_capacity: i32,
    max_grants: i32,

    // GPU dispatch
    max_dispatch_invocations: i32,
};

pub const CapacityResult = struct {
    model_bytes: i64,
    kb_store_bytes: i64,
    fact_store_bytes: i64,
    rule_store_bytes: i64,
    term_store_bytes: i64,
    text_store_bytes: i64,
    grammar_store_bytes: i64,
    live_state_bytes: i64,
    scratch_bytes: i64,
    kv_cache_bytes: i64,
    audit_bytes: i64,
    grant_store_bytes: i64,
    session_table_bytes: i64,
    control_bytes: i64,
    total_bytes: i64,
    n_devices_required: i32,
};

const FACT_SIZE: i64 = 40;
const KB_SIZE: i64 = 256;
const RULE_SIZE: i64 = 48; // Rule extern struct padded
const TERM_SIZE: i64 = 24;
const GRAMMAR_SIZE: i64 = 28;
const AUDIT_ENTRY_SIZE: i64 = 32;
const GRANT_SIZE: i64 = 48;
const SESSION_SIZE: i64 = 128;
const STATUS_ENTRY_SIZE: i64 = 4;
const PARAMS_BUFFER_SIZE: i64 = 256;

pub fn computeCapacity(config: *const SizingConfig) CapacityResult {
    const bytes_per_param: i64 = switch (config.qbasis) {
        .q16 => 8,
        .q32 => 16,
        .q335 => 240,
    };

    const model = config.model_params * bytes_per_param;
    const kb = @as(i64, config.max_total_kbs) * KB_SIZE;
    const fact = config.max_total_facts * FACT_SIZE;
    const rule = @as(i64, config.max_total_rules) * RULE_SIZE;
    const term = config.max_total_terms * TERM_SIZE;
    const text = config.text_store_bytes;
    const grammar = @as(i64, config.max_grammars) * GRAMMAR_SIZE;
    const live = @as(i64, config.max_concurrent_sessions) * config.live_state_per_session;
    const scratch = @as(i64, config.n_scratch_streams) * config.scratch_per_stream;

    // KV cache: 2 (K+V) × n_layers × max_seq × n_heads × d_head × bytes_per_param
    const kv = @as(i64, 2) *
        @as(i64, config.kv_n_layers) *
        @as(i64, config.kv_max_seq_len) *
        @as(i64, config.kv_n_heads) *
        @as(i64, config.kv_d_head) *
        bytes_per_param;

    const audit = @as(i64, config.audit_ring_capacity) * AUDIT_ENTRY_SIZE;
    const grant = @as(i64, config.max_grants) * GRANT_SIZE;
    const session = @as(i64, config.max_concurrent_sessions) * SESSION_SIZE;
    const control = @as(i64, config.max_dispatch_invocations) * STATUS_ENTRY_SIZE +
        STATUS_ENTRY_SIZE * 16 + // result count slots
        PARAMS_BUFFER_SIZE;

    const total = model + kb + fact + rule + term + text + grammar +
        live + scratch + kv + audit + grant + session + control;

    const device_mem: i64 = 80 * 1024 * 1024 * 1024; // 80 GB (H100)
    const n_devices: i32 = @intCast(@divTrunc(total + device_mem - 1, device_mem));

    return .{
        .model_bytes = model,
        .kb_store_bytes = kb,
        .fact_store_bytes = fact,
        .rule_store_bytes = rule,
        .term_store_bytes = term,
        .text_store_bytes = text,
        .grammar_store_bytes = grammar,
        .live_state_bytes = live,
        .scratch_bytes = scratch,
        .kv_cache_bytes = kv,
        .audit_bytes = audit,
        .grant_store_bytes = grant,
        .session_table_bytes = session,
        .control_bytes = control,
        .total_bytes = total,
        .n_devices_required = n_devices,
    };
}

pub fn computeLayout(config: *const SizingConfig) DeviceMemoryLayout {
    const cap = computeCapacity(config);

    var offset: i64 = 0;

    var layout: DeviceMemoryLayout = undefined;

    layout.model_weights_base = offset;
    layout.model_weights_size = cap.model_bytes;
    offset += cap.model_bytes;

    layout.kb_store_base = offset;
    layout.kb_store_size = cap.kb_store_bytes;
    layout.kb_count = 0;
    layout.kb_capacity = config.max_total_kbs;
    offset += cap.kb_store_bytes;

    layout.fact_store_base = offset;
    layout.fact_store_size = cap.fact_store_bytes;
    layout.fact_capacity = config.max_total_facts;
    offset += cap.fact_store_bytes;

    layout.rule_store_base = offset;
    layout.rule_store_size = cap.rule_store_bytes;
    layout.rule_capacity = config.max_total_rules;
    offset += cap.rule_store_bytes;

    layout.term_store_base = offset;
    layout.term_store_size = cap.term_store_bytes;
    layout.term_capacity = config.max_total_terms;
    offset += cap.term_store_bytes;

    layout.text_store_base = offset;
    layout.text_store_size = cap.text_store_bytes;
    layout.text_used = 0;
    offset += cap.text_store_bytes;

    layout.grammar_store_base = offset;
    layout.grammar_store_size = cap.grammar_store_bytes;
    layout.grammar_capacity = config.max_grammars;
    offset += cap.grammar_store_bytes;

    layout.live_state_base = offset;
    layout.live_state_size = cap.live_state_bytes;
    offset += cap.live_state_bytes;

    layout.scratch_base = offset;
    layout.scratch_size = cap.scratch_bytes;
    offset += cap.scratch_bytes;

    layout.kv_cache_base = offset;
    layout.kv_cache_size = cap.kv_cache_bytes;
    offset += cap.kv_cache_bytes;

    layout.audit_base = offset;
    layout.audit_size = cap.audit_bytes;
    layout.audit_capacity = config.audit_ring_capacity;
    layout.audit_head = 0;
    offset += cap.audit_bytes;

    layout.grant_store_base = offset;
    layout.grant_store_size = cap.grant_store_bytes;
    layout.grant_capacity = config.max_grants;
    offset += cap.grant_store_bytes;

    layout.session_table_base = offset;
    layout.session_table_size = cap.session_table_bytes;
    layout.session_capacity = config.max_concurrent_sessions;
    offset += cap.session_table_bytes;

    layout.status_buffer_base = offset;
    layout.status_buffer_size = @as(i64, config.max_dispatch_invocations) * STATUS_ENTRY_SIZE;
    offset += layout.status_buffer_size;

    layout.result_counts_base = offset;
    layout.result_counts_size = STATUS_ENTRY_SIZE * 16;
    offset += layout.result_counts_size;

    layout.params_buffer_base = offset;
    layout.params_buffer_size = PARAMS_BUFFER_SIZE;

    return layout;
}

// ---- Default configuration for 7B model, 10K sessions ----

pub fn defaultSizingConfig() SizingConfig {
    return .{
        .model_params = 7_000_000_000,
        .qbasis = .q16,
        .max_total_kbs = 100_000,
        .max_total_facts = 10_000_000,
        .max_total_rules = 100_000,
        .max_total_terms = 1_000_000,
        .text_store_bytes = 100 * 1024 * 1024,
        .max_grammars = 10_000,
        .max_concurrent_sessions = 10_000,
        .live_state_per_session = 50 * 1024,
        .scratch_per_stream = 10 * 1024 * 1024,
        .n_scratch_streams = 100,
        .kv_max_seq_len = 4096,
        .kv_n_layers = 32,
        .kv_n_heads = 32,
        .kv_d_head = 128,
        .audit_ring_capacity = 1_000_000,
        .max_grants = 100_000,
        .max_dispatch_invocations = 65536,
    };
}
