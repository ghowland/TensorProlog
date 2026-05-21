// ============================================================
// vlp_gpu_params.zig
// Dispatch parameter structs for all GPU compute kernels.
// Written to uniform buffer (Set 3, binding 0) before dispatch.
// All extern struct for GPU layout compatibility.
// ============================================================

const types = @import("vlp_types.zig");

// ============================================================
// LLM Kernel Params
// ============================================================

pub const EmbeddingLookupParams = extern struct {
    n_tokens: i32,
    d_model: i32,
    _pad: [2]i32 = [_]i32{0} ** 2,
};

pub const LayerNormParams = extern struct {
    n_tokens: i32,
    d_model: i32,
    layer_idx: i32,
    norm_idx: i32, // 0=pre-attn, 1=pre-mlp, 2=final
    epsilon_v: i32 = 1, // Q16 epsilon, default ~1/65536
    _pad: [3]i32 = [_]i32{0} ** 3,
};

pub const QkvProjectParams = extern struct {
    n_tokens: i32,
    d_model: i32,
    n_heads: i32,
    d_head: i32,
    layer_idx: i32,
    weights_offset: i32,
    _pad: [2]i32 = [_]i32{0} ** 2,
};

pub const AttentionScoresParams = extern struct {
    n_tokens: i32,
    n_heads: i32,
    d_head: i32,
    seq_len: i32,
    scale_v: i32, // Q16: 1/sqrt(d_head)
    causal_mask: i32, // 1=apply, 0=no
    _pad: [2]i32 = [_]i32{0} ** 2,
};

pub const SoftmaxExactParams = extern struct {
    row_length: i32,
    n_rows: i32,
    denominator: i32 = types.Q16.D,
    _pad: i32 = 0,
};

pub const AttentionWeightedSumParams = extern struct {
    n_tokens: i32,
    n_heads: i32,
    d_head: i32,
    seq_len: i32,
};

pub const OutputProjectParams = extern struct {
    n_tokens: i32,
    d_model: i32,
    layer_idx: i32,
    weights_offset: i32,
};

pub const MlpParams = extern struct {
    n_tokens: i32,
    d_model: i32,
    mlp_dim: i32,
    layer_idx: i32,
    up_weights_offset: i32,
    down_weights_offset: i32,
    activation_type: i32, // 0=SiLU, 1=GELU, 2=ReLU
    _pad: i32 = 0,
};

pub const LmHeadParams = extern struct {
    n_tokens: i32,
    d_model: i32,
    vocab_size: i32,
    _pad: i32 = 0,
};

pub const KvCacheAppendParams = extern struct {
    n_new_tokens: i32,
    n_heads: i32,
    d_head: i32,
    layer_idx: i32,
    start_position: i32,
    max_seq_len: i32,
    _pad: [2]i32 = [_]i32{0} ** 2,
};

pub const ResidualAddParams = extern struct {
    n_elements: i32,
    _pad: [3]i32 = [_]i32{0} ** 3,
};

// ============================================================
// KB Kernel Params
// ============================================================

pub const FactWriteBatchParams = extern struct {
    n_facts: i32,
    base_offset: i32,
    fact_store_capacity: i32,
    _pad: i32 = 0,
};

pub const FactReadBatchParams = extern struct {
    n_reads: i32,
    _pad: [3]i32 = [_]i32{0} ** 3,
};

pub const FactScanByTagParams = extern struct {
    base_offset: i32,
    scan_length: i32,
    target_tag: i32,
    max_results: i32,
};

pub const ScopedSearchParams = extern struct {
    n_chain_entries: i32,
    total_facts: i32,
    target_tag: i32,
    max_results: i32,
};

// ============================================================
// Prolog Kernel Params
// ============================================================

pub const UnifyCandidatesParams = extern struct {
    n_candidates: i32,
    query_term_type: i32,
    query_atom_id: i32,
    query_int_value: i32,
    query_vdr_v: i32,
    query_vdr_r0: i16,
    _pad0: i16 = 0,
    query_functor_id: i32,
    query_args_count: i32,
    query_args_offset: i32,
    max_bindings_per: i32,
};

pub const RuleMatchScanParams = extern struct {
    n_rules: i32,
    rules_base_offset: i32,
    query_term_type: i32,
    query_atom_id: i32,
    query_functor_id: i32,
    query_args_count: i32,
    max_matches: i32,
    _pad: i32 = 0,
};

pub const RuleBodyEvalParams = extern struct {
    n_matched: i32,
    max_body: i32,
    facts_base_offset: i32,
    facts_count: i32,
};

pub const RuleCheckSatisfiedParams = extern struct {
    n_matched: i32,
    max_body: i32,
    max_fires: i32,
    _pad: i32 = 0,
};

// ============================================================
// Builtin Kernel Params
// ============================================================

pub const UnaryOpCode = enum(i32) {
    abs = 0,
    negate = 1,
    sign = 2,
    complement = 3,
    clz = 4,
    ctz = 5,
    popcount = 6,
    is_zero = 7,
    is_positive = 8,
    is_negative = 9,
    square = 10,
    double = 11,
    halve = 12,
};

pub const BinaryOpCode = enum(i32) {
    add = 0,
    sub = 1,
    mul = 2,
    div = 3,
    mod = 4,
    min = 5,
    max = 6,
    gcd = 7,
    lcm = 8,
    bit_and = 9,
    bit_or = 10,
    bit_xor = 11,
    shift_left = 12,
    shift_right = 13,
    compare = 14,
    cross_multiply_compare = 15,
    power = 16,
};

pub const ReductionOpCode = enum(i32) {
    sum = 0,
    product = 1,
    min = 2,
    max = 3,
    mean = 4,
    variance = 5,
    count_nonzero = 6,
    any_positive = 7,
    all_positive = 8,
    argmin = 9,
    argmax = 10,
    median = 11,
};

pub const BuiltinUnaryParams = extern struct {
    n_elements: i32,
    op_code: i32,
    input_offset: i32,
    output_offset: i32,
};

pub const BuiltinBinaryParams = extern struct {
    n_elements: i32,
    op_code: i32,
    input_a_offset: i32,
    input_b_offset: i32,
    output_offset: i32,
    _pad: [3]i32 = [_]i32{0} ** 3,
};

pub const BuiltinReductionParams = extern struct {
    n_elements: i32,
    op_code: i32,
    input_offset: i32,
    _pad: i32 = 0,
};

pub const BuiltinSortParams = extern struct {
    n_elements: i32,
    ascending: i32,
    input_offset: i32,
    output_offset: i32,
};

pub const BuiltinMatmulParams = extern struct {
    m: i32,
    n: i32,
    k: i32,
    _pad: i32 = 0,
    a_offset: i32,
    b_offset: i32,
    c_offset: i32,
    _pad2: i32 = 0,
};

pub const BuiltinConfidenceCombineParams = extern struct {
    n_sources: i32,
    mode: i32, // 0=agreeing, 1=conflicting
    penalty_v: i32, // Q16 penalty for conflicting
    input_offset: i32,
};

pub const BuiltinConfidenceChainParams = extern struct {
    n_links: i32,
    per_link_v: i32,
    _pad: [2]i32 = [_]i32{0} ** 2,
};

// ============================================================
// Utility Kernel Params
// ============================================================

pub const BufferCopyParams = extern struct {
    src_offset: i32,
    dst_offset: i32,
    n_elements: i32,
    element_size: i32,
};

pub const BufferFillParams = extern struct {
    dst_offset: i32,
    n_elements: i32,
    fill_value: i32,
    element_size: i32,
};

// ============================================================
// Pipeline ID enum — host uses this to select compute pipeline
// ============================================================

pub const PipelineId = enum(i32) {
    // LLM (0-10)
    embedding_lookup = 0,
    layer_norm = 1,
    qkv_project = 2,
    attention_scores = 3,
    softmax_exact = 4,
    attention_weighted_sum = 5,
    output_project = 6,
    mlp = 7,
    lm_head = 8,
    kv_cache_append = 9,
    residual_add = 10,
    // KB (11-14)
    fact_write_batch = 11,
    fact_read_batch = 12,
    fact_scan_by_tag = 13,
    scoped_search = 14,
    // Prolog (15-18)
    unify_candidates = 15,
    rule_match_scan = 16,
    rule_body_eval = 17,
    rule_check_satisfied = 18,
    // Builtins (19-25)
    builtin_unary = 19,
    builtin_binary = 20,
    builtin_reduction = 21,
    builtin_sort = 22,
    builtin_matmul = 23,
    builtin_confidence_combine = 24,
    builtin_confidence_chain = 25,
    // Utility (26-27)
    buffer_copy = 26,
    buffer_fill = 27,

    pub const count = 28;
};

// ============================================================
// Descriptor set binding indices — matches GLSL layout
// ============================================================

pub const DescriptorSet = enum(u32) {
    model = 0, // Set 0: model weights (read-only, bound once)
    kb_data = 1, // Set 1: KB/fact/rule/term/live stores (per-session)
    scratch = 2, // Set 2: scratch_a, scratch_b, kv_cache (per-dispatch)
    control = 3, // Set 3: params uniform, status, result_counts (per-dispatch)
};

// Binding indices within each set
pub const ModelBindings = struct {
    pub const embedding_table: u32 = 0;
    pub const layer_weights: u32 = 1;
    pub const lm_head: u32 = 2;
    pub const layer_norm_params: u32 = 3;
};

pub const KbDataBindings = struct {
    pub const kb_store: u32 = 0;
    pub const fact_store: u32 = 1;
    pub const rule_store: u32 = 2;
    pub const term_store: u32 = 3;
    pub const live_state: u32 = 4;
};

pub const ScratchBindings = struct {
    pub const scratch_a: u32 = 0;
    pub const scratch_b: u32 = 1;
    pub const kv_cache: u32 = 2;
};

pub const ControlBindings = struct {
    pub const dispatch_params: u32 = 0;
    pub const status_buffer: u32 = 1;
    pub const result_counts: u32 = 2;
};

// ============================================================
// Maximum workgroup size constants
// ============================================================

pub const MAX_WORKGROUP_SIZE: i32 = 256;
pub const WARP_SIZE: i32 = 32;

// Shared memory budget tiers (bytes)
pub const SHARED_MEM_BASELINE: i32 = 32 * 1024;
pub const SHARED_MEM_EXTENDED: i32 = 64 * 1024;
pub const SHARED_MEM_H100: i32 = 228 * 1024;

// pub fn unifyCandidates(n_cand: i32, q_type: i32, q_atom: i32, q_int: i32, q_vdr_v: i32, q_vdr_r0_r1: i32, q_func: i32, q_argc: i32, q_argoff: i32, max_bind: i32) UnifyCandidatesParams {
//     var buf = initParams(.unify_candidates);
//     buf[@intCast(shared.P_FIELD_0)] = n_cand;
//     buf[@intCast(shared.P_FIELD_1)] = q_type;
//     buf[@intCast(shared.P_FIELD_2)] = q_atom;
//     buf[@intCast(shared.P_FIELD_3)] = q_int;
//     buf[@intCast(shared.P_FIELD_4)] = q_vdr_v;
//     buf[@intCast(shared.P_FIELD_5)] = q_vdr_r0_r1; // packed: r0 lower 16, r1 upper 16
//     buf[@intCast(shared.P_FIELD_6)] = q_func;
//     buf[@intCast(shared.P_FIELD_7)] = q_argc;
//     buf[@intCast(shared.P_FIELD_8)] = q_argoff;
//     buf[@intCast(shared.P_FIELD_9)] = max_bind;
//     return buf;
// }
