// ============================================================
// vlp_gpu_shared.zig
// SPIR-V-safe shared definitions.
// No functions. No std. No imports. Pure declarations.
// Compiles for both spirv32-vulkan and native x86_64.
// This file is the single source of truth for host-device contract.
// ============================================================

const shared = @import("vlp_gpu_shared");

// ============================================================
// System constants
// ============================================================

pub const FACTS_PER_KB: i32 = 500;
pub const FACT_INTS: i32 = 10; // sizeof(Fact) / 4 = 40 / 4
pub const FACT_STRIDE: i32 = FACTS_PER_KB * FACT_INTS; // 5000 ints per KB's fact block
pub const KB_STRUCT_INTS: i32 = 64; // 256 bytes / 4
pub const RULE_INTS: i32 = 12; // 48 bytes / 4
pub const TERM_INTS: i32 = 6; // 24 bytes / 4
pub const BINDING_INTS: i32 = 2; // 8 bytes / 4
pub const D: i32 = 65536; // Q16 denominator
pub const D32: i64 = 4294967296; // 2^32
pub const MAX_WORKGROUP: i32 = 256;
pub const MAX_BODY_CONDITIONS: i32 = 16;
pub const MAX_COMPOUND_ARGS: i32 = 16;
pub const MAX_SORT_ELEMENTS: i32 = 256; // single-workgroup bitonic sort
pub const MAX_CHAIN_LINKS: i32 = 100;
pub const MAX_BINDINGS_PER: i32 = 8;

// ============================================================
// Buffer size declarations — upper bounds for extern struct arrays.
// Actual buffer sizes determined by Vulkan descriptor binding.
// These are compile-time constants the SPIR-V module uses for typing.
// ============================================================

pub const MAX_BUFFER_INTS: i32 = 16 * 1024 * 1024; // 64 MB per buffer
pub const MAX_KV_CACHE_INTS: i32 = 64 * 1024 * 1024; // 256 MB
pub const MAX_STATUS_ENTRIES: i32 = 65536;
pub const MAX_RESULT_SLOTS: i32 = 16;
pub const PARAMS_INTS: i32 = 64; // 256 bytes uniform

// ============================================================
// Op codes — kernel entry point switch
// ============================================================

pub const OpCode = enum(i32) {
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
    fact_write_batch = 11,
    fact_read_batch = 12,
    fact_scan_by_tag = 13,
    scoped_search = 14,
    unify_candidates = 15,
    rule_match_scan = 16,
    rule_body_eval = 17,
    rule_check_satisfied = 18,
    builtin_unary = 19,
    builtin_binary = 20,
    builtin_reduction = 21,
    builtin_sort = 22,
    builtin_matmul = 23,
    confidence_combine = 24,
    confidence_chain = 25,
    buffer_copy = 26,
    buffer_fill = 27,
};

// ============================================================
// Fact tag enum
// ============================================================

pub const FactTag = enum(i32) {
    value = 0,
    text = 1,
    reference = 2,
    timestamp = 3,
    enum_tag = 4,
    boolean = 5,
    vector = 6,
    matrix = 7,
    provenance_tag = 8,
    rule_ref = 9,
    grammar_ref = 10,
    counter = 11,
    empty = 255,
};

// ============================================================
// Term type enum
// ============================================================

pub const TermType = enum(i32) {
    atom = 0,
    variable = 1,
    integer = 2,
    vdr = 3,
    text = 4,
    list = 5,
    compound = 6,
    vector = 7,
    matrix = 8,
    pair = 9,
};

// ============================================================
// Source type enum
// ============================================================

pub const SourceType = enum(i32) {
    vdr_computation = 0,
    prolog_derivation = 1,
    database = 2,
    prometheus = 3,
    script = 4,
    rest_api = 5,
    published = 6,
    user_stated = 7,
    web_search = 8,
    llm_generated = 9,
    unknown = 10,
};

// ============================================================
// Confidence table — immutable, indexed by SourceType
// ============================================================

pub const confidence_table = [11]i32{
    65536, // vdr_computation 1/1
    65536, // prolog_derivation 1/1
    64225, // database 98/100
    62259, // prometheus 95/100
    62259, // script 95/100
    55705, // rest_api 85/100
    52428, // published 80/100
    45875, // user_stated 70/100
    32768, // web_search 50/100
    19660, // llm_generated 30/100
    0, // unknown 0/1
};

// ============================================================
// Descriptor set and binding indices
// ============================================================

pub const SET_MODEL: u32 = 0;
pub const SET_KB_DATA: u32 = 1;
pub const SET_SCRATCH: u32 = 2;
pub const SET_CONTROL: u32 = 3;

pub const BIND_EMBEDDING: u32 = 0;
pub const BIND_LAYER_WEIGHTS: u32 = 1;
pub const BIND_LM_HEAD: u32 = 2;
pub const BIND_LN_PARAMS: u32 = 3;

pub const BIND_KB_STORE: u32 = 0;
pub const BIND_FACT_STORE: u32 = 1;
pub const BIND_RULE_STORE: u32 = 2;
pub const BIND_TERM_STORE: u32 = 3;
pub const BIND_LIVE_STATE: u32 = 4;

pub const BIND_SCRATCH_A: u32 = 0;
pub const BIND_SCRATCH_B: u32 = 1;
pub const BIND_KV_CACHE: u32 = 2;

pub const BIND_PARAMS: u32 = 0;
pub const BIND_STATUS: u32 = 1;
pub const BIND_RESULT_COUNTS: u32 = 2;

// ============================================================
// Fact field offsets within FACT_INTS (10 ints = 40 bytes)
// ============================================================

pub const FACT_TAG: i32 = 0;
pub const FACT_VALUE_V: i32 = 1;
pub const FACT_VALUE_R0: i32 = 2; // packed: r0 in lower 16 bits, r1 in upper 16 bits
pub const FACT_PROV_SOURCE: i32 = 3;
pub const FACT_PROV_KB: i32 = 4;
pub const FACT_PROV_SLOT: i32 = 5;
pub const FACT_PROV_CONF_V: i32 = 6;
pub const FACT_PROV_CONF_R0: i32 = 7;
pub const FACT_PROV_TIME: i32 = 8;
pub const FACT_PROV_RULE: i32 = 9;

// ============================================================
// Term field offsets within TERM_INTS (6 ints = 24 bytes)
// ============================================================

pub const TERM_TYPE: i32 = 0; // lower 8 bits
pub const TERM_PRIMARY: i32 = 1; // atom_id / var_id / int_value / functor_id
pub const TERM_SECONDARY: i32 = 2; // text_offset / list_head / args_offset
pub const TERM_AUX: i32 = 3; // text_len / list_tail / args_count
pub const TERM_VDR_V: i32 = 4;
pub const TERM_VDR_R0: i32 = 5;

// ============================================================
// Rule field offsets within RULE_INTS (12 ints = 48 bytes)
// ============================================================

pub const RULE_ID: i32 = 0;
pub const RULE_HEAD: i32 = 1;
pub const RULE_BODY_OFF: i32 = 2;
pub const RULE_BODY_COUNT: i32 = 3; // lower 16 bits
pub const RULE_ACTION_OFF: i32 = 4;
pub const RULE_ACTION_COUNT: i32 = 5; // lower 16 bits
pub const RULE_FIRE_COUNT: i32 = 6;
pub const RULE_LAST_FIRED: i32 = 7;
pub const RULE_SUCCESS: i32 = 8;
pub const RULE_FAILURE: i32 = 9;
pub const RULE_CREATED: i32 = 10;
pub const RULE_CREATOR: i32 = 11;

// ============================================================
// Params field offsets — common header then per-op fields
// All offsets in i32 units into the params uniform buffer.
// ============================================================

pub const P_OP_CODE: i32 = 0;
pub const P_FIELD_0: i32 = 1;
pub const P_FIELD_1: i32 = 2;
pub const P_FIELD_2: i32 = 3;
pub const P_FIELD_3: i32 = 4;
pub const P_FIELD_4: i32 = 5;
pub const P_FIELD_5: i32 = 6;
pub const P_FIELD_6: i32 = 7;
pub const P_FIELD_7: i32 = 8;
pub const P_FIELD_8: i32 = 9;
pub const P_FIELD_9: i32 = 10;
pub const P_FIELD_10: i32 = 11;
pub const P_FIELD_11: i32 = 12;

// ============================================================
// Per-op param field assignments (documented, not enforced)
//
// embedding_lookup:  F0=n_tokens F1=d_model
// layer_norm:        F0=n_tokens F1=d_model F2=layer_idx F3=norm_idx F4=epsilon_v
// qkv_project:       F0=n_tokens F1=d_model F2=n_heads F3=d_head F4=layer_idx F5=weights_offset
// attention_scores:  F0=n_tokens F1=n_heads F2=d_head F3=seq_len F4=scale_v F5=causal_mask
// softmax_exact:     F0=row_length F1=n_rows F2=denominator
// attn_weighted_sum: F0=n_tokens F1=n_heads F2=d_head F3=seq_len
// output_project:    F0=n_tokens F1=d_model F2=layer_idx F3=weights_offset
// mlp:               F0=n_tokens F1=d_model F2=mlp_dim F3=layer_idx F4=up_w_off F5=down_w_off F6=act_type
// lm_head:           F0=n_tokens F1=d_model F2=vocab_size
// kv_cache_append:   F0=n_new_tokens F1=n_heads F2=d_head F3=layer_idx F4=start_pos F5=max_seq
// residual_add:      F0=n_elements
// fact_write_batch:  F0=n_facts F1=base_offset F2=capacity
// fact_read_batch:   F0=n_reads
// fact_scan_by_tag:  F0=base_offset F1=scan_length F2=target_tag F3=max_results
// scoped_search:     F0=n_chain F1=total_facts F2=target_tag F3=max_results
// unify_candidates:  F0=n_cand F1=q_type F2=q_atom F3=q_int F4=q_vdr_v F5=q_vdr_r0 F6=q_func F7=q_argc F8=q_argoff F9=max_bind
// rule_match_scan:   F0=n_rules F1=rules_base F2=q_type F3=q_atom F4=q_func F5=q_argc F6=max_matches
// rule_body_eval:    F0=n_matched F1=max_body F2=facts_base F3=facts_count
// rule_check_sat:    F0=n_matched F1=max_body F2=max_fires
// builtin_unary:     F0=n_elements F1=sub_op F2=input_off F3=output_off
// builtin_binary:    F0=n_elements F1=sub_op F2=in_a_off F3=in_b_off F4=output_off
// builtin_reduction: F0=n_elements F1=sub_op F2=input_off
// builtin_sort:      F0=n_elements F1=ascending F2=input_off F3=output_off
// builtin_matmul:    F0=m F1=n F2=k F3=a_off F4=b_off F5=c_off
// confidence_combine:F0=n_sources F1=mode F2=penalty_v F3=input_off
// confidence_chain:  F0=n_links F1=per_link_v
// buffer_copy:       F0=src_off F1=dst_off F2=n_elements F3=elem_size
// buffer_fill:       F0=dst_off F1=n_elements F2=fill_value F3=elem_size
// ============================================================

// ============================================================
// Integer exp lookup table for softmax
// exp(-k) * D for k=0..10
// ============================================================

pub const exp_table = [11]i32{
    65536, 24109, 8874, 3263, 1201, 442, 162, 60, 22, 8, 3,
};
