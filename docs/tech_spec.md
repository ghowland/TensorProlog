# VDR-LLM-Prolog TensorProlog Technical Specification

## Complete System Integration Layer

### Version 0.1

---

## 1. Scope

This spec covers the layer between TensorProlog (the GPU compute API) and VDR-LLM-Prolog (the application architecture). TensorProlog provides integer arithmetic on GPU. VDR-LLM-Prolog provides KBs, Prolog, grammars, runners, sessions, builtins, safety, and the LLM-as-judgment-component model. This spec defines how the full system runs on TensorProlog hardware — the orchestration, data flow, memory management, scheduling, and lifecycle that turns raw integer GPU compute into the operational system described in the paper.

The paper's VDR-14 consolidated specification defines WHAT the system does. The TensorProlog spec defines HOW the GPU computes. This spec defines HOW THE SYSTEM RUNS — the integration contract between architecture and silicon.

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     HOST (CPU)                               │
│                                                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Session   │  │ Runner    │  │ Grant    │  │ Snapshot  │  │
│  │ Manager   │  │ Scheduler │  │ Enforcer │  │ Manager   │  │
│  └─────┬─────┘  └─────┬─────┘  └─────┬────┘  └─────┬─────┘  │
│        │              │              │              │         │
│  ┌─────┴──────────────┴──────────────┴──────────────┴─────┐  │
│  │              Orchestration Layer (vlp_orch)             │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌────────────────────────┴───────────────────────────────┐  │
│  │              Host-Device Bridge (vlp_bridge)            │  │
│  └────────────────────────┬───────────────────────────────┘  │
└───────────────────────────┼──────────────────────────────────┘
                            │ PCIe / NVLink
┌───────────────────────────┼──────────────────────────────────┐
│                     DEVICE (GPU)                             │
│                                                              │
│  ┌────────────────────────┴───────────────────────────────┐  │
│  │              Device Runtime (vlp_device)                │  │
│  └───┬────────┬─────────┬──────────┬──────────┬──────────┘  │
│      │        │         │          │          │              │
│  ┌───┴──┐ ┌──┴───┐ ┌───┴──┐ ┌────┴───┐ ┌───┴──────┐      │
│  │ LLM  │ │ KB   │ │Prolog│ │Grammar │ │ Builtin  │      │
│  │Engine│ │Store │ │Engine│ │Engine  │ │ Executor │      │
│  └──────┘ └──────┘ └──────┘ └────────┘ └──────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              TensorProlog Hardware Layer                     │   │
│  │  Integer ALUs │ KB Cache │ FRU │ Warp Scheduler      │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

Five device-side engines. One host-side orchestration layer. One bridge. All communication is typed integers through declared interfaces.

---

## 3. Core Data Types

### 3.1 System-Wide Type Declarations

```
// ============================================================
// VDR Value Types — the arithmetic foundation
// ============================================================

// Q-basis enum — compile-time declaration, runtime-checked
enum vlp_qbasis: i32 {
    VLP_Q16  = 16,    // D = 65536, inference
    VLP_Q32  = 32,    // D = 4294967296, intermediate
    VLP_Q335 = 335,   // D = 2^335, high precision / transcendental
};

// Q16 value — the primary operational type
struct vlp_q16 {
    v: i32,       // numerator
    r0: i16,      // remainder level 0
    _pad: i16,    // alignment to 8 bytes
};
// D = 65536 is never stored. Implicit. Universal.
// sizeof(vlp_q16) = 8 bytes. Packs cleanly in registers and cache lines.

// Q32 value
struct vlp_q32 {
    v: i64,
    r0: i32,
    r1: i32,
};
// sizeof(vlp_q32) = 16 bytes.

// Q335 value — wide integer, limb-based
struct vlp_q335 {
    v: [6]i64,      // 384-bit value as 6 × 64-bit limbs
    r0: [6]i64,     // remainder level 0
    r1: [6]i64,     // remainder level 1
    r2: [6]i64,     // remainder level 2
    r3: [6]i64,     // remainder level 3
};
// sizeof(vlp_q335) = 240 bytes.
// Depth 4 fixed. Paper says pick your depth, fix the struct.
// r3 tells you exactly what's lost if you don't go deeper.

// ============================================================
// KB Types — the data layer
// ============================================================

// Fact tag — classifies what the fact represents
enum vlp_fact_tag: i32 {
    TAG_VALUE       = 0,   // plain VDR value
    TAG_TEXT        = 1,   // text reference (offset + length in text store)
    TAG_REFERENCE   = 2,   // reference to another kb_id + slot_id
    TAG_TIMESTAMP   = 3,   // i32 unix timestamp
    TAG_ENUM        = 4,   // i32 enum value from declared set
    TAG_BOOLEAN     = 5,   // i32 0 or 1
    TAG_VECTOR      = 6,   // reference to contiguous VDR array
    TAG_MATRIX      = 7,   // reference to VDR array + dimensions
    TAG_PROVENANCE  = 8,   // compound: source + confidence + derivation
    TAG_RULE_REF    = 9,   // reference to rule_id
    TAG_GRAMMAR_REF = 10,  // reference to grammar_id
    TAG_COUNTER     = 11,  // inline counter state
    TAG_EMPTY       = 255, // unoccupied slot
};

// Provenance — tracks where a fact came from
struct vlp_provenance {
    source_type: i32,       // enum from confidence hierarchy (Appendix F)
    source_kb_id: i32,      // which KB produced this
    source_slot_id: i32,    // which slot
    confidence: vlp_q16,    // exact fraction
    timestamp: i32,         // when asserted
    derivation_rule_id: i32,// which Prolog rule derived this (-1 if direct)
};
// sizeof(vlp_provenance) = 28 bytes.

// Fact — the atomic unit of knowledge
struct vlp_fact {
    tag: vlp_fact_tag,           // what kind of value
    value: vlp_q16,              // the value (or offset for text/vector/matrix)
    provenance: vlp_provenance,  // where it came from
};
// sizeof(vlp_fact) = 40 bytes.

// KB struct — the 26-field structure from paper Appendix K
struct vlp_kb {
    // Identity (12 bytes)
    name_offset: i32,        // offset into text store
    name_length: i16,        // byte length
    path_offset: i32,        // offset into text store for dotted path
    path_length: i16,        // byte length
    id: i32,                 // sequential integer ID

    // Persistent state — survives reset (offsets into respective stores)
    facts_offset: i32,       // offset into fact store
    facts_count: i32,        // current count
    facts_capacity: i32,     // max count (set at creation)
    rules_offset: i32,       // offset into rule store
    rules_count: i32,
    rules_capacity: i32,
    constraints_offset: i32, // offset into constraint store
    constraints_count: i32,
    connections_offset: i32, // offset into connection store
    connections_count: i32,
    grammars_offset: i32,    // offset into grammar store
    grammars_count: i32,
    iose_offset: i32,        // offset into IOSE declaration store (-1 if none)

    // Live state — cleared by reset (inline, fixed-size)
    working_data_offset: i32,   // offset into working data store
    lru_table_offset: i32,      // offset into LRU table
    lru_count: i16,
    counter_table_offset: i32,  // offset into counter table
    counter_count: i16,
    lock_table_offset: i32,
    lock_count: i16,
    queue_table_offset: i32,
    queue_count: i16,
    stack_table_offset: i32,
    stack_count: i16,
    ring_table_offset: i32,
    ring_count: i16,
    bitset_table_offset: i32,
    bitset_count: i16,

    // Structural links
    parent_id: i32,             // -1 for root
    children_offset: i32,       // offset into children ID array
    children_count: i16,
    children_capacity: i16,
    mounts_offset: i32,         // offset into mount table
    mounts_count: i16,

    // Metadata
    visibility: i8,             // 0=public, 1=internal, 2=owner_only
    frozen: i8,                 // 0=mutable, 1=frozen
    owner_offset: i32,          // offset into text store for owner name
    owner_length: i16,
    created_at: i32,            // timestamp
    last_modified: i32,         // timestamp
};
// sizeof(vlp_kb) ≈ 148 bytes. Fixed. Every KB same size.
// Alignment to 256 bytes for cache line efficiency → 256 bytes per KB.

// ============================================================
// Prolog Types — the deduction layer
// ============================================================

// Term type enum
enum vlp_term_type: i8 {
    TERM_ATOM       = 0,   // named constant, identified by i32 atom_id
    TERM_VARIABLE   = 1,   // binding slot, identified by i32 var_id
    TERM_INTEGER    = 2,   // i32 integer literal
    TERM_VDR        = 3,   // vlp_q16 value
    TERM_TEXT       = 4,   // text reference
    TERM_LIST       = 5,   // head + tail (Prolog-style linked list)
    TERM_COMPOUND   = 6,   // functor(arg1, arg2, ..., argN)
    TERM_VECTOR     = 7,   // VDR vector reference
    TERM_MATRIX     = 8,   // VDR matrix reference
    TERM_PAIR       = 9,   // key-value pair for dict-like structures
};

// Term — recursive, but bounded by depth limit
struct vlp_term {
    type: vlp_term_type,
    // Union (tagged by type):
    atom_id: i32,           // for ATOM
    var_id: i32,            // for VARIABLE
    int_value: i32,         // for INTEGER
    vdr_value: vlp_q16,     // for VDR
    text_offset: i32,       // for TEXT
    text_length: i16,
    list_head_offset: i32,  // for LIST — offset into term store
    list_tail_offset: i32,
    functor_id: i32,        // for COMPOUND
    args_offset: i32,       // offset into term store
    args_count: i16,
};
// sizeof(vlp_term) = 24 bytes (union packing).

// Rule — head :- body
struct vlp_rule {
    id: i32,
    head: i32,              // offset to head term in term store
    body_offset: i32,       // offset to array of body terms
    body_count: i16,
    action_offset: i32,     // offset to array of action terms (assert/retract)
    action_count: i16,
    // Statistics
    fire_count: i32,
    last_fired: i32,        // timestamp
    success_count: i32,
    failure_count: i32,
    created_at: i32,
    creator_session_id: i32,
};
// sizeof(vlp_rule) = 44 bytes.

// Binding — variable assignment from unification
struct vlp_binding {
    var_id: i32,
    bound_term_offset: i32, // offset into term store
};

// Unification result
struct vlp_unification_result {
    unified: i8,             // 0=failed, 1=succeeded
    bindings_offset: i32,    // offset into binding store
    bindings_count: i16,
};

// ============================================================
// Grammar Types — the structural token layer
// ============================================================

// Slot type enum
enum vlp_slot_type: i8 {
    SLOT_VDR_VALUE  = 0,   // filled with VDR value, rendered as number
    SLOT_TEXT       = 1,   // filled with text
    SLOT_INTEGER    = 2,   // filled with i32
    SLOT_ENUM       = 3,   // filled with one of declared enum values
    SLOT_KB_REF     = 4,   // filled from KB fact at runtime
    SLOT_GRAMMAR    = 5,   // nested grammar (composition)
};

// Slot declaration
struct vlp_grammar_slot {
    name_offset: i32,       // offset into text store
    name_length: i16,
    type: vlp_slot_type,
    enum_values_offset: i32, // for ENUM: offset into enum value list
    enum_count: i16,
    kb_id: i32,             // for KB_REF: default KB to pull from
    kb_slot_id: i32,        // for KB_REF: default slot
};

// Grammar template
struct vlp_grammar {
    id: i32,
    template_offset: i32,   // offset into text store (raw template bytes)
    template_length: i32,
    slots_offset: i32,       // offset into slot declaration array
    slots_count: i16,
    validated: i8,           // 0=unvalidated, 1=valid
    created_at: i32,
    creator_session_id: i32,
};
// sizeof(vlp_grammar) = 28 bytes.

// Grammar fill — runtime value for a slot
struct vlp_grammar_fill {
    slot_index: i16,         // which slot in the grammar
    fill_type: vlp_slot_type,
    // Union:
    vdr_value: vlp_q16,
    text_offset: i32,
    text_length: i16,
    int_value: i32,
    enum_index: i16,
};

// ============================================================
// Session Types — the isolation layer
// ============================================================

enum vlp_session_state: i8 {
    SESSION_ACTIVE      = 0,
    SESSION_SNAPSHOTTED = 1,
    SESSION_KILLED      = 2,
    SESSION_FROZEN      = 3,
};

struct vlp_session {
    id: i32,
    user_id: i32,
    kb_root_id: i32,
    visibility_level: i8,    // 0=public, 1=internal, 2=owner_only
    state: vlp_session_state,

    // Resource bounds
    max_kb_count: i32,
    max_live_memory_bytes: i64,
    max_turns: i32,

    // Counters
    current_turn: i32,
    facts_asserted: i32,
    facts_retracted: i32,
    rules_fired: i64,
    prolog_queries: i64,
    primitive_calls: i64,
    grammar_renders: i64,
    llm_tokens_consumed: i64,
    command_tokens_consumed: i64,

    // Device references
    device_id: i32,
    stream_id: i32,          // primary session stream
    kb_store_offset: i64,    // offset into device global KB store

    // Snapshot reference
    last_snapshot_id: i32,   // -1 if never snapshotted
    last_snapshot_timestamp: i32,

    // Clone lineage
    parent_session_id: i32,  // -1 if original
    clone_generation: i32,   // 0 for originals, increments per clone
};
// sizeof(vlp_session) ≈ 128 bytes.

// ============================================================
// Runner Types — the autonomy layer
// ============================================================

enum vlp_runner_type: i8 {
    RUNNER_POLLER     = 0,   // periodic check, fresh LLM per cycle
    RUNNER_PROCESSOR  = 1,   // persistent connection, continuous ingest
    RUNNER_INTERNAL   = 2,   // scheduled internal computation
    RUNNER_BATCH      = 3,   // task queue consumer, clone-per-task
};

enum vlp_runner_state: i8 {
    RUNNER_STOPPED    = 0,
    RUNNER_RUNNING    = 1,
    RUNNER_ERROR      = 2,
    RUNNER_RECYCLING  = 3,
};

struct vlp_runner {
    id: i32,
    type: vlp_runner_type,
    state: vlp_runner_state,
    session_id: i32,         // owning session

    // Configuration
    interval_ms: i32,        // for POLLER and INTERNAL
    max_turns_before_recycle: i32,  // for PROCESSOR
    max_consecutive_errors: i32,

    // Counters
    iterations_completed: i64,
    errors_consecutive: i32,
    errors_total: i64,
    last_iteration_ms: i32,
    last_iteration_timestamp: i32,

    // Recycle state
    recycle_count: i32,
    last_recycle_timestamp: i32,
};
// sizeof(vlp_runner) = 72 bytes.

// ============================================================
// Grant Types — the safety layer
// ============================================================

enum vlp_grant_class: i8 {
    GRANT_FILESYSTEM  = 0,
    GRANT_COMPILE     = 1,
    GRANT_EXECUTE     = 2,
    GRANT_LINT        = 3,
    GRANT_NETWORK     = 4,
    GRANT_PROCESS     = 5,
};

enum vlp_grant_state: i8 {
    GRANT_ACTIVE    = 0,
    GRANT_EXPIRED   = 1,
    GRANT_EXHAUSTED = 2,
    GRANT_REVOKED   = 3,
};

struct vlp_grant {
    id: i32,
    class: vlp_grant_class,
    state: vlp_grant_state,
    holder_user_id: i32,
    target_pattern_offset: i32,  // offset into text store
    target_pattern_length: i16,
    max_uses: i32,               // -1 = unlimited
    remaining_uses: i32,
    expires_at: i32,             // 0 = never
    created_at: i32,
    created_by: i32,             // admin user_id
    revoked_at: i32,             // 0 = not revoked
    revoked_by: i32,
};
// sizeof(vlp_grant) = 48 bytes.

// ============================================================
// Confidence Types — the provenance layer
// ============================================================

enum vlp_source_type: i8 {
    SOURCE_VDR_COMPUTATION   = 0,   // confidence 1/1
    SOURCE_PROLOG_DERIVATION = 1,   // confidence 1/1
    SOURCE_DATABASE          = 2,   // confidence 98/100
    SOURCE_PROMETHEUS        = 3,   // confidence 95/100
    SOURCE_SCRIPT            = 4,   // confidence 95/100
    SOURCE_REST_API          = 5,   // confidence 85/100
    SOURCE_PUBLISHED         = 6,   // confidence 80/100
    SOURCE_USER_STATED       = 7,   // confidence 70/100
    SOURCE_WEB_SEARCH        = 8,   // confidence 50/100
    SOURCE_LLM_GENERATED     = 9,   // confidence 30/100
    SOURCE_UNKNOWN           = 10,  // confidence 0/1
};

// Default confidence values — exact VDR fractions, immutable
// Stored in seed KB at system initialization
const vlp_q16 CONFIDENCE_TABLE[11] = {
    { .v = 65536, .r0 = 0 },   // 1/1
    { .v = 65536, .r0 = 0 },   // 1/1
    { .v = 64225, .r0 = 0 },   // 98/100 → 64225/65536
    { .v = 62259, .r0 = 0 },   // 95/100
    { .v = 62259, .r0 = 0 },   // 95/100
    { .v = 55705, .r0 = 0 },   // 85/100
    { .v = 52428, .r0 = 0 },   // 80/100
    { .v = 45875, .r0 = 0 },   // 70/100
    { .v = 32768, .r0 = 0 },   // 50/100
    { .v = 19660, .r0 = 0 },   // 30/100
    { .v = 0,     .r0 = 0 },   // 0/1
};

// ============================================================
// Command Token Types — LLM→System interface
// ============================================================

enum vlp_command_type: i8 {
    CMD_KB_ASSERT        = 0,
    CMD_KB_QUERY         = 1,
    CMD_KB_RETRACT       = 2,
    CMD_PROLOG_QUERY     = 3,
    CMD_PROLOG_ASSERT_RULE = 4,
    CMD_BUILTIN_CALL     = 5,
    CMD_GRAMMAR_RENDER   = 6,
    CMD_DIRECT_OUTPUT    = 7,
    CMD_OP_FILESYSTEM    = 8,
    CMD_OP_COMPILE       = 9,
    CMD_OP_EXECUTE       = 10,
    CMD_OP_NETWORK       = 11,
    CMD_OP_PROCESS       = 12,
    CMD_SESSION_SNAPSHOT  = 13,
    CMD_SESSION_CLONE    = 14,
};

struct vlp_command {
    type: vlp_command_type,
    target_kb_id: i32,       // which KB (resolved from dotted path)
    target_slot_id: i32,     // which slot (for assert/query/retract)
    builtin_id: i32,         // which builtin (for BUILTIN_CALL)
    args_offset: i32,        // offset into command argument buffer
    args_count: i16,
    grant_required: vlp_grant_class,  // which grant class needed (-1 if none)
};
// sizeof(vlp_command) = 24 bytes.
// ~8 LLM tokens to generate. Low entropy — small vocabulary selection.

// ============================================================
// Audit Types — the accountability layer
// ============================================================

enum vlp_audit_action: i8 {
    AUDIT_FACT_ASSERT    = 0,
    AUDIT_FACT_RETRACT   = 1,
    AUDIT_FACT_QUERY     = 2,
    AUDIT_RULE_FIRE      = 3,
    AUDIT_RULE_ASSERT    = 4,
    AUDIT_RULE_RETRACT   = 5,
    AUDIT_GRANT_CHECK    = 6,
    AUDIT_GRANT_CREATE   = 7,
    AUDIT_GRANT_REVOKE   = 8,
    AUDIT_SESSION_CREATE = 9,
    AUDIT_SESSION_KILL   = 10,
    AUDIT_SNAPSHOT       = 11,
    AUDIT_CLONE          = 12,
    AUDIT_OP_EXECUTE     = 13,
    AUDIT_ACCESS_DENIED  = 14,
};

struct vlp_audit_entry {
    timestamp: i32,
    session_id: i32,
    user_id: i32,
    action: vlp_audit_action,
    target_kb_id: i32,
    target_slot_id: i32,
    grant_id: i32,           // -1 if no grant involved
    result: i8,              // 0=denied, 1=allowed
    detail_offset: i32,      // offset into audit detail store for additional context
};
// sizeof(vlp_audit_entry) = 28 bytes.
```

---

## 4. Device Memory Layout

### 4.1 Global Memory Regions

The device global memory is partitioned into fixed regions at initialization.

```
struct vlp_device_memory_layout {
    // ============ Model Weights Region ============
    // Read-only after load. Shared across all sessions.
    model_weights_base: i64,        // byte offset
    model_weights_size: i64,        // bytes allocated
    // Contains: embedding table, layer weights (QKV, output proj, MLP up/down,
    // layer norm gamma/beta, lm_head). All vlp_q16 arrays.
    // Layout within region:
    //   [0..vocab_size*d_model)                    = embedding table
    //   [embedding_end..embedding_end+layer_size)  = layer 0
    //   ... repeated for n_layers ...
    //   [layers_end..layers_end+vocab_size*d_model) = lm_head
    // Each layer_size = QKV_weights + out_proj + MLP_up + MLP_down +
    //                   ln1_gamma + ln1_beta + ln2_gamma + ln2_beta

    // ============ KB Store Region ============
    // Primary data store. Partitioned per session with COW for clones.
    kb_store_base: i64,
    kb_store_size: i64,
    // Contains: array of vlp_kb structs (256 bytes each, contiguous)
    //   [0..max_total_kbs * 256)
    kb_count: i32,
    kb_capacity: i32,

    // ============ Fact Store Region ============
    // All facts across all KBs. Indexed by kb.facts_offset + slot_id.
    fact_store_base: i64,
    fact_store_size: i64,
    // Contains: array of vlp_fact structs (40 bytes each)
    // Each KB owns a contiguous chunk of slots.

    // ============ Rule Store Region ============
    rule_store_base: i64,
    rule_store_size: i64,
    // Contains: array of vlp_rule structs + associated term store

    // ============ Term Store Region ============
    // Prolog terms for rules, queries, and unification results.
    term_store_base: i64,
    term_store_size: i64,
    // Contains: array of vlp_term structs (24 bytes each)
    // Rules and queries reference terms by offset into this store.

    // ============ Text Store Region ============
    // All text data: KB names, paths, owner names, grammar templates,
    // text fact values, grant target patterns.
    text_store_base: i64,
    text_store_size: i64,
    // Contains: raw bytes. Referenced by offset + length pairs.
    // Append-only during session. Compacted during snapshot.

    // ============ Grammar Store Region ============
    grammar_store_base: i64,
    grammar_store_size: i64,
    // Contains: array of vlp_grammar structs + slot declarations

    // ============ Live State Region ============
    // Per-session mutable state: LRUs, counters, queues, stacks, rings, bitsets.
    live_state_base: i64,
    live_state_size: i64,
    // Contains: tables of data primitives, indexed from vlp_kb fields.
    // Each primitive type has a fixed-size entry:
    //   LRU: header (16 bytes) + entries (key i32 + vlp_fact 40 bytes each)
    //   Counter: 16 bytes (value, min, max, initial)
    //   Lock: 4 bytes (state)
    //   Queue: header (12 bytes) + entries (vlp_fact 40 bytes each)
    //   Stack: header (12 bytes) + entries (vlp_fact 40 bytes each)
    //   Ring: header (16 bytes) + entries (vlp_fact 40 bytes each)
    //   Bitset: header (8 bytes) + bit array (ceil(n_bits/8) bytes)

    // ============ Scratch Region ============
    // Temporary buffers for forward pass, Prolog queries, grammar rendering.
    // Per-stream allocation. Recycled between operations.
    scratch_base: i64,
    scratch_size: i64,

    // ============ Audit Region ============
    // Ring buffer of vlp_audit_entry. Fixed size, oldest overwritten.
    audit_base: i64,
    audit_size: i64,
    audit_capacity: i32,     // max entries
    audit_head: i32,         // write position (atomic)

    // ============ Grant Store Region ============
    grant_store_base: i64,
    grant_store_size: i64,
    // Contains: array of vlp_grant structs (48 bytes each)

    // ============ Session Table Region ============
    session_table_base: i64,
    session_table_size: i64,
    // Contains: array of vlp_session structs
    session_count: i32,
    session_capacity: i32,

    // ============ Path Index Region ============
    // Hash map: dotted path string → kb_id
    path_index_base: i64,
    path_index_size: i64,
    // Open addressing, i32 keys (path hash) → i32 values (kb_id)
    // Collision resolution: linear probe. Load factor < 0.7.
};
```

### 4.2 Memory Sizing

```
// Reference configuration for 7B model, 10K concurrent sessions

model_weights:  7B params × 8 bytes (vlp_q16) = 56 GB
                → Requires multi-GPU. 80GB HBM3 per device, model-parallel.

kb_store:       100K KBs × 256 bytes = 25.6 MB
fact_store:     10M facts × 40 bytes = 400 MB
rule_store:     100K rules × 44 bytes + terms = ~50 MB
term_store:     1M terms × 24 bytes = 24 MB
text_store:     ~100 MB (names, paths, templates, text values)
grammar_store:  10K grammars × 28 bytes + slots = ~5 MB
live_state:     Per session: ~50 KB typical (bounded primitives)
                10K sessions × 50 KB = 500 MB
scratch:        Per stream: 1-10 MB depending on operation
                100 concurrent streams × 10 MB = 1 GB
audit:          1M entries × 28 bytes = 28 MB
grant_store:    100K grants × 48 bytes = 4.8 MB
session_table:  10K sessions × 128 bytes = 1.28 MB
path_index:     200K entries × 8 bytes = 1.6 MB

Total non-model: ~2.2 GB
Total with model: ~58 GB (fits in single H100 80GB for model-parallel shard)

// Key insight: the non-model memory is tiny relative to model weights.
// The entire KB/Prolog/grammar/session infrastructure fits in ~2 GB.
// Model weights dominate, same as conventional LLMs.
// The infrastructure overhead is negligible.
```

### 4.3 Shared Memory Usage (Per SM)

```
// H100: 228 KB shared memory per SM (configurable L1/shared split)

struct vlp_sm_shared_layout {
    // KB cache — primary use of shared memory
    kb_cache: [16]vlp_kb,           // 16 KBs × 256 bytes = 4 KB
    fact_cache: [512]vlp_fact,      // 512 facts × 40 bytes = 20 KB
    // Covers: active KB subtree for current Prolog query or builtin

    // Term scratch — for unification
    term_scratch: [256]vlp_term,    // 256 terms × 24 bytes = 6 KB

    // Softmax scratch — for attention
    softmax_scratch: [4096]vlp_q16, // 4096 × 8 bytes = 32 KB
    // Covers: one head's attention scores for seq_len up to 4096

    // Binding scratch — for Prolog results
    binding_scratch: [64]vlp_binding, // 64 bindings × 8 bytes = 512 bytes

    // Grammar render buffer
    render_buffer: [4096]u8,        // 4 KB output buffer

    // Total: ~67 KB. Well within 228 KB budget.
    // Remaining ~161 KB available for larger fact caches or longer sequences.
};
```

---

## 5. Orchestration Layer (vlp_orch)

Host-side system that manages sessions, runners, and the LLM-to-system interface.

### 5.1 Session Manager

```
vlp_session_manager_init(config: *vlp_system_config) -> vlp_status
  System-wide initialization. Allocates device memory layout.
  Creates root KB with seed layers (~23,400 entries, ~1.5 MB).
  Seed layers contain: sentence templates, format grammars, operational
  rules, self-maintenance rules, 15 engineering principles as ~176
  Prolog terms at root.system.oso. Loaded from seed snapshot file.
  Initializes path index hash map. Starts audit ring buffer.
  Side effects: device memory allocated, seed KBs populated.

vlp_session_create(config: *vlp_session_config) -> vlp_session_handle
  Allocates session in session table. Creates session-bound TensorProlog stream.
  Initializes session KB subtree with root pointing to config.kb_root_id.
  Allocates live state region for session's bounded primitives.
  Sets up COW (copy-on-write) page table if session is a clone.
  Returns opaque handle used by all subsequent session operations.
  Side effects: session table entry, TensorProlog stream, KB allocations.

vlp_session_destroy(handle: vlp_session_handle) -> vlp_status
  If auto_snapshot configured: calls vlp_session_snapshot first.
  Frees session's live state region. Frees session's COW pages.
  Marks session table entry as available. Destroys TensorProlog stream.
  Persistent KB facts written during session remain in fact store
  (unless session was COW and changes weren't merged).
  Side effects: memory freed, stream destroyed, audit entry written.

vlp_session_snapshot(handle: vlp_session_handle) -> vlp_snapshot_handle
  Atomic capture:
  1. Pause all runners owned by this session (wait for current iteration).
  2. Fence all pending TensorProlog operations on session stream.
  3. Copy session's KB subtree (persistent + live) to host memory.
     For COW sessions: resolve all COW pages to concrete copies.
  4. Copy session metadata (counters, runner states).
  5. Compute integer CRC32 checksum over entire snapshot.
  6. Resume runners.
  Snapshot is a host memory blob. Portable across devices.
  Side effects: host memory allocated, audit entry, runners briefly paused.

vlp_session_restore(handle: vlp_session_handle, snapshot: vlp_snapshot_handle) -> vlp_status
  Validate checksum. If mismatch → ERR_SNAPSHOT_CORRUPT (never silent).
  Overwrite session's device state from snapshot:
  1. KB structs, fact store region, rule store, term store.
  2. Live state: LRUs, counters, queues, stacks, rings, bitsets.
  3. Grammar store.
  4. Session metadata (counters reset to snapshot values).
  After restore: session is in exactly the state at snapshot time.
  Bit-identical. Because integers.
  Side effects: device memory overwritten, audit entry.

vlp_session_clone(parent: vlp_session_handle, config: *vlp_clone_config) -> vlp_session_handle
  Creates new session sharing parent's persistent KBs via COW.
  COW implementation: clone's KB page table points to parent's pages.
  First write to a shared page triggers copy of that page to clone's
  private region. Subsequent writes go to the private copy.
  If config.fresh_live = true: clone's live state initialized to empty.
  If config.fresh_live = false: clone's live state copied from parent's current.
  If config.inherit_rules = true: clone sees parent's rules.
  Clone generation = parent.clone_generation + 1.
  Side effects: session created, COW page table allocated, audit entry.

vlp_session_merge(parent: vlp_session_handle, child: vlp_session_handle, policy: vlp_merge_policy) -> vlp_status
  Merges child's COW-dirtied pages back to parent.
  Policy OURS: parent's version wins on conflict. THEIRS: child wins.
  FAIL_ON_CONFLICT: returns error listing conflicting kb_id + slot_id pairs.
  Conflict detection: compare timestamps on both versions. If both modified
  since clone creation, it's a conflict.
  Side effects: parent's KB state modified, audit entries for each merged fact.

vlp_session_kill(handle: vlp_session_handle) -> vlp_status
  Immediate termination. No snapshot. All runners force-stopped.
  COW pages discarded without merge. Live state freed.
  This is the drift-kill. Fast and violent. Knowledge in parent persists.
  Drift in clone dies.
  Side effects: all resources freed immediately, audit entry.
```

### 5.2 Runner Scheduler

```
vlp_runner_scheduler_init(max_runners: i32) -> vlp_status
  Allocates runner table on host. Creates runner management thread pool.
  Thread pool size = min(max_runners, hardware_threads / 2).
  Side effects: host threads created.

vlp_runner_create_poller(config: *vlp_poller_config) -> vlp_runner_handle
  Allocates runner in runner table. Does not start.
  Config specifies:
    session: vlp_session_handle — runner executes within this session's scope
    interval_ms: i32 — poll frequency
    poll_fn: callback — the function called each cycle
    max_consecutive_errors: i32 — auto-stop threshold
  poll_fn signature: (session: vlp_session_handle) -> vlp_status
  Each invocation gets a fresh TensorProlog stream on the session.
  The LLM context is fresh each cycle — no attention degradation.
  Side effects: runner table entry allocated.

vlp_runner_create_processor(config: *vlp_processor_config) -> vlp_runner_handle
  Config specifies:
    session: vlp_session_handle
    source_config: *vlp_source_config — connection parameters
    ingest_fn: callback — called per incoming data item
    compact_fn: callback — transforms raw data into KB facts
    max_turns_before_recycle: i32 — default 200
  Processor lifecycle:
    1. Establish connection to external source.
    2. Loop: receive data → ingest_fn → compact_fn → KB assert.
    3. At max_turns: snapshot connection state, kill session, clone from
       snapshot, restore connection from saved state. Runner continues
       with fresh session. Accumulated KB facts preserved.
    4. On connection loss: retry with exponential backoff.
       Backoff intervals: 1s, 2s, 4s, 8s, 16s, 32s, 60s (cap).
  Side effects: runner table entry, network connection initiated on start.

vlp_runner_create_internal(config: *vlp_internal_config) -> vlp_runner_handle
  Config specifies:
    session: vlp_session_handle
    interval_ms: i32
    compute_fn: callback — internal computation
  compute_fn: derives facts from existing facts. Rolling averages,
  trend detection, coverage gap analysis. No external connections.
  All computation on KB data via exact VDR arithmetic.
  Side effects: runner table entry.

vlp_runner_create_batch(config: *vlp_batch_config) -> vlp_runner_handle
  Config specifies:
    session: vlp_session_handle
    task_queue_kb_id: i32 — KB containing task queue
    task_queue_name: text — name of the queue primitive
    process_fn: callback — processes one task
    max_concurrent: i32 — max simultaneous task clones
  Batch lifecycle:
    1. Pop task from queue.
    2. Clone session for isolation.
    3. Call process_fn(clone_session, task).
    4. Merge clone results back to parent.
    5. Kill clone.
    6. Repeat.
  Up to max_concurrent clones active simultaneously.
  Side effects: runner table entry, clones created/killed during operation.

vlp_runner_start(handle: vlp_runner_handle) -> vlp_status
  Begins execution. Assigns runner to thread pool thread.
  Sets state to RUNNING. Returns immediately (non-blocking).
  Side effects: thread pool task scheduled, runner state updated.

vlp_runner_stop(handle: vlp_runner_handle) -> vlp_status
  Signals stop. Current iteration completes. Blocks until stopped.
  Sets state to STOPPED.
  Side effects: runner state updated, audit entry.

vlp_runner_kill(handle: vlp_runner_handle) -> vlp_status
  Immediate stop. Current iteration abandoned.
  If processor: connection dropped without cleanup.
  Sets state to STOPPED.
  Side effects: runner state updated, possible connection loss.

vlp_runner_recycle(handle: vlp_runner_handle) -> vlp_status
  Manual recycle for processors:
    1. Pause runner.
    2. Snapshot session.
    3. Kill session.
    4. Clone from snapshot.
    5. Assign clone to runner.
    6. Resume runner.
  Turn counter resets. Live state is fresh. Knowledge persists.
  Side effects: old session killed, new session created, audit entries.

vlp_runner_get_status(handle: vlp_runner_handle) -> vlp_runner_status
  Returns current state, iteration counts, error counts, timing.
  All integer values. No approximate metrics.
```

### 5.3 Grant Enforcer

```
vlp_grant_enforcer_init(grant_store: *vlp_grant_store) -> vlp_status
  Initializes grant checking infrastructure. Builds index on
  (holder_user_id, grant_class) for fast lookup.
  Side effects: index built in device memory.

vlp_grant_check(session: vlp_session_handle, grant_class: vlp_grant_class, target: *const u8, target_len: i32) -> vlp_grant_result
  The critical path for operational primitives.
  Flow:
    1. Extract user_id from session. (Integer read.)
    2. Look up grants for (user_id, grant_class) in index. (Integer comparison.)
    3. For each matching grant:
       a. Check state == ACTIVE. (Integer comparison.)
       b. Check not expired: expires_at == 0 OR current_time < expires_at. (Integer comparison.)
       c. Check uses remaining: remaining_uses == -1 OR remaining_uses > 0. (Integer comparison.)
       d. Check target pattern matches. (Prefix match on integer-indexed text.)
    4. If match found:
       a. If remaining_uses > 0: decrement. (Atomic integer decrement.)
       b. Write audit entry: AUDIT_GRANT_CHECK, result=ALLOWED.
       c. Return GRANTED with grant_id.
    5. If no match:
       a. Write audit entry: AUDIT_GRANT_CHECK, result=DENIED.
       b. Return DENIED.
  Every step is integer comparison. No heuristics. No probability.
  No text evaluation. No LLM judgment.
  Side effects: remaining_uses decremented if granted, audit entry written.

vlp_grant_create(admin_session: vlp_session_handle, grant: *vlp_grant) -> vlp_status
  Precondition: admin_session's user_id must hold GRANT_ADMIN privilege.
  GRANT_ADMIN is a meta-grant that can only be created during system
  initialization and cannot be created by any runtime operation.
  Validates: grant_class, holder_user_id, target_pattern syntax.
  Allocates grant in grant store. Sets state = ACTIVE.
  Side effects: grant store entry, audit entry.

vlp_grant_revoke(admin_session: vlp_session_handle, grant_id: i32) -> vlp_status
  Sets grant state = REVOKED. Records revoked_at and revoked_by.
  Permanent. No unrevoke.
  Side effects: grant state updated, audit entry.

vlp_grant_list(user_id: i32, grants: *vlp_grant, max: i32, count: *i32) -> vlp_status
  Returns all grants (active and historical) for user_id.
  For audit review.

vlp_grant_cleanup() -> vlp_status
  Scans grant store. Marks expired grants (current_time > expires_at)
  as EXPIRED. Marks exhausted grants (remaining_uses == 0) as EXHAUSTED.
  Run periodically by internal runner.
  Side effects: grant states updated.
```

### 5.4 Command Processor

The interface between LLM output and system operations.

```
vlp_command_parse(tokens: *i32, n_tokens: i32, command: *vlp_command) -> vlp_status
  Parses LLM-generated command tokens into a vlp_command struct.
  Command token vocabulary: ~300 known names.
  Expected format: COMMAND_TYPE target_path [args...]
  Examples:
    KB_ASSERT root.ops.incidents.inc_001 fact(service, checkout_api)
    KB_QUERY root.ops.metrics.health
    BUILTIN list_sort root.ops.data.items
    OP_FN net_fetch https://prometheus.internal/...
    DIRECT_OUTPUT kb://root.ops.incidents.inc_001.triage_summary
  Parse steps:
    1. Match first token against command type enum. (~15 options, ~4 bits entropy.)
    2. Resolve dotted path to kb_id via path index. (Hash lookup.)
    3. Parse remaining tokens as typed arguments. (~2 bits entropy per token.)
  Failure modes: unknown command type, unresolvable path, type mismatch.
  All detectable. All produce specific error codes, not silent failures.
  Side effects: none (pure parse).

vlp_command_execute(session: vlp_session_handle, command: *vlp_command) -> vlp_command_result
  Executes parsed command within session scope.
  Flow:
    1. Access check: vlp_access_check(session, command.target_kb_id).
       If denied → return ACCESS_DENIED immediately. No data touched.

    2. Grant check (if operational command): vlp_grant_check(session, command.grant_required, target).
       If denied → return GRANT_DENIED immediately. No operation performed.

    3. Dispatch by command type:

       CMD_KB_ASSERT:
         Validate fact type against KB constraints.
         Call TensorPrologKBFactAssert. Write provenance with session's user_id,
         current timestamp, and LLM_GENERATED confidence (30/100).
         Update KB.last_modified.
         Return: slot_id of asserted fact.

       CMD_KB_QUERY:
         Call TensorPrologKBFactQuery or TensorPrologKBFactScopedSearch.
         Return: matching facts (copied to session scratchpad for LLM inspection).
         The LLM reads the scratchpad — a few tokens. Not the raw data.

       CMD_KB_RETRACT:
         Call TensorPrologKBFactRetract. Write audit entry.
         Return: success/not_found.

       CMD_PROLOG_QUERY:
         Parse query terms. Call TensorPrologPrologQuery.
         Return: matching bindings (copied to scratchpad).

       CMD_PROLOG_ASSERT_RULE:
         Parse rule head, body, actions. Validate term types.
         Call TensorPrologPrologRuleAssert. This is the LLM formalizing judgment
         as a deterministic rule. Cost: ~25-40 LLM tokens for the command.
         Value: fires at zero LLM cost on every future match.
         Return: rule_id.

       CMD_BUILTIN_CALL:
         Look up builtin by id. Validate argument types.
         Execute builtin on device (dispatch to appropriate TensorProlog kernel).
         Result lands at KB address, not in token stream.
         Return: result_kb_id + result_slot_id.

       CMD_GRAMMAR_RENDER:
         Load grammar from KB. Populate fills from KB facts or command args.
         Call TensorPrologGrammarRender. Output bytes go to session output buffer.
         Every structural byte from grammar. Zero LLM forward passes for structure.
         Return: rendered byte count.

       CMD_DIRECT_OUTPUT:
         Resolve kb:// URL to kb_id + slot_id.
         Load data from KB. Format through associated grammar (if any).
         Inject into output stream. The LLM emitted the reference,
         not the content. Data never entered the token stream.
         Return: output byte count.

       CMD_OP_*:
         Grant already checked in step 2.
         Execute operational primitive in sandboxed environment.
         CMD_OP_FILESYSTEM: read/write/append/delete via host filesystem API.
         CMD_OP_EXECUTE: run script in Docker sandbox, capture output to KB.
         CMD_OP_NETWORK: HTTP fetch, result parsed and stored to KB.
         All outputs go to KB addresses, not token stream.
         Return: operation-specific result.

    4. Write audit entry for the executed command.
    5. Increment session counters (command_tokens_consumed, etc).

  Side effects: vary by command type. All side effects are to KB store,
  audit log, or external systems via grant-gated operations.

vlp_command_batch_execute(session: vlp_session_handle, commands: *vlp_command, n_commands: i32, results: *vlp_command_result) -> vlp_status
  Executes multiple commands in sequence. Aborts on first failure
  unless continue_on_error flag is set in session config.
  Used for multi-step LLM outputs that emit several commands per turn.
  Side effects: cumulative of all executed commands.
```

### 5.5 Access Control

```
vlp_access_check(session: vlp_session_handle, kb_id: i32) -> bool
  The gate that makes data absent, not redacted.
  Algorithm:
    1. Load KB at kb_id. Read visibility field. (Integer.)
    2. If visibility == PUBLIC: return true.
    3. If visibility == INTERNAL: check session.visibility_level <= INTERNAL.
    4. If visibility == OWNER_ONLY: check session.user_id == kb.owner.
       (Owner stored as text, but owner_id is resolved to integer at creation.)
    5. Walk parent chain: load kb.parent_id, repeat check.
       If any ancestor fails visibility check, return false.
       (An OWNER_ONLY KB inside a PUBLIC tree is still OWNER_ONLY.
        A PUBLIC KB inside an OWNER_ONLY tree is invisible to non-owners
        because the ancestor is unreachable.)
    6. Check mount points: if KB is accessed via mount, also check
       mount source visibility.
  Every step is integer load + integer compare.
  No string matching. No regex. No LLM evaluation.
  No prompt can modify any integer in this check.
  The LLM doesn't execute this check — the system does.
  Side effects: none (pure check). Audit entry written by caller if denied.

vlp_visibility_resolve(session: vlp_session_handle, scope_kb_id: i32, visible_kbs: *i32, max_kbs: i32, n_visible: *i32) -> vlp_status
  Enumerates all KBs visible from scope_kb_id for this session.
  Used to build the session's "view" — everything the LLM can see.
  Walks the KB tree from scope_kb_id, applies access_check at each node.
  Prunes subtrees when a parent fails (children inherit parent's restriction).
  Side effects: none.
```

---

## 6. Device Engines

### 6.1 LLM Engine

```
vlp_llm_engine_init(model_config: *vlp_model_config) -> vlp_status
  Loads model weights from checkpoint into model_weights region.
  Config: n_layers, d_model, n_heads, d_head, vocab_size, mlp_dim,
  qbasis, checkpoint_path.
  Validates checksum of loaded weights.
  Precomputes attention scale factor as exact VDR fraction.
  Side effects: model_weights region populated.

vlp_llm_forward(session: vlp_session_handle, input_ids: *i32, n_tokens: i32, logits: *vlp_q16, stream: TensorPrologStream_t) -> vlp_status
  Full forward pass through the model.
  Dispatches the GEMM/attention/layernorm/softmax sequence from
  TensorProlog Cookbook 1.2. All on the session's stream.
  KV-cache managed in session's KB (see 6.1.1).
  Output: logits for each input position, vlp_q16 array.
  Side effects: KV-cache KB updated with new K,V entries.

vlp_llm_generate_token(session: vlp_session_handle, sampling_config: *vlp_sampling_config) -> i32
  Single autoregressive step:
    1. Forward pass for last position.
    2. Softmax over vocabulary (exact, sum = D).
    3. Sampling:
       GREEDY: argmax. (Integer comparison scan.)
       TOP_K: sort top K by exact comparison, sample from normalized sub-distribution.
       TOP_P: accumulate sorted probabilities (exact sum) until reaching P threshold.
       TEMPERATURE: divide logits by temperature (exact VDR division) before softmax.
    4. Return token_id.
  Side effects: KV-cache updated, session.current_turn incremented.

vlp_llm_generate_command(session: vlp_session_handle) -> vlp_command
  Specialized generation for command tokens:
    1. Constrain vocabulary to command token vocabulary (~300 names).
       Mask all other tokens to zero probability before softmax.
    2. Generate tokens until command is complete (detected by end marker).
    3. Parse generated tokens into vlp_command struct.
    4. Return parsed command.
  ~8 tokens generated per command. ~2 bits entropy per token.
  ~99.2% reliability because of constrained vocabulary.
  Side effects: KV-cache updated.

vlp_llm_generate_prose(session: vlp_session_handle, max_tokens: i32, output: *i32, n_generated: *i32) -> vlp_status
  Unconstrained generation for judgment and framing.
  This is the only generation mode where the full vocabulary is active.
  The LLM exercises judgment — decides what to say, how to frame it,
  what matters. Everything else (data, structure, computation) is
  handled by other engines.
  Side effects: KV-cache updated, tokens written to output buffer.
```

#### 6.1.1 KV-Cache Management

```
vlp_kv_cache_init(session: vlp_session_handle, config: *vlp_kv_config) -> vlp_status
  Creates KV-cache KB in session's KB subtree.
  Config: max_seq_len, n_layers, n_heads, d_head, qbasis.
  Allocates fact slots: n_layers × max_seq_len × n_heads × 2 (K and V).
  Each K/V entry is a vlp_fact containing a vector reference.
  Total slots: e.g., 32 layers × 4096 positions × 32 heads × 2 = 8,388,608.
  At 40 bytes per fact = ~336 MB. Large but bounded and declared.
  Side effects: KB created, fact slots allocated.

vlp_kv_cache_append(session: vlp_session_handle, layer: i32, position: i32, K: *vlp_q16, V: *vlp_q16) -> vlp_status
  Stores K and V vectors for one layer at one position.
  O(1) write: computed slot_id = layer × max_seq × heads × 2 + position × heads × 2 + head_offset.
  Direct integer index into fact store. No search.
  Side effects: fact store entries written.

vlp_kv_cache_load(session: vlp_session_handle, layer: i32, start_pos: i32, end_pos: i32, K_out: *vlp_q16, V_out: *vlp_q16) -> vlp_status
  Loads K and V for position range. Contiguous read from fact store.
  Coalesced memory access pattern — sequential integer offsets.
  Side effects: none (read-only).

vlp_kv_cache_truncate(session: vlp_session_handle, position: i32) -> vlp_status
  Marks all entries beyond position as TAG_EMPTY. For beam search
  rollback or context window management.
  Side effects: fact tags updated.

// KV-cache is in the KB. Therefore:
// - Snapshot includes KV-cache. Restore continues from exact prior state.
// - Clone shares parent's KV-cache via COW. Branch generation without copy.
// - Kill clone → KV-cache gone. No leak.
// - Visible only to owning session. Other sessions can't read your cache.
```

### 6.2 KB Store Engine

```
vlp_kb_store_init(layout: *vlp_device_memory_layout) -> vlp_status
  Initializes KB store, fact store, text store, path index from layout.
  Zeroes all stores. Sets all fact slots to TAG_EMPTY.
  Builds empty path index hash map.
  Side effects: device memory zeroed and initialized.

vlp_kb_store_create_kb(config: *vlp_kb_create_config) -> i32
  Allocates next available KB struct in KB store.
  Assigns sequential id. Populates identity fields.
  Allocates fact slots in fact store (contiguous chunk of config.max_facts).
  Adds to parent's children list.
  Registers dotted path in path index.
  Returns kb_id.
  Side effects: KB store, fact store, path index, parent KB updated.

vlp_kb_store_fact_write(kb_id: i32, slot_id: i32, fact: *vlp_fact) -> vlp_status
  Validates:
    - kb_id in range (integer comparison)
    - slot_id < kb.facts_capacity (integer comparison)
    - kb.frozen == false (integer comparison)
  Computes physical offset: kb.facts_offset + slot_id.
  Writes vlp_fact struct to fact store at computed offset.
  Updates kb.last_modified.
  All device-side. No host round-trip for the write itself.
  Side effects: fact store entry, KB metadata.

vlp_kb_store_fact_read(kb_id: i32, slot_id: i32, fact: *vlp_fact) -> vlp_status
  Computes physical offset. Reads vlp_fact. O(1).
  No search. No index. Direct integer arithmetic to memory address.
  Side effects: none.

vlp_kb_store_scoped_search(start_kb_id: i32, tag: vlp_fact_tag, max_depth: i32) -> vlp_search_result
  Walk from start_kb_id up parent chain. At each KB:
    Scan facts for matching tag.
    If found: return facts + kb_id where found.
    If not found: move to parent_id.
  Max_depth bounds the walk (default: 100, matching Prolog depth limit).
  This is lexical scoping — resolution depends on starting position.
  Side effects: none.

vlp_kb_store_cow_init(parent_session: vlp_session_handle, clone_session: vlp_session_handle) -> vlp_status
  Sets up COW page table for clone. Page size: 4 KB (one page = ~100 facts).
  Clone's page table points to parent's pages. Marked read-only.
  On first write to a page: trap, copy page to clone's private region,
  update page table, retry write.
  Implementation: device-side page fault handler using CUDA unified memory
  access faults or explicit software COW with per-page dirty bits.
  Side effects: page table allocated.

vlp_kb_store_cow_resolve(clone_session: vlp_session_handle) -> vlp_status
  Copies all dirty pages from parent to clone's private region.
  After resolve, clone is fully independent — no shared pages.
  Used before snapshot to ensure snapshot is self-contained.
  Side effects: memory copies, page table updated.
```

### 6.3 Prolog Engine

```
vlp_prolog_engine_init(config: *vlp_prolog_config) -> vlp_status
  Config: max_depth (default 100), max_bindings (default 1000),
  max_results_per_query (default 100).
  Allocates term scratch and binding scratch in shared memory.
  Side effects: shared memory reserved.

vlp_prolog_unify(a: *vlp_term, b: *vlp_term, bindings: *vlp_binding, max_bindings: i32, n_bindings: *i32) -> bool
  Core unification algorithm. Recursive with depth bound.
  Cases:
    ATOM-ATOM: succeed if atom_id equal. (Integer comparison.)
    VARIABLE-anything: bind variable to term. Check occurs check.
    VDR-VDR: cross-multiply comparison.
      a.v * b_D vs b.v * a_D. (Integer multiply + integer compare.)
      Exact. No tolerance.
    COMPOUND-COMPOUND: functors must match (integer comparison).
      Then unify each arg pair recursively.
    LIST-LIST: unify heads, then unify tails recursively.
    Anything else: fail.
  No float comparison anywhere. Every comparison is integer equality or
  integer cross-multiply.
  Side effects: bindings written.

vlp_prolog_query(kb_store: *vlp_kb_store, start_kb_id: i32, query: *vlp_term, config: *vlp_query_config) -> vlp_query_result
  Depth-first search with backtracking.
  Flow:
    1. Load candidate facts into shared memory KB cache.
    2. For each fact: attempt unification with query.
       Parallelized: one warp per candidate fact.
       Cross-multiply comparison runs simultaneously across 32 facts.
    3. For matching facts: check if any body conditions remain.
       If body conditions: recursive query on each condition.
    4. Collect bindings for successful unifications.
    5. Apply depth limit. If exceeded: return with partial results + depth flag.
  GPU parallelism model:
    - Frontier-based: current unification candidates form the frontier.
    - Each frontier step processed in parallel across warps.
    - Backtracking is sequential within a search branch but branches
      are independent and can be explored in parallel.
  Side effects: term scratch and binding scratch used.

vlp_prolog_fire_rules(kb_store: *vlp_kb_store, kb_id: i32) -> vlp_fire_result
  Evaluates all rules in kb_id against current fact state.
  For each rule:
    1. Load rule head and body.
    2. Query body conditions against fact store.
    3. If all conditions satisfied: rule fires.
    4. Record fired rule_id, resulting bindings, and proposed actions.
  Does NOT apply actions. Returns list of (rule_id, bindings, actions).
  Caller decides whether to commit.
  Rule statistics updated: fire_count++, last_fired = now.
  Side effects: rule statistics updated.

vlp_prolog_apply_actions(kb_store: *vlp_kb_store, actions: *vlp_prolog_action, n_actions: i32) -> vlp_status
  Applies assert/retract actions from fired rules.
  Each action is:
    ASSERT: write new fact to KB with provenance = PROLOG_DERIVATION (confidence 1/1).
    RETRACT: remove matching fact from KB.
  Side effects: fact store modified, audit entries.

vlp_prolog_fire_and_commit(kb_store: *vlp_kb_store, kb_id: i32) -> i32
  fire_rules + apply_actions in one call. Returns n_fired.
  Convenience for polling runners where review isn't needed.
  Side effects: facts modified, rules stats updated, audit entries.
```

### 6.4 Grammar Engine

```
vlp_grammar_engine_init() -> vlp_status
  Minimal initialization. Grammar engine is stateless — all state
  is in the grammar structs stored in KBs.
  Side effects: none.

vlp_grammar_compile(template: *u8, template_len: i32, grammar: *vlp_grammar) -> vlp_status
  Parses template string. Extracts slot declarations.
  Validates:
    - All braces matched.
    - All slot names unique.
    - All slot types valid.
    - For ENUM slots: all declared values are valid identifiers.
    - Template is syntactically valid with any valid slot fill.
  Sets grammar.validated = 1 on success.
  Compilation output: slot offset table (where each slot appears in template),
  literal byte ranges (everything between slots).
  Side effects: grammar struct populated.

vlp_grammar_render(grammar: *vlp_grammar, fills: *vlp_grammar_fill, n_fills: i32, output: *u8, capacity: i32, length: *i32) -> vlp_status
  Rendering algorithm:
    1. Validate: n_fills == grammar.slots_count. Type of each fill matches slot type.
    2. Walk template:
       For each literal range: copy bytes to output. (memcpy.)
       For each slot: render fill value to text.
         VDR_VALUE: convert to decimal string via vlp_vdr_to_string.
         TEXT: copy text bytes.
         INTEGER: convert i32 to decimal string.
         ENUM: copy enum value string.
         KB_REF: load fact from KB, render its value.
         GRAMMAR: recursively render nested grammar.
    3. Write total length.
  Every structural byte (pipes, braces, colons, commas, spaces, headers)
  comes from the template. Zero LLM forward passes for structure.
  If grammar.validated == 1: output is syntactically valid by construction.
  No malformed JSON. No mismatched brackets. No missing commas.
  The grammar guarantees it. Not the LLM.
  Side effects: none (pure render).

vlp_grammar_render_from_kb(grammar: *vlp_grammar, kb_store: *vlp_kb_store, mappings: *vlp_grammar_kb_mapping, n_mappings: i32, output: *u8, capacity: i32, length: *i32) -> vlp_status
  Same as render, but fills come from KB facts.
  Mapping: slot_index → (kb_id, slot_id).
  Data flows: KB integer address → fact read → value extraction → text render → output.
  The data never enters any token stream. It goes from integer storage
  to formatted output without the LLM seeing it.
  Side effects: none.

vlp_grammar_inherit(kb_store: *vlp_kb_store, kb_id: i32, grammar_slot: i32) -> *vlp_grammar
  Walks from kb_id up parent chain looking for grammar at grammar_slot.
  Returns first found. This is grammar inheritance through the KB tree.
  A grammar created in root.sre persists and is available to all
  children of root.sre without re-creation.
  Side effects: none.
```

### 6.5 Builtin Executor

```
vlp_builtin_executor_init() -> vlp_status
  Builds dispatch table: builtin_id → function pointer.
  448 entries. Each entry points to a device kernel or device function.
  Table is in constant memory — one read per dispatch.
  Side effects: constant memory initialized.

vlp_builtin_dispatch(builtin_id: i32, args: *vlp_builtin_args) -> vlp_builtin_result
  Flow:
    1. Lookup builtin_id in dispatch table. (Constant memory read.)
    2. Validate argument count and types against builtin's IOSE declaration.
       IOSE: Input types, Output types, Side effects, Error conditions.
       All declared. All checked. No runtime surprise.
    3. Load input data from KB (args contain kb_id + slot_id references).
    4. Execute builtin kernel on device.
    5. Store result at declared output KB location.
    6. Return result reference (kb_id + slot_id of output).
  Pure builtins (404 of 448): deterministic, bounded, no side effects.
    Same inputs → same outputs. Always. On every device. Every time.
    Termination guaranteed by bounded input size.
  Operational builtins (44 of 448): side effects, grant-gated.
    Grant already checked by command processor before dispatch reaches here.
  Side effects: output KB facts written. Operational builtins may
  access filesystem, network, or processes.

vlp_builtin_validate_iose(builtin_id: i32, args: *vlp_builtin_args) -> vlp_status
  Validates arguments against IOSE declaration without executing.
  For pre-flight checking by the command processor.
  Returns: OK or specific type mismatch error.
  Side effects: none.
```

---

## 7. Inference Loop

The complete cycle from user input to system output.

```
vlp_inference_cycle(session: vlp_session_handle, user_input: *u8, input_len: i32, output: *vlp_output_buffer) -> vlp_status

  // Phase 1: Input Processing
  // Tokenize user input (compiled tokenizer, not LLM).
  tokens = vlp_tokenize(user_input, input_len);

  // Phase 2: Context Assembly
  // Build LLM context. This is the ONLY thing the LLM sees.
  context = vlp_context_build(session);
  // Context contains:
  //   - System prompt (from seed KB, ~200 tokens)
  //   - Current turn input (user tokens)
  //   - Scratchpad (results of previous commands this turn, ~0-50 tokens)
  //   - Scope reference (active KB path, ~5 tokens)
  // Context does NOT contain:
  //   - Previous turns (they're in KB)
  //   - Data (it's at KB addresses)
  //   - Prior reasoning (it's in rules)
  //   - Formatting templates (they're in grammars)
  // Context size is approximately constant regardless of turn number.

  // Phase 3: LLM Generation
  // The LLM reads the context and generates a response.
  // Response is a mix of:
  //   - Command tokens (dispatched to system)
  //   - Prose tokens (passed through to output)
  //   - DIRECT_OUTPUT references (resolved from KB)
  loop {
      next_token = vlp_llm_generate_token(session, &sampling_config);

      if (is_command_start(next_token)) {
          // LLM is emitting a command
          command = vlp_llm_generate_command(session);

          // Phase 4: Command Execution
          result = vlp_command_execute(session, &command);

          // Result goes to scratchpad, not to output.
          // LLM can inspect result on next generation step.
          vlp_scratchpad_write(session, &result);

      } else if (is_direct_output(next_token)) {
          // LLM is referencing KB data for output
          kb_url = vlp_parse_kb_url(session);
          data = vlp_kb_store_fact_read(kb_url.kb_id, kb_url.slot_id);

          // Load associated grammar (if any)
          grammar = vlp_grammar_inherit(kb_store, kb_url.kb_id, 0);
          if (grammar != NULL) {
              vlp_grammar_render_from_kb(grammar, kb_store, &mapping, output);
              // Structured output. Every bracket from grammar.
          } else {
              vlp_render_fact_as_text(data, output);
          }

      } else if (is_end_of_turn(next_token)) {
          break;
      } else {
          // Prose token — LLM judgment and framing
          vlp_output_write_token(output, next_token);
      }
  }

  // Phase 5: Post-Processing
  // Verify output grammar constraints (if output grammar declared).
  // Update session counters.
  // Write audit entry for the complete turn.
  session.current_turn++;
  session.llm_tokens_consumed += tokens_generated;
  session.command_tokens_consumed += commands_generated * 8;

  // Phase 6: Auto-Persist
  // If session configured for auto-persist, snapshot periodically.
  if (session.current_turn % auto_snapshot_interval == 0) {
      vlp_session_snapshot(session);
  }

  return VLP_OK;
```

---

## 8. Execution Levels

### 8.1 L1 — Full LLM Judgment

```
// LLM exercises full judgment. 50-500 tokens per interaction.
// No stored rule covers the situation.

vlp_execute_l1(session: vlp_session_handle, input: *u8, len: i32) -> vlp_status
  Full inference cycle (Section 7).
  LLM reads input, reasons about it, generates commands and prose.
  This is the most expensive execution level.
  At the end, the LLM may formalize its judgment as a Prolog rule:
    CMD_PROLOG_ASSERT_RULE target_kb rule(head, body, actions)
  This costs 25-40 tokens. It transitions the pattern from L1 to L2.
  Next time this pattern is seen, the rule fires without LLM.
```

### 8.2 L2 — LLM Invokes Stored Rule

```
// LLM recognizes that a stored rule applies. 8 tokens.

vlp_execute_l2(session: vlp_session_handle, pattern: *vlp_term) -> vlp_status
  LLM generates a single command: CMD_PROLOG_QUERY
  The Prolog engine finds the matching rule and returns results.
  LLM wraps results in prose framing. Total: ~8 command tokens + ~10 prose tokens.
  Cost is ~3% of L1.
```

### 8.3 L3 — Automatic Rule Firing

```
// The Prolog rule fires automatically during polling or batch processing.
// 0 LLM tokens.

vlp_execute_l3(session: vlp_session_handle, kb_id: i32) -> vlp_status
  Called by polling runner or batch runner.
  vlp_prolog_fire_and_commit(kb_store, kb_id)
  Rules match against current facts. Matching rules fire.
  Actions applied (assert new facts, retract old).
  Zero LLM involvement. Pure integer Prolog unification.
  This is the steady state for mature deployments.
  93% of operations at investigation 100 (paper Appendix D).
```

### 8.4 Level Transition Tracking

```
struct vlp_level_stats {
    l1_count: i64,
    l1_tokens: i64,
    l2_count: i64,
    l2_tokens: i64,
    l3_count: i64,
    // l3_tokens is always 0

    // Derived
    auto_triage_percent_num: i32,    // l3_count
    auto_triage_percent_den: i32,    // l1_count + l2_count + l3_count
    // Exact fraction. At investigation 100: ~93/100.
};

vlp_level_stats_update(session: vlp_session_handle, level: i8, tokens: i32) -> vlp_status
  Called after each execution. Updates counters.
  Side effects: session stats updated.

vlp_level_stats_query(session: vlp_session_handle) -> vlp_level_stats
  Returns current L1/L2/L3 distribution.
  auto_triage_percent is the key operational metric.
```

---

## 9. Confidence Propagation

### 9.1 Confidence Assignment

```
vlp_confidence_assign(fact: *vlp_fact, source_type: vlp_source_type) -> vlp_status
  Sets fact.provenance.confidence from CONFIDENCE_TABLE[source_type].
  Called automatically during KB_ASSERT from command processor.
  The confidence is an exact VDR fraction, not generated language.
  Side effects: fact provenance updated.
```

### 9.2 Confidence Combination

```
vlp_confidence_combine_agreeing(confidences: *vlp_q16, n: i32, result: *vlp_q16) -> vlp_status
  Formula: 1 - ∏(1 - C_i)
  Implementation:
    1. For each C_i: compute (D - C_i.v) = complement. (Integer subtract.)
    2. Multiply all complements: product = ∏ complement_i. (Widening multiply + shift.)
    3. Result = D - product. (Integer subtract.)
  Exact. Two sources at 95/100: 1 - (5/100)² = 1 - 25/10000 = 9975/10000.
  In Q16: 1 - (3276 * 3276 / 65536) = 65536 - 163 = 65373. (Approximately 99.75%.)
  Exact within Q16 frame. Remainder tells you the sub-Q16 precision.
  Side effects: none.

vlp_confidence_combine_conflicting(confidences: *vlp_q16, n: i32, penalty: vlp_q16, result: *vlp_q16) -> vlp_status
  Same as agreeing, but each conflicting pair applies penalty multiplier.
  Conflict detection: caller determines which sources conflict.
  Side effects: none.

vlp_confidence_chain(per_link: vlp_q16, n_links: i32, result: *vlp_q16) -> vlp_status
  C^N via repeated exact multiplication.
  3 links at 85/100: (55705/65536)^3 = exact Q16 value.
  Side effects: none.
```

### 9.3 Confidence Propagation Through Derivation

```
vlp_confidence_propagate(kb_store: *vlp_kb_store, fact_kb_id: i32, fact_slot_id: i32) -> vlp_q16
  Walks the provenance chain of a fact.
  If fact.provenance.derivation_rule_id != -1:
    The fact was derived by a Prolog rule.
    Load the rule's body conditions.
    For each body condition: recursively propagate confidence of source facts.
    Chain the confidences: result = product of source confidences.
  If fact.provenance.source_type is direct (not derived):
    Return CONFIDENCE_TABLE[source_type].
  Handles cycles: if a fact references itself transitively, confidence = 0/1.
  (Circular derivation is undefined knowledge.)
  Side effects: none.
```

---

## 10. Snapshot Format

### 10.1 Binary Layout

```
struct vlp_snapshot_header {
    magic: [4]u8,            // "VLPS" — VLP Snapshot
    version: i32,            // format version
    timestamp: i32,
    session_id: i32,
    user_id: i32,

    // Region sizes (bytes)
    kb_region_size: i64,
    fact_region_size: i64,
    rule_region_size: i64,
    term_region_size: i64,
    text_region_size: i64,
    grammar_region_size: i64,
    live_state_region_size: i64,
    grant_region_size: i64,
    path_index_region_size: i64,

    // Counts
    kb_count: i32,
    fact_count: i64,
    rule_count: i32,
    term_count: i64,
    grammar_count: i32,
    grant_count: i32,

    // Session metadata
    session_metadata: vlp_session,  // full session struct

    // Integrity
    checksum: i32,           // CRC32 over all data regions
    total_size: i64,         // total snapshot size in bytes
};

// Snapshot body: contiguous regions in header-declared order.
// [header][kb_region][fact_region][rule_region][term_region]
// [text_region][grammar_region][live_state_region][grant_region]
// [path_index_region]
//
// All data is raw integers. No float serialization.
// No endian conversion needed if all devices are same endianness.
// (If cross-endian support needed: byte-swap during save/load.
//  Integer byte-swap is exact. Float byte-swap is also exact but
//  the float values themselves are approximate. Integers are not.)
//
// Snapshot is the deployable binary. Clone it for instances.
// Ship it to another machine. Bit-identical restore. Always.
```

### 10.2 Snapshot Operations

```
vlp_snapshot_save(snapshot: vlp_snapshot_handle, path: *u8) -> vlp_status
  Writes snapshot to host filesystem as a single file.
  Format: header + data regions, contiguous.
  File size: typically 10 KB - 500 KB for operational sessions.
  (Model weights are NOT in the snapshot — they're shared and loaded separately.)
  The snapshot captures session state, not model state.
  Side effects: file written.

vlp_snapshot_load(path: *u8) -> vlp_snapshot_handle
  Reads snapshot from file. Validates magic, version, checksum.
  If checksum mismatch: ERR_SNAPSHOT_CORRUPT. Hard fail. Never silent.
  Returns handle to host-memory snapshot.
  Side effects: host memory allocated.

vlp_snapshot_diff(a: vlp_snapshot_handle, b: vlp_snapshot_handle) -> vlp_diff_result
  Binary diff between two snapshots.
  Reports: which KB facts differ, which rules differ, which live state differs.
  Every difference is a real change — not float noise.
  Because integers. If two snapshots are supposed to be identical and
  the diff is non-empty, something changed. Find it.
  Side effects: none.

vlp_snapshot_merge(base: vlp_snapshot_handle, branch_a: vlp_snapshot_handle, branch_b: vlp_snapshot_handle, policy: vlp_merge_policy) -> vlp_snapshot_handle
  Three-way merge of snapshots. For merging work from two parallel clones.
  Conflict detection: same fact/rule modified in both branches.
  Policy determines conflict resolution (same as session merge).
  Side effects: new snapshot created in host memory.
```

---

## 11. Seed Layer

### 11.1 Seed Contents

```
vlp_seed_init(kb_store: *vlp_kb_store) -> vlp_status
  Populates the initial KB tree that every session inherits.
  ~23,400 entries across ~1.5 MB.
  Structure:
    root
    ├── system
    │   ├── oso              # 15 engineering principles as ~176 Prolog terms
    │   ├── confidence       # Confidence table as KB facts
    │   ├── builtins         # IOSE declarations for all 448 builtins
    │   ├── command_vocab    # Command token vocabulary (~300 names)
    │   └── hygiene          # Self-maintenance rules
    │       ├── stale_rule_detector
    │       ├── failing_rule_detector
    │       └── orphan_rule_detector
    ├── templates
    │   ├── sentences        # Sentence structure templates
    │   └── formats          # Common format grammars (JSON, CSV, table, etc.)
    └── (user-created KBs go here)

  Seed KB is frozen after initialization. Immutable.
  All sessions inherit from it. No session can modify seed data.
  Side effects: KB store populated with seed data.
```

### 11.2 Hygiene Rules (Self-Maintenance)

```
// Stale rule detector
// Fires during internal runner cycle.
rule(stale_rule_detector,
    [rule_exists(RuleId, KB),
     rule_last_fired(RuleId, LastFired),
     current_time(Now),
     vdr_sub(Now, LastFired, Age),
     vdr_compare(Age, 7776000, greater)],  // 90 days in seconds
    [assert(candidate_for_pruning(RuleId, stale, Age))])

// Failing rule detector
rule(failing_rule_detector,
    [rule_exists(RuleId, KB),
     rule_fire_count(RuleId, Fires),
     vdr_compare(Fires, 5, greater),       // at least 5 attempts
     rule_success_rate(RuleId, SuccNum, SuccDen),
     vdr_compare(SuccNum * 100, SuccDen * 20, less)],  // < 20% success
    [assert(candidate_for_pruning(RuleId, failing, SuccNum/SuccDen))])

// Orphan rule detector
rule(orphan_rule_detector,
    [rule_exists(RuleId, KB),
     rule_references_grant(RuleId, GrantId),
     grant_state(GrantId, revoked)],
    [assert(candidate_for_pruning(RuleId, orphaned, GrantId))])

// These rules don't auto-delete. They assert candidates.
// A hygiene runner reviews candidates and decides whether to retract.
// The self-maintenance system is itself auditable and conservative.
```

---

## 12. Multi-Device

### 12.1 Model Parallelism

```
vlp_multi_device_init(n_devices: i32, model_config: *vlp_model_config) -> vlp_status
  Distributes model layers across devices.
  Strategy: pipeline parallelism. Device 0 gets layers 0..n/k,
  device 1 gets layers n/k..2n/k, etc.
  Each device loads its shard from checkpoint.
  KB store replicated on each device (small — ~2 GB).
  Sessions bound to device 0 (or any single device — KB is small enough).
  Forward pass: input enters device 0, hidden state transferred to device 1
  at layer boundary, continues through pipeline.
  Side effects: multi-device memory allocated, model shards loaded.

vlp_multi_device_forward(session: vlp_session_handle, input_ids: *i32, n_tokens: i32, logits: *vlp_q16) -> vlp_status
  Pipeline forward pass across devices.
  Hidden state transferred via NVLink between stages.
  Transfer data: vlp_q16 arrays. Integer data. Bit-identical after transfer.
  No precision loss at device boundaries.
  Side effects: KV-cache updated on each device for its layers.
```

### 12.2 KB Replication

```
vlp_kb_replicate(source_device: i32, target_device: i32, kb_id: i32) -> vlp_status
  Copies KB and its facts from one device to another.
  Bit-identical. Integers transfer exactly.
  Used for: deploying a trained session to multiple inference devices.
  Side effects: target device KB store updated.

vlp_kb_sync(devices: *i32, n_devices: i32, kb_id: i32) -> vlp_status
  Synchronizes KB across multiple devices.
  Broadcast model: one device is authoritative, others receive updates.
  Update detection: compare last_modified timestamps. (Integer comparison.)
  Transfer only changed facts. (Diff by slot_id scan.)
  Side effects: all devices' KB stores synchronized.
```

---

## 13. Operational Deployment

### 13.1 Runner Configurations for Production

```
// Standard SRE deployment: 4 runners

vlp_deployment_sre_init(config: *vlp_sre_config) -> vlp_status
  Creates:
    1. Prometheus processor runner
       - Connects to Prometheus API
       - Ingests metrics every scrape interval
       - Compacts to KB facts: service → error_rate, latency, throughput
       - Recycles every 200 turns

    2. Deploy pipeline processor runner
       - Connects to deploy system webhook/API
       - Ingests deployment events
       - Compacts to KB facts: service → version, timestamp, config_diff
       - Recycles every 200 turns

    3. Triage polling runner (60s interval)
       - Fires all triage rules against current facts
       - Creates incident KBs for fired rules
       - Renders findings through stored grammars
       - Zero LLM tokens for routine triage

    4. Hygiene internal runner (daily)
       - Fires hygiene rules (stale, failing, orphaned detection)
       - Reports candidates for review
       - Computes coverage metrics

  Side effects: 4 runners created and started, 4 sessions created.

vlp_deployment_sre_status() -> vlp_sre_status
  Reports:
    - Runner states (all should be RUNNING)
    - KB counts (facts, rules, grammars accumulated)
    - Level distribution (L1/L2/L3 percentages — exact fractions)
    - Auto-triage percentage (exact fraction)
    - Mean tokens per investigation (exact fraction, should decrease over time)
    - Rule hygiene: stale/failing/orphaned counts
```

### 13.2 Capacity Planning

```
vlp_capacity_estimate(config: *vlp_capacity_config) -> vlp_capacity_result
  Config: model_size_params, max_concurrent_sessions, max_seq_len,
  expected_facts_per_session, expected_rules_total.
  Returns:
    model_memory_bytes: i64,          // model_size * 8 (Q16)
    kb_memory_bytes: i64,             // from session and fact counts
    live_state_memory_bytes: i64,     // from session count * avg live size
    total_device_memory_bytes: i64,   // sum of all regions
    n_devices_required: i32,          // ceil(total / device_memory)
    tokens_per_second_estimate: i64,  // from TensorProlog device throughput
  All integer calculations. No "approximately."
```

---

## 14. Testing Infrastructure

### 14.1 Determinism Verification

```
vlp_test_determinism(test_case: *vlp_test_case, n_runs: i32) -> vlp_test_result
  Runs test_case n_runs times. Compares all outputs bit-by-bit.
  If any difference: test FAILS. Not "within tolerance." FAILS.
  The difference is a bug, not noise. Find it.
  Test cases include: forward pass, backward pass, Prolog query,
  grammar render, builtin execution, snapshot save/restore roundtrip.
  Side effects: none beyond test execution.

vlp_test_snapshot_roundtrip(session: vlp_session_handle) -> vlp_test_result
  1. Snapshot session.
  2. Modify session state (add facts, fire rules).
  3. Restore from snapshot.
  4. Compare all state to pre-modification state. Bit-identical.
  5. If not identical: test FAILS. Snapshot/restore is broken.
  Side effects: session state modified and restored.

vlp_test_clone_independence(parent: vlp_session_handle) -> vlp_test_result
  1. Clone parent.
  2. Modify clone's KB.
  3. Verify parent's KB unchanged.
  4. Modify parent's KB.
  5. Verify clone's KB unchanged (COW isolation).
  6. All comparisons exact. Integer equality.
  Side effects: clone created and destroyed.

vlp_test_access_isolation(session_a: vlp_session_handle, session_b: vlp_session_handle) -> vlp_test_result
  1. Session A asserts fact to OWNER_ONLY KB.
  2. Session B queries same KB.
  3. Verify: B receives ERR_KB_ACCESS_DENIED.
  4. Verify: B's query result contains zero facts from A's KB.
  5. Not "filtered." Absent. The query didn't return redacted data.
     It returned nothing because the KB was invisible.
  Side effects: facts asserted by A.

vlp_test_confidence_propagation(kb_store: *vlp_kb_store) -> vlp_test_result
  1. Assert fact from PROMETHEUS source (confidence 95/100).
  2. Assert rule deriving new fact from that fact.
  3. Fire rule.
  4. Check derived fact confidence = 95/100 (chain of 1 link).
  5. Add second Prometheus source agreeing. Combine.
  6. Check combined = 1 - (5/100)² = 9975/10000.
  7. Exact integer comparison at each step.
  Side effects: facts and rules created.
```

### 14.2 Performance Benchmarks

```
vlp_bench_forward_pass(model_config: *vlp_model_config, seq_len: i32, n_iters: i32) -> vlp_bench_result
  Times n_iters forward passes. Reports:
    total_ns, per_iter_ns, tokens_per_second.
  Reference: Q16 toy achieved 688 ns / 1.42M tok/s on 2019 laptop.
  Production 7B model on H100 INT8 path: target TBD by implementation.
  All timing is integer nanoseconds (host clock).

vlp_bench_prolog_query(n_facts: i32, n_rules: i32, n_iters: i32) -> vlp_bench_result
  Times Prolog query against fact store of declared size.
  Reports: per_query_ns, queries_per_second.
  Reference: FPGA achieved ~1.1μs for 200 facts.

vlp_bench_grammar_render(template_len: i32, n_slots: i32, n_iters: i32) -> vlp_bench_result
  Times grammar rendering. Reports: per_render_ns, renders_per_second.
  Should be negligible relative to LLM forward pass.

vlp_bench_builtin(builtin_id: i32, input_size: i32, n_iters: i32) -> vlp_bench_result
  Times specific builtin. Reports: per_call_ns.
  Reference from paper: primitives execute ~50,000× faster than generating
  a single LLM token. Bottleneck is always the LLM forward pass.

vlp_bench_snapshot(session: vlp_session_handle, n_iters: i32) -> vlp_bench_result
  Times snapshot save + restore roundtrip.
  Reports: save_ns, restore_ns, snapshot_size_bytes.
  Typical: 10-500 KB, microseconds.

vlp_bench_command_parse(n_iters: i32) -> vlp_bench_result
  Times command token parsing. Should be nanoseconds per command.
  Reports: per_command_ns.

vlp_bench_access_check(tree_depth: i32, n_iters: i32) -> vlp_bench_result
  Times access control check at various tree depths.
  Reports: per_check_ns. Expected: ~10ns per ancestor level.
  At depth 10: ~100ns. Negligible.
```

---

## 15. Error Handling

### 15.1 Error Categories

```
enum vlp_error_category: i8 {
    ERR_CAT_NONE        = 0,   // no error
    ERR_CAT_ARITHMETIC  = 1,   // VDR operation error (division by zero, overflow)
    ERR_CAT_KB          = 2,   // KB not found, full, frozen, access denied
    ERR_CAT_PROLOG      = 3,   // depth exceeded, no matching rule, unification failure
    ERR_CAT_GRAMMAR     = 4,   // invalid template, type mismatch, capacity exceeded
    ERR_CAT_SESSION     = 5,   // session limit, snapshot failure, clone failure
    ERR_CAT_GRANT       = 6,   // grant denied, expired, exhausted, revoked
    ERR_CAT_RUNNER      = 7,   // runner error threshold, connection loss
    ERR_CAT_DEVICE      = 8,   // TensorProlog device error, out of memory
    ERR_CAT_SYSTEM      = 9,   // initialization failure, corrupt state
};

// Every error is a specific integer code with a specific meaning.
// No "something went wrong" errors. Every failure mode is declared.
// Every declared failure mode has a recovery path.
// Deterministic errors: same input producing same error, always.
// There is no "intermittent error from float instability" category
// because float instability doesn't exist.
```

### 15.2 Error Recovery

```
vlp_error_recover(session: vlp_session_handle, error: vlp_status) -> vlp_recovery_action
  Decision tree (all integer comparisons):
    ERR_CAT_KB + KB_FULL: return RECOVERY_COMPACT (run LRU eviction on live state)
    ERR_CAT_KB + ACCESS_DENIED: return RECOVERY_LOG_AND_CONTINUE
    ERR_CAT_PROLOG + DEPTH_EXCEEDED: return RECOVERY_SIMPLIFY_QUERY
    ERR_CAT_SESSION + SNAPSHOT_FAILED: return RECOVERY_RETRY_SNAPSHOT
    ERR_CAT_GRANT + DENIED: return RECOVERY_LOG_AND_DENY
    ERR_CAT_RUNNER + CONNECTION_LOSS: return RECOVERY_RECONNECT_WITH_BACKOFF
    ERR_CAT_RUNNER + ERROR_THRESHOLD: return RECOVERY_RECYCLE_RUNNER
    ERR_CAT_DEVICE + OUT_OF_MEMORY: return RECOVERY_KILL_OLDEST_CLONE
    ERR_CAT_SYSTEM + CORRUPT_STATE: return RECOVERY_RESTORE_FROM_SNAPSHOT

  Every recovery action is deterministic. Given the same error in the
  same state, the same recovery action is taken. Always.
  No "try again and hope float non-determinism produces a different result."
```

---

## 16. Conformance

### 16.1 Test Suite

```
// The system passes the same 884-test suite as the VDR arithmetic foundation.
// Plus:
//   - KB tests: create, assert, query, retract, scope, visibility, freeze, COW
//   - Prolog tests: unify, query, fire, backtrack, depth limit, hygiene
//   - Grammar tests: compile, render, inherit, compose, validate
//   - Session tests: create, snapshot, restore, clone, merge, kill
//   - Runner tests: start, stop, recycle, error recovery
//   - Safety tests: access check, grant check, audit
//   - Confidence tests: assign, combine, chain, propagate
//   - Integration tests: full inference cycle, SRE scenario, accumulation curve
//   - Determinism tests: every operation produces identical results across runs

// Total estimated: ~2,500 tests.
// Zero tolerance for VDR arithmetic errors (per established 884-test baseline).
// Zero tolerance for determinism failures.
// Zero tolerance for access control bypasses.
```

### 16.2 Invariants

```
// These invariants hold at all times, in all states, on all devices.
// Violation of any invariant is a system bug, not a configuration issue.

INVARIANT_1: Softmax outputs sum to D exactly.
  For every softmax call, for every row, the sum of outputs equals
  the denominator of the Q-basis. Not approximately. Exactly.
  Checked by: TensorPrologAttentionVerifySoftmaxSum (zero violations expected).

INVARIANT_2: KB facts at integer addresses are exact.
  Fact N at address N returns exactly what was asserted.
  At turn 1 or turn 1,000,000. Integer storage does not degrade.

INVARIANT_3: Bounded primitives cannot exceed declared bounds.
  LRU with capacity 1000 contains at most 1000 entries.
  Counter with max 100 never exceeds 100.
  Queue with capacity 500 rejects push at 500.
  No exceptions. No overflow. No wrap-around.

INVARIANT_4: Snapshot restore is bit-identical to snapshot state.
  Save state S. Modify. Restore. State == S. Every byte.

INVARIANT_5: Clone COW is invisible to parent.
  Clone's writes never modify parent's state.
  Parent's writes never modify clone's state.

INVARIANT_6: Access-denied data is absent, not filtered.
  A query from a session without access to KB X returns zero results
  from X. Not redacted results. Not empty results with a note. Zero.
  The query path never touches X's data.

INVARIANT_7: Grant denial happens before execution.
  An operational primitive without a valid grant is rejected before
  any side effect occurs. No partial execution.

INVARIANT_8: Integer arithmetic is deterministic across devices.
  Same operation, same inputs, same result. Every device. Every time.
  No rounding modes. No thread-ordering dependence. No platform behavior.

INVARIANT_9: Prolog unification uses exact comparison.
  Two VDR values are equal iff a.v * b.D == b.v * a.D (cross-multiply).
  No tolerance. No epsilon. Equal or not equal.

INVARIANT_10: Audit log is append-only and complete.
  Every access check, grant check, fact assertion, fact retraction,
  rule firing, session creation, session destruction, and operational
  execution produces an audit entry. No gaps. No overwrites (ring buffer
  overwrites oldest, but all entries are written).
```

---

## 17. Implementation Stages

Mapping to the paper's 5-stage build plan.

```
Stage 1: Foundation
  - vlp_q16 type + arithmetic (from VDR-32 Zig implementation)
  - KB store engine (create, assert, query, retract, scope)
  - Fact store with integer addressing
  - Path index hash map
  - Basic session (create, destroy, no snapshot yet)
  - 8 pure builtin categories (text, collections, sets, mappings,
    closed arithmetic, comparison, rounding, integer/bit ops)
  - Command parser (parse command tokens → vlp_command)
  - Access check (visibility walk)
  Deliverable: system can create KBs, assert facts, query facts,
  execute basic builtins, enforce visibility. All integer. All exact.
  Estimated: ~5,000 lines Zig.

Stage 2: Intelligence
  - Prolog engine (unify, query, fire, backtrack)
  - Rule store + term store
  - Grammar engine (compile, render, inherit)
  - Session snapshot/restore/clone/merge/kill
  - Live state primitives (LRU, counter, lock, queue, stack, ring, bitset)
  - Grant system (create, check, revoke)
  - Confidence propagation
  - Remaining pure builtins (linear algebra, statistics, active arithmetic,
    structure ops, number theory, collections advanced)
  Deliverable: full Prolog deduction, grammar rendering, session lifecycle,
  bounded data primitives, grant-gated access. The system can accumulate
  rules and grammars. L1→L2→L3 transition is possible.
  Estimated: ~8,000 lines Zig.

Stage 3: Precision
  - Q32 and Q335 types
  - Q-basis reprojection
  - FRU operations (sqrt, exp, log, sin, cos) — software implementation
  - Remainder resolution (compact, normalize)
  - Polynomial, finite field, discrete calculus builtins
  - Denominator management builtins
  Deliverable: high-precision computation, transcendentals, full
  Q-basis flexibility. Models can choose precision per layer.
  Estimated: ~3,500 lines Zig.

Stage 4: Operations
  - Runner scheduler (poller, processor, internal, batch)
  - 44 operational builtins (filesystem, compile, execute, lint, network, process)
  - Docker sandbox integration
  - Network fetch with grant enforcement
  - Audit system (ring buffer, query, filter)
  - Full inference loop (Section 7)
  Deliverable: autonomous operation. Runners execute continuously.
  The system is a daemon, not a function.
  Estimated: ~3,000 lines Zig.

Stage 5: Scale
  - TensorProlog GPU kernel implementations of all engines
  - Multi-device model parallelism
  - KB replication and sync
  - Distributed training (allreduce)
  - FPGA/ASIC kernel variants (when hardware available)
  - Production serving infrastructure (continuous batching, session pools)
  Deliverable: production-scale deployment on GPU hardware.
  Estimated: ~5,000 lines Zig + TensorProlog kernels.

Total estimated: ~24,500 lines.
Paper's estimate: ~20,500 lines across 65 modules.
Difference: TensorProlog integration layer adds ~4,000 lines.
```

---

## 18. Configuration Reference

```
struct vlp_system_config {
    // Device
    device_id: i32,                  // -1 for auto-select
    n_devices: i32,                  // for multi-device

    // Model
    model_checkpoint_path: [256]u8,
    model_n_layers: i32,
    model_d_model: i32,
    model_n_heads: i32,
    model_vocab_size: i32,
    model_mlp_dim: i32,
    model_qbasis: vlp_qbasis,       // Q16 for inference, Q32 for training

    // Memory
    max_total_kbs: i32,              // default 100,000
    max_total_facts: i64,            // default 10,000,000
    max_total_rules: i32,            // default 100,000
    max_total_terms: i64,            // default 1,000,000
    text_store_bytes: i64,           // default 100 MB
    scratch_per_stream_bytes: i64,   // default 10 MB

    // Sessions
    max_concurrent_sessions: i32,    // default 10,000
    default_max_kb_per_session: i32, // default 100
    default_max_turns: i32,          // default 0 (unlimited)
    auto_snapshot_interval: i32,     // default 100 turns (0 = disabled)

    // Runners
    max_runners: i32,                // default 64
    runner_thread_pool_size: i32,    // default 0 (auto = hw_threads / 2)

    // Safety
    audit_ring_capacity: i32,        // default 1,000,000 entries
    default_visibility: i8,          // default INTERNAL

    // Seed
    seed_snapshot_path: [256]u8,     // path to seed layer snapshot

    // Sampling defaults
    default_temperature_v: i32,      // default 65536 (= 1.0 at Q16)
    default_top_k: i32,             // default 50
    default_top_p_v: i32,           // default 58982 (= 0.9 at Q16)
};
```

