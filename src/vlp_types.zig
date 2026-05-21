// ============================================================
// vlp_types.zig
// All shared types used across host and device.
// extern struct for anything that crosses the host-device boundary.
// ============================================================

const std = @import("std");

const shared = @import("vlp_gpu_shared");

// ── Re-export everything from gpu_shared ──

pub const FACTS_PER_KB = shared.FACTS_PER_KB;
pub const FACT_INTS = shared.FACT_INTS;
pub const FACT_STRIDE = shared.FACT_STRIDE;
pub const KB_STRUCT_INTS = shared.KB_STRUCT_INTS;
pub const RULE_INTS = shared.RULE_INTS;
pub const TERM_INTS = shared.TERM_INTS;
pub const BINDING_INTS = shared.BINDING_INTS;
pub const D16 = shared.D16;
pub const D32 = shared.D32;
pub const MAX_WORKGROUP = shared.MAX_WORKGROUP;

pub const OpCode = shared.OpCode;
pub const FactTag = shared.FactTag;
pub const TermType = shared.TermType;
pub const SourceType = shared.SourceType;
pub const confidence_table_raw = shared.confidence_table;

// Field offset constants
pub const FACT_TAG = shared.FACT_TAG;
pub const FACT_VALUE_V = shared.FACT_VALUE_V;
pub const FACT_VALUE_R0 = shared.FACT_VALUE_R0;
pub const FACT_PROV_SOURCE = shared.FACT_PROV_SOURCE;
pub const FACT_PROV_KB = shared.FACT_PROV_KB;
pub const FACT_PROV_SLOT = shared.FACT_PROV_SLOT;
pub const FACT_PROV_CONF_V = shared.FACT_PROV_CONF_V;
pub const FACT_PROV_CONF_R0 = shared.FACT_PROV_CONF_R0;
pub const FACT_PROV_TIME = shared.FACT_PROV_TIME;
pub const FACT_PROV_RULE = shared.FACT_PROV_RULE;

pub const P_OP_CODE = shared.P_OP_CODE;
pub const P_FIELD_0 = shared.P_FIELD_0;
pub const P_FIELD_1 = shared.P_FIELD_1;
pub const P_FIELD_2 = shared.P_FIELD_2;
pub const P_FIELD_3 = shared.P_FIELD_3;
pub const P_FIELD_4 = shared.P_FIELD_4;
pub const P_FIELD_5 = shared.P_FIELD_5;
pub const P_FIELD_6 = shared.P_FIELD_6;
pub const P_FIELD_7 = shared.P_FIELD_7;
pub const P_FIELD_8 = shared.P_FIELD_8;
pub const P_FIELD_9 = shared.P_FIELD_9;

// ── Q16 — host-side wrapper with methods ──

// ---- Q-basis ----

pub const QBasis = enum(i32) {
    q16 = 16,
    q32 = 32,
    q335 = 335,
};

// ---- VDR Value Types ----
pub const Q16 = extern struct {
    v: i32,
    r0: i16 = 0,
    r1: i16 = 0,

    // pub const D: i32 = 65536;

    pub fn zero() Q16 {
        return .{ .v = 0 };
    }

    pub fn one() Q16 {
        return .{ .v = D16 };
    }

    pub fn fromParts(v: i32, r0: i16, r1: i16) Q16 {
        return .{ .v = v, .r0 = r0, .r1 = r1 };
    }

    pub fn add(a: Q16, b: Q16) Q16 {
        // r1 + r1 → carry into r0
        const r1_sum: i32 = @as(i32, a.r1) + @as(i32, b.r1);
        const r1_carry: i32 = @divTrunc(r1_sum, 32768); // i16 max range
        const new_r1: i16 = @intCast(@mod(r1_sum, 32768));
        // r0 + r0 + carry from r1
        const r0_sum: i32 = @as(i32, a.r0) + @as(i32, b.r0) + r1_carry;
        const r0_carry: i64 = if (r0_sum >= D16) 1 else 0;
        const new_r0: i16 = @intCast(@mod(r0_sum, D16));
        // v + v + carry from r0
        const new_v: i32 = @intCast(@as(i64, a.v) + @as(i64, b.v) + r0_carry);
        return .{ .v = new_v, .r0 = new_r0, .r1 = new_r1 };
    }

    pub fn sub(a: Q16, b: Q16) Q16 {
        var r1_diff: i32 = @as(i32, a.r1) - @as(i32, b.r1);
        var r1_borrow: i32 = 0;
        if (r1_diff < 0) {
            r1_diff += 32768;
            r1_borrow = 1;
        }
        var r0_diff: i32 = @as(i32, a.r0) - @as(i32, b.r0) - r1_borrow;
        var r0_borrow: i64 = 0;
        if (r0_diff < 0) {
            r0_diff += D16;
            r0_borrow = 1;
        }
        const new_v: i32 = @intCast(@as(i64, a.v) - @as(i64, b.v) - r0_borrow);
        return .{ .v = new_v, .r0 = @intCast(r0_diff), .r1 = @intCast(r1_diff) };
    }

    pub fn mul(a: Q16, b: Q16) Q16 {
        const product: i64 = @as(i64, a.v) * @as(i64, b.v);
        const new_v: i32 = @intCast(@divTrunc(product, D16));
        const r0_full: i64 = @mod(product, D16);
        // Second remainder: capture sub-r0 precision from r0 interactions
        const r1_product: i64 = @as(i64, a.r0) * @as(i64, b.v) + @as(i64, b.r0) * @as(i64, a.v);
        const new_r1: i16 = @intCast(@mod(@divTrunc(r1_product, D16), 32768));
        return .{ .v = new_v, .r0 = @intCast(r0_full), .r1 = new_r1 };
    }

    pub fn div(a: Q16, b: Q16) Q16 {
        if (b.v == 0) return zero();
        const widened: i64 = @as(i64, a.v) * D16;
        const new_v: i32 = @intCast(@divTrunc(widened, @as(i64, b.v)));
        const r0_full: i64 = @mod(widened, @as(i64, b.v));
        // r1: sub-quotient precision from remainder
        const r1_widened: i64 = r0_full * D16;
        const new_r1: i16 = @intCast(@mod(@divTrunc(r1_widened, @as(i64, b.v)), 32768));
        return .{ .v = new_v, .r0 = @intCast(r0_full), .r1 = new_r1 };
    }

    pub fn crossMultiplyCompare(a: Q16, b: Q16) i32 {
        if (a.v < b.v) return -1;
        if (a.v > b.v) return 1;
        if (a.r0 < b.r0) return -1;
        if (a.r0 > b.r0) return 1;
        if (a.r1 < b.r1) return -1;
        if (a.r1 > b.r1) return 1;
        return 0;
    }

    pub fn eql(a: Q16, b: Q16) bool {
        return a.v == b.v and a.r0 == b.r0 and a.r1 == b.r1;
    }

    /// Convert to flat i32 array for GPU buffer writes
    /// Packs r0 and r1 into one i32: r0 in lower 16, r1 in upper 16
    pub fn toInts(self: Q16) [2]i32 {
        return .{ self.v, @as(i32, self.r0) | (@as(i32, self.r1) << 16) };
    }

    /// Read from flat i32 array from GPU buffer reads
    pub fn fromInts(ints: [2]i32) Q16 {
        return .{
            .v = ints[0],
            .r0 = @intCast(ints[1] & 0xFFFF),
            .r1 = @intCast((ints[1] >> 16) & 0xFFFF),
        };
    }
};

pub const Q32 = extern struct {
    v: i64,
    r0: i32,
    r1: i32,

    // pub const D: i64 = 4294967296; // 2^32

    pub fn zero() Q32 {
        return .{ .v = 0, .r0 = 0, .r1 = 0 };
    }

    pub fn one() Q32 {
        return .{ .v = @intCast(D32), .r0 = 0, .r1 = 0 };
    }

    pub fn fromQ16(q: Q16) Q32 {
        // Scale v from D=65536 to D=4294967296
        const scaled: i64 = @as(i64, q.v) * (D32 / Q16.D);
        return .{ .v = scaled, .r0 = @as(i32, q.r0), .r1 = 0 };
    }

    pub fn toQ16(self: Q32) Q16 {
        const scaled: i64 = @divTrunc(self.v * Q16.D, D32);
        return .{ .v = @intCast(scaled), .r0 = @intCast(@mod(self.v, Q16.D)) };
    }
};

pub const Q335 = extern struct {
    v: [6]i64,
    r0: [6]i64,
    r1: [6]i64,
    r2: [6]i64,
    r3: [6]i64,

    pub fn zero() Q335 {
        const z = [_]i64{0} ** 6;
        return .{ .v = z, .r0 = z, .r1 = z, .r2 = z, .r3 = z };
    }
};

// ---- Fact Types ----

// pub const FactTag = enum(i32) {
//     value = 0,
//     text = 1,
//     reference = 2,
//     timestamp = 3,
//     @"enum" = 4,
//     boolean = 5,
//     vector = 6,
//     matrix = 7,
//     provenance_tag = 8,
//     rule_ref = 9,
//     grammar_ref = 10,
//     counter = 11,
//     empty = 255,
// };

// pub const SourceType = enum(i8) {
//     vdr_computation = 0,
//     prolog_derivation = 1,
//     database = 2,
//     prometheus = 3,
//     script = 4,
//     rest_api = 5,
//     published = 6,
//     user_stated = 7,
//     web_search = 8,
//     llm_generated = 9,
//     unknown = 10,
// };

pub const Provenance = extern struct {
    source_type: i32,
    source_kb_id: i32,
    source_slot_id: i32,
    confidence: Q16,
    timestamp: i32,
    derivation_rule_id: i32,

    pub fn direct(source: SourceType, kb_id: i32, slot_id: i32, time: i32) Provenance {
        return .{
            .source_type = @intFromEnum(source),
            .source_kb_id = kb_id,
            .source_slot_id = slot_id,
            .confidence = confidence_table[@intCast(@intFromEnum(source))],
            .timestamp = time,
            .derivation_rule_id = -1,
        };
    }

    pub fn derived(rule_id: i32, kb_id: i32, slot_id: i32, conf: Q16, time: i32) Provenance {
        return .{
            .source_type = @intFromEnum(SourceType.prolog_derivation),
            .source_kb_id = kb_id,
            .source_slot_id = slot_id,
            .confidence = conf,
            .timestamp = time,
            .derivation_rule_id = rule_id,
        };
    }
};

pub const Fact = extern struct {
    tag: FactTag,
    value: Q16,
    provenance: Provenance,

    pub fn empty() Fact {
        return .{
            .tag = .empty,
            .value = Q16.zero(),
            .provenance = .{
                .source_type = @intFromEnum(SourceType.unknown),
                .source_kb_id = -1,
                .source_slot_id = -1,
                .confidence = Q16.zero(),
                .timestamp = 0,
                .derivation_rule_id = -1,
            },
        };
    }

    pub fn isEmpty(self: Fact) bool {
        return self.tag == .empty;
    }
};

// ---- KB Type ----

pub const KB_STRUCT_SIZE: i32 = 256;

pub const Kb = extern struct {
    // Identity (12 bytes)
    name_offset: i32,
    name_length: i16,
    path_offset: i32,
    path_length: i16,
    id: i32,

    // Persistent stores
    facts_offset: i32,
    facts_count: i32,
    facts_capacity: i32,
    rules_offset: i32,
    rules_count: i32,
    rules_capacity: i32,
    constraints_offset: i32,
    constraints_count: i32,
    connections_offset: i32,
    connections_count: i32,
    grammars_offset: i32,
    grammars_count: i32,
    iose_offset: i32,

    // Live state
    working_data_offset: i32,
    lru_table_offset: i32,
    lru_count: i16,
    counter_table_offset: i32,
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
    parent_id: i32,
    children_offset: i32,
    children_count: i16,
    children_capacity: i16,
    mounts_offset: i32,
    mounts_count: i16,

    // Metadata
    visibility: i8,
    frozen: i8,
    owner_offset: i32,
    owner_length: i16,
    created_at: i32,
    last_modified: i32,

    // Pad to 256 bytes
    _reserved: [58]u8 = [_]u8{0} ** 58,

    pub fn isPublic(self: Kb) bool {
        return self.visibility == 0;
    }

    pub fn isInternal(self: Kb) bool {
        return self.visibility <= 1;
    }

    pub fn isFrozen(self: Kb) bool {
        return self.frozen != 0;
    }

    pub fn isRoot(self: Kb) bool {
        return self.parent_id == -1;
    }

    pub fn factSlotOffset(self: Kb, slot_id: i32) i32 {
        return self.facts_offset + slot_id;
    }
};

// ---- Prolog Types ----

// pub const TermType = enum(i8) {
//     atom = 0,
//     variable = 1,
//     integer = 2,
//     vdr = 3,
//     text = 4,
//     list = 5,
//     compound = 6,
//     vector = 7,
//     matrix = 8,
//     pair = 9,
// };

pub const Term = extern struct {
    type: TermType,
    _pad0: [3]u8 = [_]u8{0} ** 3,
    primary_id: i32, // atom_id | var_id | int_value | functor_id
    secondary_offset: i32, // text_offset | list_head_offset | args_offset
    secondary_aux: i32, // text_length | list_tail_offset | args_count
    vdr_value: Q16, // for vdr type

    pub fn atom(id: i32) Term {
        return .{ .type = .atom, .primary_id = id, .secondary_offset = 0, .secondary_aux = 0, .vdr_value = Q16.zero() };
    }

    pub fn variable(id: i32) Term {
        return .{ .type = .variable, .primary_id = id, .secondary_offset = 0, .secondary_aux = 0, .vdr_value = Q16.zero() };
    }

    pub fn integer(val: i32) Term {
        return .{ .type = .integer, .primary_id = val, .secondary_offset = 0, .secondary_aux = 0, .vdr_value = Q16.zero() };
    }

    pub fn vdr(val: Q16) Term {
        return .{ .type = .vdr, .primary_id = 0, .secondary_offset = 0, .secondary_aux = 0, .vdr_value = val };
    }

    pub fn compound(functor_id: i32, args_offset: i32, args_count: i32) Term {
        return .{ .type = .compound, .primary_id = functor_id, .secondary_offset = args_offset, .secondary_aux = args_count, .vdr_value = Q16.zero() };
    }

    pub fn list(head_offset: i32, tail_offset: i32) Term {
        return .{ .type = .list, .primary_id = 0, .secondary_offset = head_offset, .secondary_aux = tail_offset, .vdr_value = Q16.zero() };
    }

    pub fn textRef(offset: i32, length: i32) Term {
        return .{ .type = .text, .primary_id = 0, .secondary_offset = offset, .secondary_aux = length, .vdr_value = Q16.zero() };
    }

    pub fn isAtom(self: Term) bool {
        return self.type == .atom;
    }

    pub fn isVariable(self: Term) bool {
        return self.type == .variable;
    }

    pub fn isCompound(self: Term) bool {
        return self.type == .compound;
    }
};

pub const Rule = extern struct {
    id: i32,
    head: i32,
    body_offset: i32,
    body_count: i16,
    _pad0: i16 = 0,
    action_offset: i32,
    action_count: i16,
    _pad1: i16 = 0,
    fire_count: i32,
    last_fired: i32,
    success_count: i32,
    failure_count: i32,
    created_at: i32,
    creator_session_id: i32,

    pub fn successRate(self: Rule) Q16 {
        const total = self.success_count + self.failure_count;
        if (total == 0) return Q16.zero();
        return Q16.fromParts(
            @intCast(@divTrunc(@as(i64, self.success_count) * Q16.D, @as(i64, total))),
            0,
        );
    }
};

pub const Binding = extern struct {
    var_id: i32,
    bound_term_offset: i32,
};

pub const UnificationResult = extern struct {
    unified: i8,
    _pad: [3]u8 = [_]u8{0} ** 3,
    bindings_offset: i32,
    bindings_count: i16,
    _pad2: i16 = 0,

    pub fn success(offset: i32, count: i16) UnificationResult {
        return .{ .unified = 1, .bindings_offset = offset, .bindings_count = count };
    }

    pub fn failure() UnificationResult {
        return .{ .unified = 0, .bindings_offset = -1, .bindings_count = 0 };
    }
};

// ---- Grammar Types ----

pub const SlotType = enum(i8) {
    vdr_value = 0,
    text = 1,
    integer = 2,
    @"enum" = 3,
    kb_ref = 4,
    grammar = 5,
};

pub const GrammarSlot = extern struct {
    name_offset: i32,
    name_length: i16,
    type: SlotType,
    _pad: i8 = 0,
    enum_values_offset: i32,
    enum_count: i16,
    _pad2: i16 = 0,
    kb_id: i32,
    kb_slot_id: i32,
};

pub const Grammar = extern struct {
    id: i32,
    template_offset: i32,
    template_length: i32,
    slots_offset: i32,
    slots_count: i16,
    validated: i8,
    _pad: i8 = 0,
    created_at: i32,
    creator_session_id: i32,

    pub fn isValid(self: Grammar) bool {
        return self.validated != 0;
    }
};

pub const GrammarFill = extern struct {
    slot_index: i16,
    fill_type: SlotType,
    _pad: i8 = 0,
    vdr_value: Q16,
    text_offset: i32,
    text_length: i16,
    _pad2: i16 = 0,
    int_value: i32,
    enum_index: i16,
    _pad3: i16 = 0,
};

pub const GrammarKbMapping = extern struct {
    slot_index: i16,
    _pad: i16 = 0,
    kb_id: i32,
    slot_id: i32,
};

// ---- Session Types ----

pub const SessionState = enum(i8) {
    active = 0,
    snapshotted = 1,
    killed = 2,
    frozen = 3,
};

pub const Session = extern struct {
    id: i32,
    user_id: i32,
    kb_root_id: i32,
    visibility_level: i8,
    state: SessionState,
    _pad0: i16 = 0,

    max_kb_count: i32,
    max_live_memory_bytes: i64,
    max_turns: i32,

    current_turn: i32,
    facts_asserted: i32,
    facts_retracted: i32,
    _pad1: i32 = 0,
    rules_fired: i64,
    prolog_queries: i64,
    primitive_calls: i64,
    grammar_renders: i64,
    llm_tokens_consumed: i64,
    command_tokens_consumed: i64,

    device_id: i32,
    stream_id: i32,
    kb_store_offset: i64,

    last_snapshot_id: i32,
    last_snapshot_timestamp: i32,

    parent_session_id: i32,
    clone_generation: i32,

    pub fn isActive(self: Session) bool {
        return self.state == .active;
    }

    pub fn isClone(self: Session) bool {
        return self.parent_session_id != -1;
    }

    pub fn hasSnapshot(self: Session) bool {
        return self.last_snapshot_id != -1;
    }

    pub fn turnsRemaining(self: Session) i32 {
        if (self.max_turns == 0) return -1; // unlimited
        return self.max_turns - self.current_turn;
    }
};

// ---- Runner Types ----

pub const RunnerType = enum(i8) {
    poller = 0,
    processor = 1,
    internal = 2,
    batch = 3,
};

pub const RunnerState = enum(i8) {
    stopped = 0,
    running = 1,
    err = 2,
    recycling = 3,
};

pub const Runner = extern struct {
    id: i32,
    type: RunnerType,
    state: RunnerState,
    _pad0: i16 = 0,
    session_id: i32,

    interval_ms: i32,
    max_turns_before_recycle: i32,
    max_consecutive_errors: i32,

    iterations_completed: i64,
    errors_consecutive: i32,
    _pad1: i32 = 0,
    errors_total: i64,
    last_iteration_ms: i32,
    last_iteration_timestamp: i32,

    recycle_count: i32,
    last_recycle_timestamp: i32,

    pub fn shouldRecycle(self: Runner) bool {
        if (self.type != .processor) return false;
        if (self.max_turns_before_recycle <= 0) return false;
        return self.iterations_completed >= self.max_turns_before_recycle;
    }

    pub fn shouldStop(self: Runner) bool {
        if (self.max_consecutive_errors <= 0) return false;
        return self.errors_consecutive >= self.max_consecutive_errors;
    }
};

// ---- Grant Types ----

pub const GrantClass = enum(i8) {
    filesystem = 0,
    compile = 1,
    execute = 2,
    lint = 3,
    network = 4,
    process = 5,
};

pub const GrantState = enum(i8) {
    active = 0,
    expired = 1,
    exhausted = 2,
    revoked = 3,
};

pub const Grant = extern struct {
    id: i32,
    class: GrantClass,
    state: GrantState,
    _pad0: i16 = 0,
    holder_user_id: i32,
    target_pattern_offset: i32,
    target_pattern_length: i16,
    _pad1: i16 = 0,
    max_uses: i32,
    remaining_uses: i32,
    expires_at: i32,
    created_at: i32,
    created_by: i32,
    revoked_at: i32,
    revoked_by: i32,

    pub fn isActive(self: Grant) bool {
        return self.state == .active;
    }

    pub fn isUnlimited(self: Grant) bool {
        return self.max_uses == -1;
    }

    pub fn isExpired(self: Grant, now: i32) bool {
        if (self.expires_at == 0) return false;
        return now >= self.expires_at;
    }

    pub fn isExhausted(self: Grant) bool {
        if (self.max_uses == -1) return false;
        return self.remaining_uses <= 0;
    }

    pub fn consumeUse(self: *Grant) bool {
        if (self.max_uses == -1) return true;
        if (self.remaining_uses <= 0) return false;
        self.remaining_uses -= 1;
        if (self.remaining_uses == 0) self.state = .exhausted;
        return true;
    }
};

// ---- Command Types ----

pub const CommandType = enum(i8) {
    kb_assert = 0,
    kb_query = 1,
    kb_retract = 2,
    prolog_query = 3,
    prolog_assert_rule = 4,
    builtin_call = 5,
    grammar_render = 6,
    direct_output = 7,
    op_filesystem = 8,
    op_compile = 9,
    op_execute = 10,
    op_network = 11,
    op_process = 12,
    session_snapshot = 13,
    session_clone = 14,
};

pub const Command = extern struct {
    type: CommandType,
    _pad0: [3]u8 = [_]u8{0} ** 3,
    target_kb_id: i32,
    target_slot_id: i32,
    builtin_id: i32,
    args_offset: i32,
    args_count: i16,
    grant_required: i8,
    _pad1: i8 = 0,

    pub fn requiresGrant(self: Command) bool {
        return self.grant_required >= 0;
    }

    pub fn grantClass(self: Command) ?GrantClass {
        if (self.grant_required < 0) return null;
        return @enumFromInt(self.grant_required);
    }

    pub fn isOperational(self: Command) bool {
        return switch (self.type) {
            .op_filesystem, .op_compile, .op_execute, .op_network, .op_process => true,
            else => false,
        };
    }
};

// ---- Audit Types ----

pub const AuditAction = enum(i8) {
    fact_assert = 0,
    fact_retract = 1,
    fact_query = 2,
    rule_fire = 3,
    rule_assert = 4,
    rule_retract = 5,
    grant_check = 6,
    grant_create = 7,
    grant_revoke = 8,
    session_create = 9,
    session_kill = 10,
    snapshot = 11,
    clone = 12,
    op_execute = 13,
    access_denied = 14,
};

pub const AuditEntry = extern struct {
    timestamp: i32,
    session_id: i32,
    user_id: i32,
    action: AuditAction,
    _pad0: [3]u8 = [_]u8{0} ** 3,
    target_kb_id: i32,
    target_slot_id: i32,
    grant_id: i32,
    result: i8,
    _pad1: [3]u8 = [_]u8{0} ** 3,
    detail_offset: i32,

    pub fn allowed(time: i32, session: i32, user: i32, action: AuditAction, kb: i32, slot: i32) AuditEntry {
        return .{
            .timestamp = time,
            .session_id = session,
            .user_id = user,
            .action = action,
            .target_kb_id = kb,
            .target_slot_id = slot,
            .grant_id = -1,
            .result = 1,
            .detail_offset = -1,
        };
    }

    pub fn denied(time: i32, session: i32, user: i32, action: AuditAction, kb: i32, slot: i32) AuditEntry {
        return .{
            .timestamp = time,
            .session_id = session,
            .user_id = user,
            .action = action,
            .target_kb_id = kb,
            .target_slot_id = slot,
            .grant_id = -1,
            .result = 0,
            .detail_offset = -1,
        };
    }
};

// ---- Confidence Table ----

pub const confidence_table = [11]Q16{
    .{ .v = 65536, .r0 = 0 }, // vdr_computation 1/1
    .{ .v = 65536, .r0 = 0 }, // prolog_derivation 1/1
    .{ .v = 64225, .r0 = 0 }, // database 98/100
    .{ .v = 62259, .r0 = 0 }, // prometheus 95/100
    .{ .v = 62259, .r0 = 0 }, // script 95/100
    .{ .v = 55705, .r0 = 0 }, // rest_api 85/100
    .{ .v = 52428, .r0 = 0 }, // published 80/100
    .{ .v = 45875, .r0 = 0 }, // user_stated 70/100
    .{ .v = 32768, .r0 = 0 }, // web_search 50/100
    .{ .v = 19660, .r0 = 0 }, // llm_generated 30/100
    .{ .v = 0, .r0 = 0 }, // unknown 0/1
};

// ---- Error / Status Types ----

pub const ErrorCategory = enum(i8) {
    none = 0,
    arithmetic = 1,
    kb = 2,
    prolog = 3,
    grammar = 4,
    session = 5,
    grant = 6,
    runner = 7,
    device = 8,
    system = 9,
};

pub const ErrorCode = enum(i32) {
    ok = 0,
    // Arithmetic
    division_by_zero = 100,
    overflow = 101,
    // KB
    kb_not_found = 200,
    kb_full = 201,
    kb_frozen = 202,
    kb_access_denied = 203,
    slot_out_of_range = 204,
    slot_empty = 205,
    // Prolog
    depth_exceeded = 300,
    no_matching_rule = 301,
    unification_failed = 302,
    max_bindings_exceeded = 303,
    // Grammar
    invalid_template = 400,
    slot_type_mismatch = 401,
    render_capacity_exceeded = 402,
    // Session
    session_limit = 500,
    snapshot_failed = 501,
    snapshot_corrupt = 502,
    clone_failed = 503,
    merge_conflict = 504,
    // Grant
    grant_denied = 600,
    grant_expired = 601,
    grant_exhausted = 602,
    grant_revoked = 603,
    grant_admin_required = 604,
    // Runner
    runner_error_threshold = 700,
    runner_connection_lost = 701,
    // Device
    device_not_found = 800,
    device_out_of_memory = 801,
    dispatch_failed = 802,
    // System
    init_failed = 900,
    corrupt_state = 901,
    seed_load_failed = 902,
};

pub const Status = extern struct {
    category: ErrorCategory,
    _pad0: [3]u8 = [_]u8{0} ** 3,
    code: ErrorCode,
    detail: i32,

    pub fn ok() Status {
        return .{ .category = .none, .code = .ok, .detail = 0 };
    }

    pub fn err(cat: ErrorCategory, code: ErrorCode, detail: i32) Status {
        return .{ .category = cat, .code = code, .detail = detail };
    }

    pub fn isOk(self: Status) bool {
        return self.category == .none;
    }

    pub fn isErr(self: Status) bool {
        return self.category != .none;
    }
};

pub const RecoveryAction = enum(i32) {
    none = 0,
    compact = 1,
    log_and_continue = 2,
    simplify_query = 3,
    retry_snapshot = 4,
    log_and_deny = 5,
    reconnect_with_backoff = 6,
    recycle_runner = 7,
    kill_oldest_clone = 8,
    restore_from_snapshot = 9,
};

pub fn recoverFromError(status: Status) RecoveryAction {
    return switch (status.code) {
        .kb_full => .compact,
        .kb_access_denied => .log_and_continue,
        .depth_exceeded => .simplify_query,
        .snapshot_failed => .retry_snapshot,
        .grant_denied, .grant_expired, .grant_exhausted, .grant_revoked => .log_and_deny,
        .runner_connection_lost => .reconnect_with_backoff,
        .runner_error_threshold => .recycle_runner,
        .device_out_of_memory => .kill_oldest_clone,
        .corrupt_state => .restore_from_snapshot,
        else => .none,
    };
}

// ---- Level Stats ----

pub const LevelStats = struct {
    l1_count: i64,
    l1_tokens: i64,
    l2_count: i64,
    l2_tokens: i64,
    l3_count: i64,

    pub fn totalCount(self: LevelStats) i64 {
        return self.l1_count + self.l2_count + self.l3_count;
    }

    pub fn autoTriageNum(self: LevelStats) i32 {
        return @intCast(self.l3_count);
    }

    pub fn autoTriageDen(self: LevelStats) i32 {
        return @intCast(self.totalCount());
    }

    pub fn avgTokensPerInteraction(self: LevelStats) Q16 {
        const total_tokens = self.l1_tokens + self.l2_tokens; // L3 is always 0
        const total_ops = self.totalCount();
        if (total_ops == 0) return Q16.zero();
        return Q16.fromParts(
            @intCast(@divTrunc(total_tokens * Q16.D, total_ops)),
            0,
        );
    }
};

// ---- Handle Types (opaque, host-only) ----

pub const SessionHandle = struct {
    id: i32,
    index: i32,
};

pub const SnapshotHandle = struct {
    id: i32,
    index: i32,
};

pub const RunnerHandle = struct {
    id: i32,
    index: i32,
};
