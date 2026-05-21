// ============================================================
// src/vdr/types.zig
// ============================================================

pub const VlpStatus = enum(i32) {
    ok = 0,
    err_not_init = 1,
    err_invalid_device = 2,
    err_out_of_memory = 3,
    err_invalid_qbasis = 4,
    err_qbasis_mismatch = 5,
    err_kb_not_found = 6,
    err_kb_access_denied = 7,
    err_kb_full = 8,
    err_kb_frozen = 9,
    err_slot_out_of_range = 10,
    err_grant_denied = 11,
    err_session_limit = 12,
    err_prolog_depth = 13,
    err_remainder_overflow = 14,
    err_grammar_invalid = 15,
    err_primitive_bounds = 16,
    err_snapshot_corrupt = 17,
    err_command_parse = 18,
    err_determinism = 19,
    err_auth_invalid = 20,
    err_auth_suspended = 21,
    err_rate_limited = 22,
    err_connection_closed = 23,
    err_timeout = 24,
    err_protocol_malformed = 25,
    err_prolog_no_match = 26,
    err_grammar_capacity = 27,
    err_grammar_type_mismatch = 28,
    err_runner_error_threshold = 29,
    err_runner_connection_loss = 30,
    err_division_by_zero = 31,
};

pub const VlpQBasis = enum(i32) {
    q16 = 16,
    q32 = 32,
    q335 = 335,
};

pub const VlpFactTag = enum(i8) {
    value = 0,
    text = 1,
    reference = 2,
    timestamp = 3,
    enum_val = 4,
    boolean = 5,
    vector = 6,
    matrix = 7,
    provenance = 8,
    rule_ref = 9,
    grammar_ref = 10,
    counter = 11,
    empty = -1,
};

pub const VlpSourceType = enum(i8) {
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

pub const VlpVisibility = enum(i8) {
    public = 0,
    internal = 1,
    owner_only = 2,
};

pub const VlpGrantClass = enum(i8) {
    filesystem = 0,
    compile = 1,
    execute = 2,
    lint = 3,
    network = 4,
    process = 5,
};

pub const VlpGrantState = enum(i8) {
    active = 0,
    expired = 1,
    exhausted = 2,
    revoked = 3,
};

pub const VlpSessionState = enum(i8) {
    active = 0,
    snapshotted = 1,
    killed = 2,
    frozen = 3,
};

pub const VlpTermType = enum(i8) {
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

pub const VlpSlotType = enum(i8) {
    vdr_value = 0,
    text = 1,
    integer = 2,
    enum_val = 3,
    kb_ref = 4,
    grammar = 5,
};

pub const VlpCommandType = enum(i8) {
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

pub const VlpAuditAction = enum(i8) {
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
    snapshot_action = 11,
    clone_action = 12,
    op_execute = 13,
    access_denied = 14,
};

pub const VlpRunnerType = enum(i8) {
    poller = 0,
    processor = 1,
    internal = 2,
    batch = 3,
};

pub const VlpRunnerState = enum(i8) {
    stopped = 0,
    running = 1,
    err = 2,
    recycling = 3,
};

pub const VlpTokenClass = enum(i8) {
    command_start = 0,
    direct_output = 1,
    end_of_turn = 2,
    prose = 3,
};

pub const VlpExecutionLevel = enum(i8) {
    l1 = 0,
    l2 = 1,
    l3 = 2,
};

pub const VlpMergePolicy = enum(i8) {
    ours = 0,
    theirs = 1,
    fail_on_conflict = 2,
};

pub const VlpProtocolType = enum(i8) {
    http = 0,
    websocket = 1,
    smtp = 2,
    mqtt = 3,
    raw_tcp = 4,
};

pub const VlpConnectionState = enum(i8) {
    handshake = 0,
    active = 1,
    draining = 2,
    closed = 3,
};

pub const VlpReduceOp = enum(i8) {
    sum = 0,
    max = 1,
    min = 2,
};
