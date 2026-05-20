// ============================================================
// src/prolog/types.zig
// ============================================================

const q16_mod = @import("../vdr/q16.zig");
const vdr_types = @import("../vdr/types.zig");
const kb_types = @import("../kb/types.zig");

pub const Q16 = q16_mod.Q16;
pub const VlpStatus = vdr_types.VlpStatus;
pub const VlpTermType = vdr_types.VlpTermType;
pub const VlpFact = kb_types.VlpFact;
pub const VlpProvenance = kb_types.VlpProvenance;

pub const VlpTerm = struct {
    ttype: VlpTermType = .atom,
    d: TermData = .{ .atom_id = 0 },
};

pub const TermData = union {
    atom_id: i32,
    var_id: i32,
    int_value: i32,
    vdr_value: Q16,
    text: TextSpan,
    list: ListPair,
    compound: CompoundRef,
};

pub const TextSpan = struct {
    offset: i32 = 0,
    length: i16 = 0,
};

pub const ListPair = struct {
    head: i32 = -1,
    tail: i32 = -1,
};

pub const CompoundRef = struct {
    functor_id: i32 = 0,
    args_offset: i32 = 0,
    args_count: i16 = 0,
};

pub const VlpBinding = struct {
    var_id: i32,
    term: VlpTerm,
};

pub const BindingSet = struct {
    bindings: []VlpBinding,
    count: i32,

    pub fn init(backing: []VlpBinding) BindingSet {
        return .{ .bindings = backing, .count = 0 };
    }

    pub fn bind(self: *BindingSet, var_id: i32, term: VlpTerm) bool {
        if (self.count >= @as(i32, @intCast(self.bindings.len))) return false;
        self.bindings[@intCast(self.count)] = .{ .var_id = var_id, .term = term };
        self.count += 1;
        return true;
    }

    pub fn lookup(self: *const BindingSet, var_id: i32) ?VlpTerm {
        var i: i32 = self.count - 1;
        while (i >= 0) : (i -= 1) {
            if (self.bindings[@intCast(i)].var_id == var_id) return self.bindings[@intCast(i)].term;
        }
        return null;
    }

    pub fn checkpoint(self: *const BindingSet) i32 {
        return self.count;
    }

    pub fn undo(self: *BindingSet, cp: i32) void {
        self.count = cp;
    }

    pub fn clear(self: *BindingSet) void {
        self.count = 0;
    }
};

pub const VlpRule = struct {
    id: i32 = -1,
    head: i32 = -1,
    body_offset: i32 = -1,
    body_count: i16 = 0,
    action_offset: i32 = -1,
    action_count: i16 = 0,
    fire_count: i32 = 0,
    last_fired: i32 = 0,
    success_count: i32 = 0,
    failure_count: i32 = 0,
    created_at: i32 = 0,
    creator_session_id: i32 = -1,
    alive: bool = false,
};

pub const PrologActionType = enum(i8) {
    assert_fact = 0,
    retract_fact = 1,
    direct_output = 2,
};

pub const PrologAction = struct {
    atype: PrologActionType = .assert_fact,
    target_kb_id: i32 = -1,
    target_slot_id: i32 = -1,
    fact: VlpFact = .{},
};

pub const PrologFired = struct {
    rule_id: i32 = -1,
    bindings: [64]VlpBinding = undefined,
    binding_count: i32 = 0,
    actions: [16]PrologAction = undefined,
    action_count: i32 = 0,
    confidence: Q16 = Q16.zero(),
};

pub const QueryConfig = struct {
    max_depth: i32 = 100,
    max_results: i32 = 100,
};

pub const RuleStats = struct {
    fire_count: i32,
    last_fired: i32,
    success_count: i32,
    failure_count: i32,
};

pub const HygieneReason = enum(i8) {
    stale = 0,
    failing = 1,
    orphaned = 2,
};

pub const HygieneCandidate = struct {
    rule_id: i32,
    reason: HygieneReason,
    detail: i32,
};
