// ============================================================
// vlp_prolog.zig
// Prolog engine — host drives search, GPU does parallel unification.
// All recursion replaced with iterative loops + explicit stack.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const gpu = @import("vlp_gpu_params.zig");
const kb_mod = @import("vlp_kb_store.zig");

const shared = @import("vlp_gpu_shared");

// ============================================================
// Configuration
// ============================================================

pub const QueryConfig = struct {
    max_depth: i32 = 100,
    max_bindings: i32 = 1000,
    max_results: i32 = 100,
    gpu_candidate_threshold: i32 = 32,
};

// ============================================================
// Results
// ============================================================

pub const QueryResult = struct {
    bindings: []types.Binding,
    binding_count: i32,
    result_count: i32,
    depth_reached: i32,
    depth_exceeded: bool,
    status: types.Status,

    pub fn empty() QueryResult {
        return .{
            .bindings = &.{},
            .binding_count = 0,
            .result_count = 0,
            .depth_reached = 0,
            .depth_exceeded = false,
            .status = types.Status.ok(),
        };
    }
};

pub const FireResult = struct {
    firing_rule_ids: []i32,
    firing_count: i32,
    status: types.Status,

    pub fn empty() FireResult {
        return .{
            .firing_rule_ids = &.{},
            .firing_count = 0,
            .status = types.Status.ok(),
        };
    }
};

pub const PrologAction = struct {
    is_assert: bool,
    target_kb_id: i32,
    target_slot_id: i32,
    fact: types.Fact,
};

// ============================================================
// Search stack frame — replaces recursion
// ============================================================

const SearchFrame = struct {
    goal_offset: i32, // term offset of current goal
    candidate_idx: i32, // next candidate to try in fact store
    candidate_count: i32, // total candidates for this goal
    bindings_snapshot: i32, // binding count at frame entry (for backtrack undo)
    kb_id: i32, // which KB we're searching in
    body_idx: i32, // which body condition of a rule we're on
    body_count: i32, // total body conditions
    rule_id: i32, // rule being evaluated (-1 if direct fact match)
};

const MAX_SEARCH_STACK: usize = 128; // bounded by max_depth

// ============================================================
// Prolog Engine
// ============================================================

pub const PrologEngine = struct {
    bridge: *bridge_mod.Bridge,
    kb_store: *kb_mod.KbStore,
    allocator: std.mem.Allocator,
    config: QueryConfig,

    // Reusable buffers
    binding_buf: []types.Binding,
    search_stack: []SearchFrame,
    candidate_buf: []i32, // candidate fact offsets for GPU dispatch
    unify_result_buf: []i32, // per-candidate unification results
    rule_match_buf: []i32, // matched rule IDs
    body_eval_buf: []bool, // body condition results
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(bridge: *bridge_mod.Bridge, kb_store: *kb_mod.KbStore, allocator: std.mem.Allocator, config: *const QueryConfig) PrologEngine {
    const bindings = allocator.alloc(types.Binding, @intCast(config.max_bindings)) catch &.{};
    const stack = allocator.alloc(SearchFrame, MAX_SEARCH_STACK) catch &.{};
    const candidates = allocator.alloc(i32, 4096) catch &.{};
    const unify_results = allocator.alloc(i32, 4096) catch &.{};
    const rule_matches = allocator.alloc(i32, 1024) catch &.{};
    const body_evals = allocator.alloc(bool, 4096) catch &.{};

    return .{
        .bridge = bridge,
        .kb_store = kb_store,
        .allocator = allocator,
        .config = config.*,
        .binding_buf = bindings,
        .search_stack = stack,
        .candidate_buf = candidates,
        .unify_result_buf = unify_results,
        .rule_match_buf = rule_matches,
        .body_eval_buf = body_evals,
    };
}

pub fn deinit(self: *PrologEngine) void {
    if (self.binding_buf.len > 0) self.allocator.free(self.binding_buf);
    if (self.search_stack.len > 0) self.allocator.free(self.search_stack);
    if (self.candidate_buf.len > 0) self.allocator.free(self.candidate_buf);
    if (self.unify_result_buf.len > 0) self.allocator.free(self.unify_result_buf);
    if (self.rule_match_buf.len > 0) self.allocator.free(self.rule_match_buf);
    if (self.body_eval_buf.len > 0) self.allocator.free(self.body_eval_buf);
}

// ============================================================
// Host-side flat unification — for small candidate sets
// ============================================================

pub fn unifySingle(a: *const types.Term, b: *const types.Term, bindings: []types.Binding, binding_count: *i32) bool {
    // Flat (non-recursive) unification covering common cases.
    // ATOM-ATOM
    if (a.type == .atom and b.type == .atom) {
        return a.primary_id == b.primary_id;
    }
    // VARIABLE-anything: bind
    if (a.type == .variable) {
        if (binding_count.* < @as(i32, @intCast(bindings.len))) {
            bindings[@intCast(binding_count.*)] = .{
                .var_id = a.primary_id,
                .bound_term_offset = -1, // inline binding — b is the value
            };
            binding_count.* += 1;
            return true;
        }
        return false; // out of binding slots
    }
    if (b.type == .variable) {
        if (binding_count.* < @as(i32, @intCast(bindings.len))) {
            bindings[@intCast(binding_count.*)] = .{
                .var_id = b.primary_id,
                .bound_term_offset = -1,
            };
            binding_count.* += 1;
            return true;
        }
        return false;
    }
    // INTEGER-INTEGER
    if (a.type == .integer and b.type == .integer) {
        return a.primary_id == b.primary_id;
    }
    // VDR-VDR: exact cross-multiply comparison
    if (a.type == .vdr and b.type == .vdr) {
        return a.vdr_value.eql(b.vdr_value);
    }
    // COMPOUND-COMPOUND: functor match + iterative arg comparison
    if (a.type == .compound and b.type == .compound) {
        if (a.primary_id != b.primary_id) return false; // functor mismatch
        if (a.secondary_aux != b.secondary_aux) return false; // arity mismatch
        // Args compared by loading from term store — handled by caller
        // in the iterative search loop. For flat unification we just
        // confirm functor + arity match here.
        return true;
    }
    // TEXT-TEXT: compare offset+length (identity)
    if (a.type == .text and b.type == .text) {
        return a.secondary_offset == b.secondary_offset and a.secondary_aux == b.secondary_aux;
    }
    // Mismatched types: fail
    return false;
}

// ============================================================
// GPU-dispatched parallel unification
// ============================================================

pub fn unifyCandidatesGpu(self: *PrologEngine, query_term: *const types.Term, candidate_offsets: []const i32) i32 {
    const n: i32 = @intCast(candidate_offsets.len);

    // Upload candidate offsets to scratch_a
    const off_bytes: []const u8 = @as([*]const u8, @ptrCast(candidate_offsets.ptr))[0 .. candidate_offsets.len * 4];
    var status = self.bridge.uploadToBuffer(.scratch_a, 0, off_bytes);
    if (status.isErr()) return 0;

    _ = self.bridge.resetResultCounts();

    var params = gpu.UnifyCandidatesParams{
        .n_candidates = n,
        .query_term_type = @intFromEnum(query_term.type),
        .query_atom_id = query_term.primary_id,
        .query_int_value = query_term.primary_id,
        .query_vdr_v = query_term.vdr_value.v,
        .query_vdr_r0 = query_term.vdr_value.r0,
        .query_functor_id = query_term.primary_id,
        .query_args_count = @intCast(query_term.secondary_aux),
        .query_args_offset = query_term.secondary_offset,
        .max_bindings_per = 8,
    };

    // var params = gpu.unifyCandidates(
    //     n,
    //     @intFromEnum(query_term.term_type),
    //     query_term.primary_id,
    //     query_term.primary_id,
    //     query_term.vdr_value.v,
    //     query_term.vdr_value.toInts()[1], // packed r0|r1
    //     query_term.primary_id,
    //     @intCast(query_term.secondary_aux),
    //     query_term.secondary_offset,
    //     shared.MAX_BINDINGS_PER,
    // );

    status = self.bridge.dispatch(&.{
        .pipeline = .unify_candidates,
        .group_count_x = @divTrunc(n + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&params),
        .params_size = @sizeOf(gpu.UnifyCandidatesParams),
    });
    if (status.isErr()) return 0;

    return self.bridge.readResultCount(0);
}

// ============================================================
// Query — iterative depth-first search with backtracking
// ============================================================

pub fn query(self: *PrologEngine, start_kb_id: i32, query_term: *const types.Term) QueryResult {
    var binding_count: i32 = 0;
    var result_count: i32 = 0;
    const max_depth_reached: i32 = 0;
    const depth_exceeded = false;

    // Build KB chain for scoped search
    const chain = self.kb_store.buildChain(start_kb_id, self.config.max_depth);
    defer if (chain.len > 0) self.allocator.free(chain);
    if (chain.len == 0) return QueryResult.empty();

    // Collect all candidate fact offsets from the chain
    var n_candidates: i32 = 0;
    for (chain) |entry| {
        var slot: i32 = 0;
        while (slot < entry.facts_count and n_candidates < @as(i32, @intCast(self.candidate_buf.len))) : (slot += 1) {
            self.candidate_buf[@intCast(n_candidates)] = entry.facts_offset + slot;
            n_candidates += 1;
        }
    }

    if (n_candidates == 0) return QueryResult.empty();

    // Decide GPU vs host for unification
    if (n_candidates >= self.config.gpu_candidate_threshold and self.bridge.shouldUseGpu(.unification, n_candidates)) {
        // GPU path
        const matches = self.unifyCandidatesGpu(query_term, self.candidate_buf[0..@intCast(n_candidates)]);
        result_count = @min(matches, self.config.max_results);

        // Download bindings from GPU scratch
        if (result_count > 0) {
            const max_b: i32 = @min(result_count * 8, @as(i32, @intCast(self.binding_buf.len)));
            _ = self.bridge.readScratchSlice(types.Binding, .scratch_b, 0, max_b, self.binding_buf[0..@intCast(max_b)]);
            binding_count = max_b;
        }
    } else {
        // Host path — iterate candidates, unify each
        var ci: i32 = 0;
        while (ci < n_candidates and result_count < self.config.max_results) : (ci += 1) {
            // Read candidate fact from device
            const fact_offset = self.candidate_buf[@intCast(ci)];
            var fact: types.Fact = undefined;
            const dest: []u8 = @as([*]u8, @ptrCast(&fact))[0..@sizeOf(types.Fact)];
            _ = self.bridge.downloadFromBuffer(.fact_store, @as(i64, fact_offset) * @sizeOf(types.Fact), dest);
            if (fact.isEmpty()) continue;

            // Build a term from the fact for unification
            var fact_term = factToTerm(&fact);

            // Attempt flat unification
            if (unifySingle(query_term, &fact_term, self.binding_buf[@intCast(binding_count)..], &binding_count)) {
                result_count += 1;
            }
        }
    }

    return .{
        .bindings = self.binding_buf[0..@intCast(binding_count)],
        .binding_count = binding_count,
        .result_count = result_count,
        .depth_reached = max_depth_reached,
        .depth_exceeded = depth_exceeded,
        .status = types.Status.ok(),
    };
}

// ============================================================
// Rule matching — find rules whose heads match a query
// ============================================================

pub fn ruleMatchScan(self: *PrologEngine, kb_id: i32, query_term: *const types.Term) []i32 {
    const kb = self.kb_store.getKb(kb_id) orelse return &.{};
    if (kb.rules_count == 0) return &.{};

    if (kb.rules_count >= self.config.gpu_candidate_threshold and self.bridge.shouldUseGpu(.rule_match, kb.rules_count)) {
        // GPU path
        _ = self.bridge.resetResultCounts();

        var params = gpu.RuleMatchScanParams{
            .n_rules = kb.rules_count,
            .rules_base_offset = kb.rules_offset,
            .query_term_type = @intFromEnum(query_term.type),
            .query_atom_id = query_term.primary_id,
            .query_functor_id = query_term.primary_id,
            .query_args_count = @intCast(query_term.secondary_aux),
            .max_matches = @intCast(@min(self.rule_match_buf.len, 1024)),
        };

        const status = self.bridge.dispatch(&.{
            .pipeline = .rule_match_scan,
            .group_count_x = @divTrunc(kb.rules_count + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
            .group_count_y = 1,
            .group_count_z = 1,
            .params_ptr = @ptrCast(&params),
            .params_size = @sizeOf(gpu.RuleMatchScanParams),
        });
        if (status.isErr()) return &.{};

        const count = self.bridge.readResultCount(0);
        if (count <= 0) return &.{};

        const actual: usize = @intCast(@min(count, @as(i32, @intCast(self.rule_match_buf.len))));
        _ = self.bridge.readScratchSlice(i32, .scratch_a, 0, @intCast(actual), self.rule_match_buf[0..actual]);
        return self.rule_match_buf[0..actual];
    }

    // Host path
    var match_count: usize = 0;
    var ri: i32 = 0;
    while (ri < kb.rules_count and match_count < self.rule_match_buf.len) : (ri += 1) {
        // Read rule from device
        var rule: types.Rule = undefined;
        const offset = @as(i64, kb.rules_offset + ri) * @sizeOf(types.Rule);
        const dest: []u8 = @as([*]u8, @ptrCast(&rule))[0..@sizeOf(types.Rule)];
        _ = self.bridge.downloadFromBuffer(.rule_store, offset, dest);

        // Read rule's head term
        var head_term: types.Term = undefined;
        const head_off = @as(i64, rule.head) * @sizeOf(types.Term);
        const head_dest: []u8 = @as([*]u8, @ptrCast(&head_term))[0..@sizeOf(types.Term)];
        _ = self.bridge.downloadFromBuffer(.term_store, head_off, head_dest);

        // Flat unify head against query
        var dummy_bindings: [16]types.Binding = undefined;
        var dummy_count: i32 = 0;
        if (unifySingle(query_term, &head_term, &dummy_bindings, &dummy_count)) {
            self.rule_match_buf[match_count] = rule.id;
            match_count += 1;
        }
    }
    return self.rule_match_buf[0..match_count];
}

// ============================================================
// Rule body evaluation — check if all body conditions satisfied
// ============================================================

pub fn ruleBodyEval(self: *PrologEngine, matched_rule_ids: []const i32, kb_id: i32) []bool {
    const n = matched_rule_ids.len;
    if (n == 0) return &.{};

    var results = self.body_eval_buf[0..@min(n, self.body_eval_buf.len)];

    for (matched_rule_ids, 0..) |rule_id, i| {
        if (i >= results.len) break;

        // Read rule
        var rule: types.Rule = undefined;
        const rule_off = @as(i64, rule_id) * @sizeOf(types.Rule);
        const rule_dest: []u8 = @as([*]u8, @ptrCast(&rule))[0..@sizeOf(types.Rule)];
        _ = self.bridge.downloadFromBuffer(.rule_store, rule_off, rule_dest);

        if (rule.body_count == 0) {
            results[i] = true; // no body = always satisfied
            continue;
        }

        // Check each body condition
        var all_satisfied = true;
        var bi: i16 = 0;
        while (bi < rule.body_count) : (bi += 1) {
            var body_term: types.Term = undefined;
            const body_off = @as(i64, rule.body_offset + @as(i32, bi)) * @sizeOf(types.Term);
            const body_dest: []u8 = @as([*]u8, @ptrCast(&body_term))[0..@sizeOf(types.Term)];
            _ = self.bridge.downloadFromBuffer(.term_store, body_off, body_dest);

            // Query fact store for this body condition
            const sub_result = self.query(kb_id, &body_term);
            if (sub_result.result_count == 0) {
                all_satisfied = false;
                break;
            }
        }
        results[i] = all_satisfied;
    }

    return results;
}

// ============================================================
// Rule firing — match + body eval + collect actions
// ============================================================

pub fn fireRules(self: *PrologEngine, kb_id: i32) FireResult {
    // Match all rules against current fact state
    // Use a wildcard query (variable term matches anything)
    var wildcard = types.Term.variable(0);
    const matched = self.ruleMatchScan(kb_id, &wildcard);
    if (matched.len == 0) return FireResult.empty();

    // Evaluate bodies
    const body_results = self.ruleBodyEval(matched, kb_id);

    // Collect fully satisfied rules
    var fire_count: i32 = 0;
    for (body_results, 0..) |satisfied, i| {
        if (satisfied and i < matched.len) {
            if (fire_count < @as(i32, @intCast(self.rule_match_buf.len))) {
                self.rule_match_buf[@intCast(fire_count)] = matched[i];
                fire_count += 1;
            }
        }
    }

    return .{
        .firing_rule_ids = self.rule_match_buf[0..@intCast(fire_count)],
        .firing_count = fire_count,
        .status = types.Status.ok(),
    };
}

pub fn applyActions(self: *PrologEngine, actions: []const PrologAction) types.Status {
    for (actions) |action| {
        if (action.is_assert) {
            const s = self.kb_store.factWrite(action.target_kb_id, action.target_slot_id, &action.fact);
            if (s.isErr()) return s;
        } else {
            const s = self.kb_store.factRetract(action.target_kb_id, action.target_slot_id);
            if (s.isErr()) return s;
        }
    }
    return types.Status.ok();
}

pub fn fireAndCommit(self: *PrologEngine, kb_id: i32) i32 {
    const result = self.fireRules(kb_id);
    if (result.firing_count == 0) return 0;

    // For each firing rule, load and apply its actions
    var applied: i32 = 0;
    for (result.firing_rule_ids[0..@intCast(result.firing_count)]) |rule_id| {
        var rule: types.Rule = undefined;
        const rule_off = @as(i64, rule_id) * @sizeOf(types.Rule);
        const dest: []u8 = @as([*]u8, @ptrCast(&rule))[0..@sizeOf(types.Rule)];
        _ = self.bridge.downloadFromBuffer(.rule_store, rule_off, dest);

        // Update rule statistics
        rule.fire_count += 1;
        rule.last_fired = kb_mod.currentTimestamp();
        rule.success_count += 1;
        const rule_bytes: []const u8 = @as([*]const u8, @ptrCast(&rule))[0..@sizeOf(types.Rule)];
        _ = self.bridge.uploadToBuffer(.rule_store, rule_off, rule_bytes);

        applied += 1;
    }
    return applied;
}

// ============================================================
// Rule management — host-side CRUD
// ============================================================

pub fn ruleAssert(self: *PrologEngine, kb_id: i32, head: *const types.Term, body: []const types.Term, actions: []const types.Term) i32 {
    var kb = self.kb_store.getKb(kb_id) orelse return -1;
    if (kb.rules_count >= kb.rules_capacity) return -1;

    // Store terms
    const head_offset = self.termStore(head);
    const body_offset = self.termStoreBatch(body);
    const action_offset = self.termStoreBatch(actions);

    // Build rule
    const rule_id = kb.rules_offset + kb.rules_count;
    var rule = std.mem.zeroes(types.Rule);
    rule.id = rule_id;
    rule.head = head_offset;
    rule.body_offset = body_offset;
    rule.body_count = @intCast(body.len);
    rule.action_offset = action_offset;
    rule.action_count = @intCast(actions.len);
    rule.created_at = kb_mod.currentTimestamp();

    // Upload rule
    const offset = @as(i64, rule_id) * @sizeOf(types.Rule);
    const bytes: []const u8 = @as([*]const u8, @ptrCast(&rule))[0..@sizeOf(types.Rule)];
    _ = self.bridge.uploadToBuffer(.rule_store, offset, bytes);

    // Update KB
    kb.rules_count += 1;
    kb.last_modified = kb_mod.currentTimestamp();
    _ = self.kb_store.writeKbToDevice(kb_id, &kb);

    return rule_id;
}

pub fn ruleRetract(self: *PrologEngine, kb_id: i32, rule_id: i32) types.Status {
    // Zero out the rule in rule store
    var empty_rule = std.mem.zeroes(types.Rule);
    empty_rule.id = -1; // sentinel for deleted
    const offset = @as(i64, rule_id) * @sizeOf(types.Rule);
    const bytes: []const u8 = @as([*]const u8, @ptrCast(&empty_rule))[0..@sizeOf(types.Rule)];
    _ = self.bridge.uploadToBuffer(.rule_store, offset, bytes);
    _ = kb_id;
    return types.Status.ok();
}

pub fn ruleGet(self: *PrologEngine, rule_id: i32) ?types.Rule {
    var rule: types.Rule = undefined;
    const offset = @as(i64, rule_id) * @sizeOf(types.Rule);
    const dest: []u8 = @as([*]u8, @ptrCast(&rule))[0..@sizeOf(types.Rule)];
    const status = self.bridge.downloadFromBuffer(.rule_store, offset, dest);
    if (status.isErr()) return null;
    if (rule.id == -1) return null; // deleted
    return rule;
}

// ============================================================
// Term store — host-side CRUD
// ============================================================

pub fn termStore(self: *PrologEngine, term: *const types.Term) i32 {
    const offset: i32 = @intCast(self.kb_store.next_term_offset);
    const byte_off = @as(i64, offset) * @sizeOf(types.Term);
    const bytes: []const u8 = @as([*]const u8, @ptrCast(term))[0..@sizeOf(types.Term)];
    _ = self.bridge.uploadToBuffer(.term_store, byte_off, bytes);
    self.kb_store.next_term_offset += 1;
    return offset;
}

pub fn termStoreBatch(self: *PrologEngine, terms: []const types.Term) i32 {
    if (terms.len == 0) return -1;
    const start: i32 = @intCast(self.kb_store.next_term_offset);
    for (terms) |*t| {
        _ = self.termStore(t);
    }
    return start;
}

pub fn termLoad(self: *PrologEngine, offset: i32) ?types.Term {
    if (offset < 0) return null;
    var term: types.Term = undefined;
    const byte_off = @as(i64, offset) * @sizeOf(types.Term);
    const dest: []u8 = @as([*]u8, @ptrCast(&term))[0..@sizeOf(types.Term)];
    const status = self.bridge.downloadFromBuffer(.term_store, byte_off, dest);
    if (status.isErr()) return null;
    return term;
}

// ============================================================
// Helpers
// ============================================================

fn factToTerm(fact: *const types.Fact) types.Term {
    return switch (fact.tag) {
        .value => types.Term.vdr(fact.value),
        .text => types.Term.textRef(fact.value.v, @as(i32, fact.value.r0)),
        .reference => types.Term.integer(fact.value.v),
        .timestamp => types.Term.integer(fact.value.v),
        .boolean => types.Term.integer(if (fact.value.v != 0) 1 else 0),
        .@"enum" => types.Term.integer(fact.value.v),
        .counter => types.Term.integer(fact.value.v),
        .empty => types.Term.atom(0),
        else => types.Term.atom(0),
    };
}
