// ============================================================
// src/prolog/query.zig
// ============================================================

const prolog_types = @import("types.zig");
const term_mod = @import("term.zig");
const unify_mod = @import("unify.zig");
const kb_types = @import("../kb/types.zig");
const store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");

const VlpTerm = prolog_types.VlpTerm;
const VlpTermType = prolog_types.VlpTermType;
const BindingSet = prolog_types.BindingSet;
const VlpBinding = prolog_types.VlpBinding;
const QueryConfig = prolog_types.QueryConfig;
const VlpFact = kb_types.VlpFact;
const KBStore = store_mod.KBStore;

pub const QueryResult = struct {
    binding_sets: []BindingSetCopy,
    count: i32,
};

pub const BindingSetCopy = struct {
    bindings: [64]VlpBinding = undefined,
    count: i32 = 0,
};

pub fn query(
    store: *const KBStore,
    term_store: []const VlpTerm,
    start_kb_id: i32,
    goal_idx: i32,
    config: QueryConfig,
    result: *QueryResult,
) void {
    if (goal_idx < 0 or goal_idx >= @as(i32, @intCast(term_store.len))) return;

    var binding_buf: [256]VlpBinding = undefined;
    var bindings = BindingSet.init(&binding_buf);

    queryInner(store, term_store, start_kb_id, goal_idx, config.max_depth, &bindings, result, config.max_results);
}

fn queryInner(
    store: *const KBStore,
    term_store: []const VlpTerm,
    kb_id: i32,
    goal_idx: i32,
    depth: i32,
    bindings: *BindingSet,
    result: *QueryResult,
    max_results: i32,
) void {
    if (depth <= 0) return;
    if (result.count >= max_results) return;
    if (result.count >= @as(i32, @intCast(result.binding_sets.len))) return;

    const goal = term_store[@intCast(goal_idx)];
    const resolved = term_mod.resolve(goal, bindings);

    var cur_kb = kb_id;
    var walk: i32 = 0;
    while (cur_kb >= 0 and walk < 100) : (walk += 1) {
        const kb = store.getKBConst(cur_kb) orelse break;
        const s: usize = @intCast(kb.facts_offset);
        const e: usize = @intCast(kb.facts_offset + kb.facts_capacity);

        for (store.facts[s..e]) |f| {
            if (result.count >= max_results) return;
            if (f.tag == .empty) continue;

            const fact_term = factToTerm(f);
            const cp = bindings.checkpoint();

            if (unify_mod.unify(resolved, fact_term, bindings, term_store, depth - 1)) {
                copyBindings(bindings, result);
                bindings.undo(cp);
            } else {
                bindings.undo(cp);
            }
        }
        cur_kb = kb.parent_id;
    }
}

fn factToTerm(f: VlpFact) VlpTerm {
    return switch (f.tag) {
        .value => .{ .ttype = .vdr, .d = .{ .vdr_value = f.value } },
        .integer, .timestamp, .counter => .{ .ttype = .integer, .d = .{ .int_value = f.value.v } },
        .boolean => .{ .ttype = .integer, .d = .{ .int_value = if (f.value.v != 0) 1 else 0 } },
        .text => .{ .ttype = .text, .d = .{ .text = .{ .offset = f.value.v, .length = f.value.r0 } } },
        .reference => .{ .ttype = .integer, .d = .{ .int_value = f.value.v } },
        .enum_val => .{ .ttype = .integer, .d = .{ .int_value = f.value.v } },
        else => .{ .ttype = .atom, .d = .{ .atom_id = 0 } },
    };
}

fn copyBindings(bindings: *const BindingSet, result: *QueryResult) void {
    if (result.count >= @as(i32, @intCast(result.binding_sets.len))) return;
    var copy = &result.binding_sets[@intCast(result.count)];
    copy.count = bindings.count;
    var i: i32 = 0;
    while (i < bindings.count and i < 64) : (i += 1) {
        copy.bindings[@intCast(i)] = bindings.bindings[@intCast(i)];
    }
    result.count += 1;
}

pub fn queryCompound(
    store: *const KBStore,
    term_store: []const VlpTerm,
    rule_heads: []const i32,
    rule_bodies: []const RuleBody,
    start_kb_id: i32,
    goal_idx: i32,
    config: QueryConfig,
    result: *QueryResult,
) void {
    if (goal_idx < 0 or goal_idx >= @as(i32, @intCast(term_store.len))) return;

    var binding_buf: [256]VlpBinding = undefined;
    var bindings = BindingSet.init(&binding_buf);

    // first try facts
    queryInner(store, term_store, start_kb_id, goal_idx, config.max_depth, &bindings, result, config.max_results);

    // then try rules
    const goal = term_store[@intCast(goal_idx)];

    for (rule_heads, 0..) |head_idx, ri| {
        if (result.count >= config.max_results) return;
        if (head_idx < 0 or head_idx >= @as(i32, @intCast(term_store.len))) continue;

        const head = term_store[@intCast(head_idx)];
        const cp = bindings.checkpoint();

        if (unify_mod.unify(goal, head, &bindings, term_store, config.max_depth)) {
            const body = rule_bodies[ri];
            if (body.count == 0) {
                copyBindings(&bindings, result);
            } else {
                var all_match = true;
                var bi: i32 = 0;
                while (bi < body.count) : (bi += 1) {
                    const body_goal = body.goals[@intCast(bi)];
                    var sub_result = QueryResult{
                        .binding_sets = result.binding_sets[0..0],
                        .count = 0,
                    };
                    var sub_buf: [16]BindingSetCopy = undefined;
                    sub_result.binding_sets = &sub_buf;

                    queryInner(store, term_store, start_kb_id, body_goal, config.max_depth - 1, &bindings, &sub_result, 1);
                    if (sub_result.count == 0) {
                        all_match = false;
                        break;
                    }
                    // apply first sub-result bindings
                    var si: i32 = 0;
                    while (si < sub_result.binding_sets[0].count) : (si += 1) {
                        const b = sub_result.binding_sets[0].bindings[@intCast(si)];
                        _ = bindings.bind(b.var_id, b.term);
                    }
                }
                if (all_match) {
                    copyBindings(&bindings, result);
                }
            }
        }
        bindings.undo(cp);
    }
}

pub const RuleBody = struct {
    goals: [16]i32 = .{-1} ** 16,
    count: i32 = 0,
};
