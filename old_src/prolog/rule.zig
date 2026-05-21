// ============================================================
// src/prolog/rule.zig
// ============================================================

const prolog_types = @import("types.zig");
const query_mod = @import("query.zig");
const unify_mod = @import("unify.zig");
const term_mod = @import("term.zig");
const kb_types = @import("../kb/types.zig");
const store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");

const VlpTerm = prolog_types.VlpTerm;
const VlpRule = prolog_types.VlpRule;
const VlpBinding = prolog_types.VlpBinding;
const BindingSet = prolog_types.BindingSet;
const PrologAction = prolog_types.PrologAction;
const PrologFired = prolog_types.PrologFired;
const RuleStats = prolog_types.RuleStats;
const QueryConfig = prolog_types.QueryConfig;
const VlpFact = kb_types.VlpFact;
const KBStore = store_mod.KBStore;
const RuleBody = query_mod.RuleBody;

pub const RuleStore = struct {
    rules: []VlpRule,
    rule_count: i32,

    terms: []VlpTerm,
    term_count: i32,

    actions: []PrologAction,
    action_count: i32,

    pub fn init(rules: []VlpRule, terms: []VlpTerm, actions: []PrologAction) RuleStore {
        for (rules) |*r| r.* = VlpRule{};
        for (terms) |*t| t.* = VlpTerm{};
        for (actions) |*a| a.* = PrologAction{};
        return .{
            .rules = rules,
            .rule_count = 0,
            .terms = terms,
            .term_count = 0,
            .actions = actions,
            .action_count = 0,
        };
    }

    pub fn addTerm(self: *RuleStore, t: VlpTerm) ?i32 {
        if (self.term_count >= @as(i32, @intCast(self.terms.len))) return null;
        const idx = self.term_count;
        self.terms[@intCast(idx)] = t;
        self.term_count += 1;
        return idx;
    }

    pub fn addTerms(self: *RuleStore, ts: []const VlpTerm) ?i32 {
        if (self.term_count + @as(i32, @intCast(ts.len)) > @as(i32, @intCast(self.terms.len))) return null;
        const offset = self.term_count;
        for (ts) |t| {
            self.terms[@intCast(self.term_count)] = t;
            self.term_count += 1;
        }
        return offset;
    }

    pub fn addActions(self: *RuleStore, acts: []const PrologAction) ?i32 {
        if (self.action_count + @as(i32, @intCast(acts.len)) > @as(i32, @intCast(self.actions.len))) return null;
        const offset = self.action_count;
        for (acts) |a| {
            self.actions[@intCast(self.action_count)] = a;
            self.action_count += 1;
        }
        return offset;
    }

    pub fn assertRule(self: *RuleStore, head_idx: i32, body: []const i32, acts: []const PrologAction, now: i32, session_id: i32) ?i32 {
        if (self.rule_count >= @as(i32, @intCast(self.rules.len))) return null;

        var body_offset: i32 = -1;
        if (body.len > 0) {
            var body_terms: [16]VlpTerm = undefined;
            for (body, 0..) |b, i| {
                body_terms[i] = .{ .ttype = .integer, .d = .{ .int_value = b } };
            }
            body_offset = self.addTerms(body_terms[0..body.len]) orelse return null;
        }

        var act_offset: i32 = -1;
        if (acts.len > 0) {
            act_offset = self.addActions(acts) orelse return null;
        }

        const id = self.rule_count;
        self.rules[@intCast(id)] = .{
            .id = id,
            .head = head_idx,
            .body_offset = body_offset,
            .body_count = @intCast(body.len),
            .action_offset = act_offset,
            .action_count = @intCast(acts.len),
            .created_at = now,
            .creator_session_id = session_id,
            .alive = true,
        };
        self.rule_count += 1;
        return id;
    }

    pub fn retractRule(self: *RuleStore, rule_id: i32) bool {
        if (rule_id < 0 or rule_id >= self.rule_count) return false;
        var r = &self.rules[@intCast(rule_id)];
        if (!r.alive) return false;
        r.alive = false;
        return true;
    }

    pub fn getStats(self: *const RuleStore, rule_id: i32) ?RuleStats {
        if (rule_id < 0 or rule_id >= self.rule_count) return null;
        const r = &self.rules[@intCast(rule_id)];
        if (!r.alive) return null;
        return .{
            .fire_count = r.fire_count,
            .last_fired = r.last_fired,
            .success_count = r.success_count,
            .failure_count = r.failure_count,
        };
    }

    pub fn fireAll(self: *RuleStore, kb_store: *const KBStore, scope_kb_id: i32, now: i32, fired: []PrologFired) i32 {
        var n_fired: i32 = 0;
        const max: i32 = @intCast(fired.len);

        var ri: i32 = 0;
        while (ri < self.rule_count) : (ri += 1) {
            if (n_fired >= max) break;
            var rule = &self.rules[@intCast(ri)];
            if (!rule.alive) continue;
            if (rule.head < 0 or rule.head >= self.term_count) continue;

            const head = self.terms[@intCast(rule.head)];

            var binding_buf: [256]VlpBinding = undefined;
            var bindings = BindingSet.init(&binding_buf);

            var matched = false;

            // try matching head against facts in scope
            var cur_kb = scope_kb_id;
            var walk: i32 = 0;
            while (cur_kb >= 0 and walk < 100) : (walk += 1) {
                const kb = kb_store.getKBConst(cur_kb) orelse break;
                const s: usize = @intCast(kb.facts_offset);
                const e: usize = @intCast(kb.facts_offset + kb.facts_capacity);

                for (kb_store.facts[s..e]) |f| {
                    if (f.tag == .empty) continue;
                    const ft = query_mod.factToTerm(f);
                    const cp = bindings.checkpoint();

                    if (unify_mod.unify(head, ft, &bindings, self.terms[0..@intCast(self.term_count)], 50)) {
                        // check body conditions
                        var body_ok = true;
                        if (rule.body_count > 0) {
                            var bi: i32 = 0;
                            while (bi < rule.body_count) : (bi += 1) {
                                const body_term_idx = self.terms[@intCast(rule.body_offset + bi)].d.int_value;
                                if (!matchBodyGoal(kb_store, self, scope_kb_id, body_term_idx, &bindings)) {
                                    body_ok = false;
                                    break;
                                }
                            }
                        }

                        if (body_ok) {
                            matched = true;
                            var entry = &fired[@intCast(n_fired)];
                            entry.rule_id = rule.id;
                            entry.binding_count = bindings.count;
                            var bi2: i32 = 0;
                            while (bi2 < bindings.count and bi2 < 64) : (bi2 += 1) {
                                entry.bindings[@intCast(bi2)] = bindings.bindings[@intCast(bi2)];
                            }
                            entry.action_count = @as(i32, rule.action_count);
                            var ai: i32 = 0;
                            while (ai < rule.action_count and ai < 16) : (ai += 1) {
                                entry.actions[@intCast(ai)] = self.actions[@intCast(rule.action_offset + ai)];
                            }
                            n_fired += 1;
                            rule.fire_count += 1;
                            rule.last_fired = now;
                            rule.success_count += 1;
                            bindings.undo(cp);
                            break;
                        }
                    }
                    bindings.undo(cp);
                }
                if (matched) break;
                cur_kb = kb.parent_id;
            }

            if (!matched) {
                rule.failure_count += 1;
            }
        }
        return n_fired;
    }

    pub fn fireAndCommit(self: *RuleStore, kb_store: *KBStore, scope_kb_id: i32, now: i32, fired_buf: []PrologFired) i32 {
        const n = self.fireAll(kb_store, scope_kb_id, now, fired_buf);
        var i: i32 = 0;
        while (i < n) : (i += 1) {
            const entry = &fired_buf[@intCast(i)];
            var ai: i32 = 0;
            while (ai < entry.action_count) : (ai += 1) {
                const act = &entry.actions[@intCast(ai)];
                switch (act.atype) {
                    .assert_fact => {
                        const slot = fact_mod.firstEmpty(kb_store, act.target_kb_id) orelse continue;
                        _ = fact_mod.assert(kb_store, act.target_kb_id, slot, &act.fact);
                    },
                    .retract_fact => {
                        _ = fact_mod.retract(kb_store, act.target_kb_id, act.target_slot_id);
                    },
                    .direct_output => {},
                }
            }
        }
        return n;
    }

    pub fn aliveCount(self: *const RuleStore) i32 {
        var c: i32 = 0;
        var i: i32 = 0;
        while (i < self.rule_count) : (i += 1) {
            if (self.rules[@intCast(i)].alive) c += 1;
        }
        return c;
    }
};

fn matchBodyGoal(kb_store: *const KBStore, rule_store: *const RuleStore, scope_kb_id: i32, goal_idx: i32, bindings: *BindingSet) bool {
    if (goal_idx < 0 or goal_idx >= rule_store.term_count) return false;
    const goal = term_mod.resolve(rule_store.terms[@intCast(goal_idx)], bindings);

    var cur_kb = scope_kb_id;
    var walk: i32 = 0;
    while (cur_kb >= 0 and walk < 100) : (walk += 1) {
        const kb = kb_store.getKBConst(cur_kb) orelse break;
        const s: usize = @intCast(kb.facts_offset);
        const e: usize = @intCast(kb.facts_offset + kb.facts_capacity);

        for (kb_store.facts[s..e]) |f| {
            if (f.tag == .empty) continue;
            const ft = query_mod.factToTerm(f);
            const cp = bindings.checkpoint();
            if (unify_mod.unify(goal, ft, bindings, rule_store.terms[0..@intCast(rule_store.term_count)], 50)) {
                return true;
            }
            bindings.undo(cp);
        }
        cur_kb = kb.parent_id;
    }
    return false;
}

// expose factToTerm from query for use in rule
pub const factToTerm = query_mod.factToTerm;
