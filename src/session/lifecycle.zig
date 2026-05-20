// ============================================================
// src/session/lifecycle.zig
// ============================================================

const session_types = @import("types.zig");
const cow_mod = @import("cow.zig");
const store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const tree_mod = @import("../kb/tree.zig");

const VlpSession = session_types.VlpSession;
const VlpStatus = session_types.VlpStatus;
const VlpSessionState = session_types.VlpSessionState;
const VlpMergePolicy = session_types.VlpMergePolicy;
const SessionConfig = session_types.SessionConfig;
const CloneConfig = session_types.CloneConfig;
const MergeResult = session_types.MergeResult;
const MergeConflict = session_types.MergeConflict;
const KBStore = store_mod.KBStore;
const COWPageTable = cow_mod.COWPageTable;

pub const SessionStore = struct {
    sessions: []VlpSession,
    count: i32,

    pub fn init(backing: []VlpSession) SessionStore {
        for (backing) |*s| s.* = VlpSession{};
        return .{ .sessions = backing, .count = 0 };
    }

    pub fn create(self: *SessionStore, cfg: SessionConfig) ?i32 {
        if (self.count >= @as(i32, @intCast(self.sessions.len))) return null;
        const id = self.count;
        self.sessions[@intCast(id)] = .{
            .id = id,
            .user_id = cfg.user_id,
            .kb_root_id = cfg.kb_root_id,
            .visibility_level = cfg.visibility_level,
            .state = .active,
            .max_kb_count = cfg.max_kb_count,
            .max_live_bytes = cfg.max_live_bytes,
            .max_turns = cfg.max_turns,
            .alive = true,
        };
        self.count += 1;
        return id;
    }

    pub fn get(self: *SessionStore, id: i32) ?*VlpSession {
        if (id < 0 or id >= self.count) return null;
        const s = &self.sessions[@intCast(id)];
        if (!s.alive) return null;
        return s;
    }

    pub fn getConst(self: *const SessionStore, id: i32) ?*const VlpSession {
        if (id < 0 or id >= self.count) return null;
        const s = &self.sessions[@intCast(id)];
        if (!s.alive) return null;
        return s;
    }

    pub fn destroy(self: *SessionStore, id: i32) VlpStatus {
        const s = self.get(id) orelse return .err_kb_not_found;
        s.state = .killed;
        s.alive = false;
        return .ok;
    }

    pub fn kill(self: *SessionStore, id: i32) VlpStatus {
        const s = self.get(id) orelse return .err_kb_not_found;
        s.state = .killed;
        s.alive = false;
        return .ok;
    }

    pub fn clone(self: *SessionStore, parent_id: i32, cfg: CloneConfig) ?i32 {
        const parent = self.getConst(parent_id) orelse return null;
        if (self.count >= @as(i32, @intCast(self.sessions.len))) return null;

        const child_id = self.count;
        var child = &self.sessions[@intCast(child_id)];
        child.* = parent.*;
        child.id = child_id;
        child.parent_session_id = parent_id;
        child.clone_generation = parent.clone_generation + 1;
        child.state = .active;
        child.alive = true;

        if (cfg.max_turns > 0) child.max_turns = cfg.max_turns;

        if (cfg.fresh_live) {
            child.current_turn = 0;
            child.facts_asserted = 0;
            child.facts_retracted = 0;
            child.rules_fired = 0;
            child.prolog_queries = 0;
            child.primitive_calls = 0;
            child.grammar_renders = 0;
            child.llm_tokens = 0;
            child.command_tokens = 0;
            child.l1_count = 0;
            child.l2_count = 0;
            child.l3_count = 0;
        }

        self.count += 1;
        return child_id;
    }

    pub fn merge(self: *SessionStore, parent_id: i32, child_id: i32, policy: VlpMergePolicy, kb_store: *KBStore) MergeResult {
        var result = MergeResult{};

        const parent = self.get(parent_id) orelse {
            result.status = .err_kb_not_found;
            return result;
        };
        const child = self.getConst(child_id) orelse {
            result.status = .err_kb_not_found;
            return result;
        };

        if (child.parent_session_id != parent_id) {
            result.status = .err_kb_access_denied;
            return result;
        }

        var subtree: [256]i32 = undefined;
        const n_kbs = tree_mod.collectSubtree(kb_store, child.kb_root_id, &subtree);

        var ki: i32 = 0;
        while (ki < n_kbs) : (ki += 1) {
            const kid = subtree[@intCast(ki)];
            const kb = kb_store.getKBConst(kid) orelse continue;
            const s: usize = @intCast(kb.facts_offset);
            const e: usize = @intCast(kb.facts_offset + kb.facts_capacity);

            var si: i32 = 0;
            for (kb_store.facts[s..e]) |child_fact| {
                defer si += 1;
                if (child_fact.tag == .empty) continue;

                const parent_fact = fact_mod.query(kb_store, kid, si);
                if (parent_fact) |pf| {
                    if (pf.provenance.timestamp != child_fact.provenance.timestamp) {
                        switch (policy) {
                            .ours => continue,
                            .theirs => {
                                _ = fact_mod.assert(kb_store, kid, si, &child_fact);
                                result.merged_count += 1;
                            },
                            .fail_on_conflict => {
                                if (result.conflict_count < 64) {
                                    result.conflicts[@intCast(result.conflict_count)] = .{
                                        .kb_id = kid,
                                        .slot_id = si,
                                    };
                                    result.conflict_count += 1;
                                }
                            },
                        }
                    }
                } else {
                    _ = fact_mod.assert(kb_store, kid, si, &child_fact);
                    result.merged_count += 1;
                }
            }
        }

        if (policy == .fail_on_conflict and result.conflict_count > 0) {
            result.status = .err_kb_frozen;
        }

        _ = parent;
        return result;
    }

    pub fn aliveCount(self: *const SessionStore) i32 {
        var c: i32 = 0;
        var i: i32 = 0;
        while (i < self.count) : (i += 1) {
            if (self.sessions[@intCast(i)].alive) c += 1;
        }
        return c;
    }

    pub fn shouldRecycle(self: *const SessionStore, id: i32) bool {
        const s = self.getConst(id) orelse return false;
        if (s.max_turns <= 0) return false;
        return s.current_turn >= s.max_turns;
    }

    pub fn incrementTurn(self: *SessionStore, id: i32) void {
        const s = self.get(id) orelse return;
        s.current_turn += 1;
    }

    pub fn updateLevel(self: *SessionStore, id: i32, level: session_types.VlpExecutionLevel, tokens: i32) void {
        const s = self.get(id) orelse return;
        switch (level) {
            .l1 => {
                s.l1_count += 1;
                s.llm_tokens += @as(i64, tokens);
            },
            .l2 => {
                s.l2_count += 1;
                s.llm_tokens += @as(i64, tokens);
            },
            .l3 => s.l3_count += 1,
        }
    }

    pub fn autoTriageRate(self: *const SessionStore, id: i32) struct { num: i64, den: i64 } {
        const s = self.getConst(id) orelse return .{ .num = 0, .den = 0 };
        const total = s.l1_count + s.l2_count + s.l3_count;
        if (total == 0) return .{ .num = 0, .den = 0 };
        return .{ .num = s.l3_count, .den = total };
    }
};
