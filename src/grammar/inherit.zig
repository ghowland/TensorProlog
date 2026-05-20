// ============================================================
// src/grammar/inherit.zig
// ============================================================

const grammar_types = @import("types.zig");
const store_mod = @import("../kb/store.zig");
const vdr = @import("../kb/vdr.zig");

const VlpGrammar = grammar_types.VlpGrammar;
const KBStore = store_mod.KBStore;
const VlpStatus = vdr.VlpStatus;

pub const GrammarStore = struct {
    grammars: []VlpGrammar,
    count: i32,

    pub fn init(backing: []VlpGrammar) GrammarStore {
        for (backing) |*g| g.* = VlpGrammar{};
        return .{ .grammars = backing, .count = 0 };
    }

    pub fn store(self: *GrammarStore, kb_id: i32, grammar: VlpGrammar) ?i32 {
        if (self.count >= @as(i32, @intCast(self.grammars.len))) return null;
        const id = self.count;
        var g = &self.grammars[@intCast(id)];
        g.* = grammar;
        g.id = id;
        // kb_id stored as created_at field repurposed or via separate index
        // simple: use creator_session_id field to hold kb_id for lookup
        g.creator_session_id = kb_id;
        self.count += 1;
        return id;
    }

    pub fn get(self: *const GrammarStore, grammar_id: i32) ?*const VlpGrammar {
        if (grammar_id < 0 or grammar_id >= self.count) return null;
        return &self.grammars[@intCast(grammar_id)];
    }

    pub fn findByKB(self: *const GrammarStore, kb_id: i32) ?*const VlpGrammar {
        var i: i32 = self.count - 1;
        while (i >= 0) : (i -= 1) {
            if (self.grammars[@intCast(i)].creator_session_id == kb_id) {
                return &self.grammars[@intCast(i)];
            }
        }
        return null;
    }
};

pub fn inherit(
    gs: *const GrammarStore,
    kb_store: *const KBStore,
    start_kb_id: i32,
) ?*const VlpGrammar {
    var cur = start_kb_id;
    var depth: i32 = 0;
    while (cur >= 0 and depth < 100) : (depth += 1) {
        const found = gs.findByKB(cur);
        if (found) |g| {
            if (g.validated) return g;
        }
        const kb = kb_store.getKBConst(cur) orelse return null;
        cur = kb.parent_id;
    }
    return null;
}

pub fn inheritById(
    gs: *const GrammarStore,
    kb_store: *const KBStore,
    start_kb_id: i32,
    grammar_id_slot: i32,
) ?*const VlpGrammar {
    var cur = start_kb_id;
    var depth: i32 = 0;
    while (cur >= 0 and depth < 100) : (depth += 1) {
        const kb = kb_store.getKBConst(cur) orelse return null;
        if (kb.grammars_count > 0 and grammar_id_slot < kb.grammars_count) {
            const gid = kb.grammars_offset + grammar_id_slot;
            const g = gs.get(gid);
            if (g) |grammar| {
                if (grammar.validated) return grammar;
            }
        }
        cur = kb.parent_id;
    }
    return null;
}

pub fn compose(
    outer: *const VlpGrammar,
    inner: *const VlpGrammar,
    slot_idx: i16,
    result: *VlpGrammar,
) VlpStatus {
    if (slot_idx < 0 or slot_idx >= outer.slot_count) return .err_grammar_invalid;
    if (!outer.validated or !inner.validated) return .err_grammar_invalid;

    result.* = outer.*;
    result.slots[@intCast(slot_idx)].slot_type = .grammar;
    result.slots[@intCast(slot_idx)].default_kb_id = inner.id;
    return .ok;
}
