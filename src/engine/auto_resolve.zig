// ============================================================
// src/engine/auto_resolve.zig
// ============================================================

const q16_mod = @import("../vdr/q16.zig");
const prolog_types = @import("../prolog/types.zig");
const grammar_types = @import("../grammar/types.zig");
const grammar_inherit = @import("../grammar/inherit.zig");
const grammar_render = @import("../grammar/render.zig");
const store_mod = @import("../kb/store.zig");

const Q16 = q16_mod.Q16;
const PrologFired = prolog_types.PrologFired;
const PrologAction = prolog_types.PrologAction;
const VlpGrammar = grammar_types.VlpGrammar;
const GrammarFill = grammar_types.GrammarFill;
const GrammarKBMapping = grammar_types.GrammarKBMapping;
const RenderResult = grammar_types.RenderResult;
const GrammarStore = grammar_inherit.GrammarStore;
const KBStore = store_mod.KBStore;

pub const AutoResolution = struct {
    fully_handled: bool = false,
    grammar: ?*const VlpGrammar = null,
    mappings: [16]GrammarKBMapping = undefined,
    n_mappings: i32 = 0,
    n_actions: i32 = 0,
    output_len: i32 = 0,
};

pub fn check(
    fired: []const PrologFired,
    n_fired: i32,
    confidence_threshold: Q16,
    grammar_store: *const GrammarStore,
    kb_store: *const KBStore,
    output: []u8,
) AutoResolution {
    var result = AutoResolution{};

    var fi: i32 = 0;
    while (fi < n_fired) : (fi += 1) {
        const entry = &fired[@intCast(fi)];

        if (Q16.compare(entry.confidence, confidence_threshold) < 0) continue;

        var has_grammar: bool = false;
        var has_output: bool = false;
        var grammar_kb_id: i32 = -1;

        var ai: i32 = 0;
        while (ai < entry.action_count) : (ai += 1) {
            const act = &entry.actions[@intCast(ai)];
            switch (act.atype) {
                .direct_output => {
                    has_output = true;
                },
                .assert_fact => {
                    if (act.fact.tag == .grammar_ref) {
                        has_grammar = true;
                        grammar_kb_id = act.target_kb_id;
                    }
                },
                else => {},
            }
        }

        if (!has_output) continue;

        var grammar: ?*const VlpGrammar = null;
        if (has_grammar and grammar_kb_id >= 0) {
            grammar = grammar_inherit.inherit(grammar_store, kb_store, grammar_kb_id);
        }
        if (grammar == null) {
            grammar = grammar_inherit.inherit(grammar_store, kb_store, entry.actions[0].target_kb_id);
        }

        result.fully_handled = true;
        result.grammar = grammar;
        result.n_actions = entry.action_count;

        var mi: i32 = 0;
        ai = 0;
        while (ai < entry.action_count) : (ai += 1) {
            const act = &entry.actions[@intCast(ai)];
            if (act.atype == .assert_fact and mi < 16) {
                result.mappings[@intCast(mi)] = .{
                    .slot_index = @intCast(mi),
                    .kb_id = act.target_kb_id,
                    .slot_id = act.target_slot_id,
                };
                mi += 1;
            }
        }
        result.n_mappings = mi;

        if (result.grammar) |g| {
            const rr = grammar_render.renderFromKB(g, &kb_store.text, kb_store, result.mappings[0..@intCast(mi)], output);
            result.output_len = rr.len;
        }

        break;
    }

    return result;
}

pub fn isFullyResolved(resolution: *const AutoResolution) bool {
    return resolution.fully_handled and resolution.output_len > 0;
}
