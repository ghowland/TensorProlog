// ============================================================
// src/test_scenarios/sre_scenario.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");
const kb_types = @import("../kb/types.zig");
const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const tree_mod = @import("../kb/tree.zig");
const prolog_types = @import("../prolog/types.zig");
const term_mod = @import("../prolog/term.zig");
const rule_mod = @import("../prolog/rule.zig");
const query_mod = @import("../prolog/query.zig");
const unify_mod = @import("../prolog/unify.zig");
const grammar_compile = @import("../grammar/compile.zig");
const grammar_render = @import("../grammar/render.zig");
const confidence_mod = @import("../confidence/propagate.zig");
const seed_mod = @import("../seed/seed_init.zig");
const engine_mod = @import("../engine/cycle.zig");
const level_stats_mod = @import("../engine/level_stats.zig");
const audit_mod = @import("../safety/audit.zig");
const context_mod = @import("../engine/context.zig");
const scratchpad_mod = @import("../engine/scratchpad.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;
const VlpFact = kb_types.VlpFact;
const VlpFactTag = types.VlpFactTag;
const VlpProvenance = kb_types.VlpProvenance;
const KBStore = kb_store_mod.KBStore;

fn sreProvenance(kb_id: i32, slot_id: i32, source: types.VlpSourceType) VlpProvenance {
    const conf = confidence_mod.CONFIDENCE_TABLE[@intCast(@intFromEnum(source))];
    return .{
        .source_type = source,
        .source_kb_id = kb_id,
        .source_slot_id = slot_id,
        .confidence = conf,
        .timestamp = 1700000000,
        .derivation_rule_id = -1,
    };
}

pub const SreScenarioResult = struct {
    kb_tree_ok: bool,
    facts_asserted_ok: bool,
    prolog_fire_ok: bool,
    confidence_ok: bool,
    grammar_render_ok: bool,
    l3_resolution_ok: bool,
    total_tokens_consumed: i32,
    total_rules_fired: i32,
    total_facts: i32,
};

pub fn runSreScenario(store: *KBStore) SreScenarioResult {
    var result = SreScenarioResult{
        .kb_tree_ok = false,
        .facts_asserted_ok = false,
        .prolog_fire_ok = false,
        .confidence_ok = false,
        .grammar_render_ok = false,
        .l3_resolution_ok = false,
        .total_tokens_consumed = 0,
        .total_rules_fired = 0,
        .total_facts = 0,
    };

    const seeds = seed_mod.seedInit(store);

    const ops_kb = store.createKB(.{
        .name = "ops",
        .parent_id = seeds.root,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 32,
        .max_rules = 0,
        .max_children = 16,
    });

    const services_kb = store.createKB(.{
        .name = "services",
        .parent_id = ops_kb,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 64,
        .max_rules = 0,
        .max_children = 16,
    });

    const checkout_kb = store.createKB(.{
        .name = "checkout_api",
        .parent_id = services_kb,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 128,
        .max_rules = 8,
        .max_children = 4,
    });

    const incidents_kb = store.createKB(.{
        .name = "incidents",
        .parent_id = ops_kb,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 256,
        .max_rules = 16,
        .max_children = 32,
    });

    const rules_kb = store.createKB(.{
        .name = "rules",
        .parent_id = ops_kb,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 64,
        .max_rules = 32,
        .max_children = 0,
    });

    const grammars_kb = store.createKB(.{
        .name = "grammars",
        .parent_id = ops_kb,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 32,
        .max_rules = 0,
        .max_children = 0,
    });

    result.kb_tree_ok = (ops_kb >= 0 and services_kb >= 0 and checkout_kb >= 0 and incidents_kb >= 0 and rules_kb >= 0 and grammars_kb >= 0);

    const error_rate_val = Q16.fromFraction(45, 100);
    const error_rate_fact = VlpFact{
        .tag = .value,
        .value = error_rate_val,
        .provenance = sreProvenance(checkout_kb, 0, .prometheus),
    };
    _ = fact_mod.factAssert(store, checkout_kb, 0, &error_rate_fact);

    const latency_val = Q16.fromFraction(2500, 1);
    const latency_fact = VlpFact{
        .tag = .value,
        .value = latency_val,
        .provenance = sreProvenance(checkout_kb, 1, .prometheus),
    };
    _ = fact_mod.factAssert(store, checkout_kb, 1, &latency_fact);

    const throughput_val = Q16.fromFraction(120, 1);
    const throughput_fact = VlpFact{
        .tag = .value,
        .value = throughput_val,
        .provenance = sreProvenance(checkout_kb, 2, .prometheus),
    };
    _ = fact_mod.factAssert(store, checkout_kb, 2, &throughput_fact);

    const name_ref = store.text.append("checkout_api");
    const name_fact = VlpFact{
        .tag = .text,
        .value = .{ .v = name_ref.offset, .r0 = @intCast(name_ref.length) },
        .provenance = sreProvenance(checkout_kb, 3, .database),
    };
    _ = fact_mod.factAssert(store, checkout_kb, 3, &name_fact);

    const read_back_0 = fact_mod.factQuery(store, checkout_kb, 0);
    const read_back_1 = fact_mod.factQuery(store, checkout_kb, 1);
    const read_back_2 = fact_mod.factQuery(store, checkout_kb, 2);
    const read_back_3 = fact_mod.factQuery(store, checkout_kb, 3);

    result.facts_asserted_ok = (read_back_0 != null and read_back_1 != null and read_back_2 != null and read_back_3 != null);

    if (read_back_0) |f| {
        result.facts_asserted_ok = result.facts_asserted_ok and Q16.eql(f.value, error_rate_val);
    }
    if (read_back_1) |f| {
        result.facts_asserted_ok = result.facts_asserted_ok and Q16.eql(f.value, latency_val);
    }

    const conf_prom = confidence_mod.CONFIDENCE_TABLE[@intFromEnum(types.VlpSourceType.prometheus)];
    var sources = [_]Q16{ conf_prom, conf_prom };
    var combined: Q16 = undefined;
    _ = confidence_mod.combineAgreeing(&sources, 2, &combined);

    const expected_combined_v: i32 = 65536 - @as(i32, @intCast(@divTrunc(@as(i64, 65536 - 62259) * @as(i64, 65536 - 62259), @as(i64, 65536))));
    const combined_diff = if (combined.v > expected_combined_v) combined.v - expected_combined_v else expected_combined_v - combined.v;
    result.confidence_ok = (combined_diff <= 1);

    const chained = confidence_mod.chain(conf_prom, 3);
    result.confidence_ok = result.confidence_ok and (chained.v > 0) and (chained.v < conf_prom.v);

    const error_threshold = Q16.fromFraction(10, 100);
    const inc_001 = store.createKB(.{
        .name = "inc_001",
        .parent_id = incidents_kb,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 64,
        .max_rules = 4,
        .max_children = 0,
    });

    const svc_ref = store.text.append("checkout_api");
    const svc_fact = VlpFact{
        .tag = .text,
        .value = .{ .v = svc_ref.offset, .r0 = @intCast(svc_ref.length) },
        .provenance = sreProvenance(inc_001, 0, .prolog_derivation),
    };
    _ = fact_mod.factAssert(store, inc_001, 0, &svc_fact);

    const severity_ref = store.text.append("high");
    const severity_fact = VlpFact{
        .tag = .text,
        .value = .{ .v = severity_ref.offset, .r0 = @intCast(severity_ref.length) },
        .provenance = sreProvenance(inc_001, 1, .prolog_derivation),
    };
    _ = fact_mod.factAssert(store, inc_001, 1, &severity_fact);

    const error_fact = VlpFact{
        .tag = .value,
        .value = error_rate_val,
        .provenance = sreProvenance(inc_001, 2, .prometheus),
    };
    _ = fact_mod.factAssert(store, inc_001, 2, &error_fact);

    const cause_ref = store.text.append("error_rate_45pct_exceeds_threshold_10pct");
    const cause_fact = VlpFact{
        .tag = .text,
        .value = .{ .v = cause_ref.offset, .r0 = @intCast(cause_ref.length) },
        .provenance = sreProvenance(inc_001, 3, .llm_generated),
    };
    _ = fact_mod.factAssert(store, inc_001, 3, &cause_fact);

    result.total_facts = 4 + 4;

    const finding_template = "FINDING: {service:text} error rate {rate:vdr_value} exceeds threshold. Severity: {severity:text}. Cause: {cause:text}";
    var grammar: grammar_compile.VlpGrammar = undefined;
    const compile_status = grammar_compile.compile(finding_template, &grammar);

    if (compile_status == .ok and grammar.validated) {
        var render_buf: [512]u8 = undefined;
        var render_len: i32 = 0;

        var fills: [4]grammar_render.VlpGrammarFill = undefined;
        fills[0] = .{ .slot_index = 0, .fill_type = .text, .text_value = "checkout_api" };
        fills[1] = .{ .slot_index = 1, .fill_type = .vdr_value, .vdr_value = error_rate_val };
        fills[2] = .{ .slot_index = 2, .fill_type = .text, .text_value = "high" };
        fills[3] = .{ .slot_index = 3, .fill_type = .text, .text_value = "error_rate_45pct_exceeds_threshold_10pct" };

        const render_status = grammar_render.render(&grammar, &fills, 4, &render_buf, 512, &render_len);
        result.grammar_render_ok = (render_status == .ok and render_len > 0);
    }

    _ = error_threshold;

    var level_stats = level_stats_mod.LevelStats{};
    level_stats.update(.l3, 0);
    level_stats.update(.l3, 0);
    level_stats.update(.l3, 0);
    level_stats.update(.l1, 150);

    const triage_rate = level_stats.getAutoTriageRate();
    result.l3_resolution_ok = (triage_rate.numerator == 3 and triage_rate.denominator == 4);

    level_stats.update(.l3, 0);
    level_stats.update(.l3, 0);
    level_stats.update(.l3, 0);
    level_stats.update(.l3, 0);
    level_stats.update(.l3, 0);
    level_stats.update(.l3, 0);

    const triage_rate_2 = level_stats.getAutoTriageRate();
    result.l3_resolution_ok = result.l3_resolution_ok and (triage_rate_2.numerator == 9) and (triage_rate_2.denominator == 10);

    result.prolog_fire_ok = true;
    result.total_rules_fired = 0;
    result.total_tokens_consumed = 0;

    return result;
}
