// ============================================================
// src/seed/seed_init.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");
const kb_types = @import("../kb/types.zig");
const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const tree_mod = @import("../kb/tree.zig");
const prolog_types = @import("../prolog/types.zig");
const rule_mod = @import("../prolog/rule.zig");
const confidence_mod = @import("../confidence/propagate.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;
const VlpFact = kb_types.VlpFact;
const VlpFactTag = types.VlpFactTag;
const VlpProvenance = kb_types.VlpProvenance;
const KBStore = kb_store_mod.KBStore;
const VlpSourceType = types.VlpSourceType;

fn seedProvenance(kb_id: i32, slot_id: i32) VlpProvenance {
    return .{
        .source_type = .vdr_computation,
        .source_kb_id = kb_id,
        .source_slot_id = slot_id,
        .confidence = .{ .v = Q16.D, .r0 = 0 },
        .timestamp = 0,
        .derivation_rule_id = -1,
    };
}

fn assertTextFact(store: *KBStore, kb_id: i32, slot: i32, text: []const u8) void {
    const ref = store.text.append(text);
    const fact = VlpFact{
        .tag = .text,
        .value = .{ .v = ref.offset, .r0 = @intCast(ref.length) },
        .provenance = seedProvenance(kb_id, slot),
    };
    _ = fact_mod.factAssert(store, kb_id, slot, &fact);
}

fn assertValueFact(store: *KBStore, kb_id: i32, slot: i32, val: Q16) void {
    const fact = VlpFact{
        .tag = .value,
        .value = val,
        .provenance = seedProvenance(kb_id, slot),
    };
    _ = fact_mod.factAssert(store, kb_id, slot, &fact);
}

fn assertIntFact(store: *KBStore, kb_id: i32, slot: i32, val: i32) void {
    const fact = VlpFact{
        .tag = .counter,
        .value = .{ .v = val, .r0 = 0 },
        .provenance = seedProvenance(kb_id, slot),
    };
    _ = fact_mod.factAssert(store, kb_id, slot, &fact);
}

pub const SeedIds = struct {
    root: i32,
    system: i32,
    oso: i32,
    confidence: i32,
    builtins: i32,
    command_vocab: i32,
    hygiene: i32,
    templates: i32,
    sentences: i32,
    formats: i32,
};

pub fn seedInit(store: *KBStore) SeedIds {
    const root = store.createKB(.{
        .name = "root",
        .parent_id = -1,
        .visibility = .public,
        .owner = "system",
        .max_facts = 16,
        .max_rules = 0,
        .max_children = 32,
    });

    const system = store.createKB(.{
        .name = "system",
        .parent_id = root,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 16,
        .max_rules = 0,
        .max_children = 16,
    });

    const oso = store.createKB(.{
        .name = "oso",
        .parent_id = system,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 256,
        .max_rules = 32,
        .max_children = 0,
    });

    const confidence_kb = store.createKB(.{
        .name = "confidence",
        .parent_id = system,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 16,
        .max_rules = 0,
        .max_children = 0,
    });

    const builtins_kb = store.createKB(.{
        .name = "builtins",
        .parent_id = system,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 512,
        .max_rules = 0,
        .max_children = 0,
    });

    const command_vocab = store.createKB(.{
        .name = "command_vocab",
        .parent_id = system,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 512,
        .max_rules = 0,
        .max_children = 0,
    });

    const hygiene = store.createKB(.{
        .name = "hygiene",
        .parent_id = system,
        .visibility = .internal,
        .owner = "system",
        .max_facts = 32,
        .max_rules = 8,
        .max_children = 0,
    });

    const templates = store.createKB(.{
        .name = "templates",
        .parent_id = root,
        .visibility = .public,
        .owner = "system",
        .max_facts = 16,
        .max_rules = 0,
        .max_children = 8,
    });

    const sentences = store.createKB(.{
        .name = "sentences",
        .parent_id = templates,
        .visibility = .public,
        .owner = "system",
        .max_facts = 128,
        .max_rules = 0,
        .max_children = 0,
    });

    const formats = store.createKB(.{
        .name = "formats",
        .parent_id = templates,
        .visibility = .public,
        .owner = "system",
        .max_facts = 128,
        .max_rules = 0,
        .max_children = 0,
    });

    loadOsoRules(store, oso);
    loadConfidenceTable(store, confidence_kb);
    loadCommandVocab(store, command_vocab);
    loadHygieneRules(store, hygiene);
    loadSentenceTemplates(store, sentences);
    loadFormatGrammars(store, formats);
    loadBuiltinDeclarations(store, builtins_kb);

    return .{
        .root = root,
        .system = system,
        .oso = oso,
        .confidence = confidence_kb,
        .builtins = builtins_kb,
        .command_vocab = command_vocab,
        .hygiene = hygiene,
        .templates = templates,
        .sentences = sentences,
        .formats = formats,
    };
}
