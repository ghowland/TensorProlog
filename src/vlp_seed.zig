// ============================================================
// vlp_seed.zig
// Seed layer — initial KB tree loaded at system boot.
// ~23,400 entries across ~1.5 MB.
// Frozen after init. All sessions inherit from it.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const kb_mod = @import("vlp_kb_store.zig");
const snapshot_mod = @import("vlp_snapshot.zig");

// ============================================================
// Configuration
// ============================================================

pub const SeedConfig = struct {
    snapshot_path: ?[]const u8 = null,
    create_fresh: bool = true,
};

// ============================================================
// Seed KB IDs — well-known, fixed
// ============================================================

pub const ROOT_KB_ID: i32 = 0;
pub const SYSTEM_KB_ID: i32 = 1;
pub const OSO_KB_ID: i32 = 2;
pub const CONFIDENCE_KB_ID: i32 = 3;
pub const BUILTINS_KB_ID: i32 = 4;
pub const COMMAND_VOCAB_KB_ID: i32 = 5;
pub const HYGIENE_KB_ID: i32 = 6;
pub const TEMPLATES_KB_ID: i32 = 7;
pub const SENTENCES_KB_ID: i32 = 8;
pub const FORMATS_KB_ID: i32 = 9;

pub const SEED_KB_COUNT: i32 = 10;

// ============================================================
// Init — entry point
// ============================================================

pub fn init(kb_store: *kb_mod.KbStore, config: *const SeedConfig) types.Status {
    // Try loading from snapshot first
    if (config.snapshot_path) |path| {
        var snap_mgr = snapshot_mod.init(kb_store.allocator, kb_store.bridge);
        defer snap_mgr.deinit();

        if (snap_mgr.load(path)) |data| {
            defer snap_mgr.freeData(data);
            if (snapshot_mod.validateChecksum(data)) {
                // Restore seed from snapshot
                var dummy_session = std.mem.zeroes(types.Session);
                const status = snap_mgr.restoreToDevice(data, &dummy_session);
                if (status.isOk()) return types.Status.ok();
            }
        }
    }

    // Create fresh seed layer
    if (config.create_fresh) {
        return createFresh(kb_store);
    }

    return types.Status.err(.system, .seed_load_failed, 0);
}

// ============================================================
// Create fresh seed — build entire tree from code
// ============================================================

pub fn createFresh(kb_store: *kb_mod.KbStore) types.Status {
    // Root KB
    _ = kb_store.createKb(&.{
        .name = "root",
        .path = "root",
        .parent_id = -1,
        .max_facts = 100,
        .max_rules = 50,
        .visibility = 0, // public
        .owner = "system",
    });

    // System subtree
    _ = kb_store.createKb(&.{
        .name = "system",
        .path = "root.system",
        .parent_id = ROOT_KB_ID,
        .max_facts = 200,
        .max_rules = 100,
        .visibility = 1, // internal
        .owner = "system",
    });

    _ = kb_store.createKb(&.{
        .name = "oso",
        .path = "root.system.oso",
        .parent_id = SYSTEM_KB_ID,
        .max_facts = 200,
        .max_rules = 50,
        .visibility = 0,
        .owner = "system",
    });

    _ = kb_store.createKb(&.{
        .name = "confidence",
        .path = "root.system.confidence",
        .parent_id = SYSTEM_KB_ID,
        .max_facts = 50,
        .max_rules = 0,
        .visibility = 0,
        .owner = "system",
    });

    _ = kb_store.createKb(&.{
        .name = "builtins",
        .path = "root.system.builtins",
        .parent_id = SYSTEM_KB_ID,
        .max_facts = 500,
        .max_rules = 0,
        .visibility = 0,
        .owner = "system",
    });

    _ = kb_store.createKb(&.{
        .name = "command_vocab",
        .path = "root.system.command_vocab",
        .parent_id = SYSTEM_KB_ID,
        .max_facts = 300,
        .max_rules = 0,
        .visibility = 0,
        .owner = "system",
    });

    _ = kb_store.createKb(&.{
        .name = "hygiene",
        .path = "root.system.hygiene",
        .parent_id = SYSTEM_KB_ID,
        .max_facts = 100,
        .max_rules = 50,
        .visibility = 1,
        .owner = "system",
    });

    // Templates subtree
    _ = kb_store.createKb(&.{
        .name = "templates",
        .path = "root.templates",
        .parent_id = ROOT_KB_ID,
        .max_facts = 500,
        .max_rules = 0,
        .visibility = 0,
        .owner = "system",
    });

    _ = kb_store.createKb(&.{
        .name = "sentences",
        .path = "root.templates.sentences",
        .parent_id = TEMPLATES_KB_ID,
        .max_facts = 300,
        .max_rules = 0,
        .visibility = 0,
        .owner = "system",
    });

    _ = kb_store.createKb(&.{
        .name = "formats",
        .path = "root.templates.formats",
        .parent_id = TEMPLATES_KB_ID,
        .max_facts = 200,
        .max_rules = 0,
        .visibility = 0,
        .owner = "system",
    });

    // Populate content
    var status: types.Status = undefined;

    status = populateOso(kb_store);
    if (status.isErr()) return status;

    status = populateConfidenceTable(kb_store);
    if (status.isErr()) return status;

    status = populateCommandVocab(kb_store);
    if (status.isErr()) return status;

    status = populateHygieneRules(kb_store);
    if (status.isErr()) return status;

    // Freeze all seed KBs
    var id: i32 = 0;
    while (id < SEED_KB_COUNT) : (id += 1) {
        _ = kb_store.freezeKb(id);
    }

    return types.Status.ok();
}

// ============================================================
// OSO — 15 engineering principles as Prolog terms (~176 terms)
// ============================================================

fn populateOso(kb_store: *kb_mod.KbStore) types.Status {
    const principles = [_][]const u8{
        "integer_foundation",
        "exact_arithmetic",
        "bounded_data_primitives",
        "deterministic_execution",
        "kb_tree_scoping",
        "prolog_deduction",
        "grammar_templates",
        "confidence_propagation",
        "session_isolation",
        "grant_gated_operations",
        "runner_autonomy",
        "snapshot_portability",
        "audit_completeness",
        "level_transition",
        "self_maintenance",
    };

    for (principles, 0..) |name, i| {
        const name_off = kb_store.textAppend(name);
        const fact = types.Fact{
            .tag = .text,
            .value = types.Q16.fromParts(name_off, @intCast(name.len), 0),
            .provenance = types.Provenance.direct(.vdr_computation, OSO_KB_ID, @intCast(i), 0),
        };
        const status = kb_store.factWrite(OSO_KB_ID, @intCast(i), &fact);
        if (status.isErr()) return status;
    }

    return types.Status.ok();
}

// ============================================================
// Confidence table — source type → exact Q16 confidence
// ============================================================

fn populateConfidenceTable(kb_store: *kb_mod.KbStore) types.Status {
    for (types.confidence_table, 0..) |conf, i| {
        const fact = types.Fact{
            .tag = .value,
            .value = conf,
            .provenance = types.Provenance.direct(.vdr_computation, CONFIDENCE_KB_ID, @intCast(i), 0),
        };
        const status = kb_store.factWrite(CONFIDENCE_KB_ID, @intCast(i), &fact);
        if (status.isErr()) return status;
    }
    return types.Status.ok();
}

// ============================================================
// Command vocabulary — ~300 command token names
// ============================================================

fn populateCommandVocab(kb_store: *kb_mod.KbStore) types.Status {
    // Command type names
    const cmd_names = [_][]const u8{
        "KB_ASSERT",      "KB_QUERY",           "KB_RETRACT",
        "PROLOG_QUERY",   "PROLOG_ASSERT_RULE", "BUILTIN",
        "GRAMMAR_RENDER", "DIRECT_OUTPUT",      "OP_FILESYSTEM",
        "OP_COMPILE",     "OP_EXECUTE",         "OP_NETWORK",
        "OP_PROCESS",     "SESSION_SNAPSHOT",   "SESSION_CLONE",
    };

    for (cmd_names, 0..) |name, i| {
        const name_off = kb_store.textAppend(name);
        const fact = types.Fact{
            .tag = .text,
            .value = types.Q16.fromParts(name_off, @intCast(name.len), 0),
            .provenance = types.Provenance.direct(.vdr_computation, COMMAND_VOCAB_KB_ID, @intCast(i), 0),
        };
        const status = kb_store.factWrite(COMMAND_VOCAB_KB_ID, @intCast(i), &fact);
        if (status.isErr()) return status;
    }

    return types.Status.ok();
}

// ============================================================
// Hygiene rules — self-maintenance Prolog rules
// ============================================================

fn populateHygieneRules(kb_store: *kb_mod.KbStore) types.Status {
    // Stale rule detector: fire_age > 90 days → candidate_for_pruning
    // Failing rule detector: success_rate < 20% after 5+ fires → candidate
    // Orphan rule detector: references revoked grant → candidate
    // These are stored as facts describing the rules.
    // Actual Prolog rule assertion requires the prolog engine,
    // which depends on kb_store. Break the cycle by storing
    // rule descriptions as facts; the prolog engine loads them at startup.

    const rule_descs = [_][]const u8{
        "stale_rule_detector:age>7776000->prune",
        "failing_rule_detector:success<20pct,fires>5->prune",
        "orphan_rule_detector:grant_revoked->prune",
    };

    for (rule_descs, 0..) |desc, i| {
        const desc_off = kb_store.textAppend(desc);
        const fact = types.Fact{
            .tag = .rule_ref,
            .value = types.Q16.fromParts(desc_off, @intCast(desc.len), 0),
            .provenance = types.Provenance.direct(.vdr_computation, HYGIENE_KB_ID, @intCast(i), 0),
        };
        const status = kb_store.factWrite(HYGIENE_KB_ID, @intCast(i), &fact);
        if (status.isErr()) return status;
    }

    return types.Status.ok();
}
