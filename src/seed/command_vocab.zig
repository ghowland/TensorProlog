// ============================================================
// src/seed/command_vocab.zig
// ============================================================

pub fn loadCommandVocab(store: *KBStore, vocab_kb: i32) void {
    const commands = [_][]const u8{
        "KB_ASSERT",
        "KB_QUERY",
        "KB_RETRACT",
        "PROLOG_QUERY",
        "PROLOG_ASSERT_RULE",
        "BUILTIN_CALL",
        "GRAMMAR_RENDER",
        "DIRECT_OUTPUT",
        "OP_FILESYSTEM",
        "OP_COMPILE",
        "OP_EXECUTE",
        "OP_NETWORK",
        "OP_PROCESS",
        "SESSION_SNAPSHOT",
        "SESSION_CLONE",
    };

    for (commands, 0..) |cmd, i| {
        assertTextFact(store, vocab_kb, @intCast(i), cmd);
    }

    var slot: i32 = @intCast(commands.len);

    const kb_ops = [_][]const u8{
        "assert", "query",   "retract",  "search", "scoped_search",
        "create", "destroy", "reset",    "freeze", "unfreeze",
        "mount",  "unmount", "children", "info",   "resolve_path",
    };
    for (kb_ops) |op| {
        assertTextFact(store, vocab_kb, slot, op);
        slot += 1;
    }

    const prolog_ops = [_][]const u8{
        "rule_assert", "rule_retract", "fire_all",     "fire_commit",
        "unify",       "rule_stats",   "hygiene_scan",
    };
    for (prolog_ops) |op| {
        assertTextFact(store, vocab_kb, slot, op);
        slot += 1;
    }

    const grammar_ops = [_][]const u8{
        "compile",    "render",  "render_kb", "validate", "inherit",
        "list_slots", "compose", "store",
    };
    for (grammar_ops) |op| {
        assertTextFact(store, vocab_kb, slot, op);
        slot += 1;
    }

    const prim_ops = [_][]const u8{
        "lru_get",          "lru_put",     "lru_evict",     "lru_size",      "lru_clear",
        "counter_get",      "counter_inc", "counter_reset", "counter_bound", "lock_acquire",
        "lock_release",     "lock_query",  "queue_push",    "queue_pop",     "queue_peek",
        "queue_size",       "queue_clear", "stack_push",    "stack_pop",     "stack_peek",
        "stack_size",       "stack_clear", "ring_write",    "ring_read",     "ring_size",
        "ring_clear",       "bitset_set",  "bitset_clear",  "bitset_get",    "bitset_pop",
        "bitset_clear_all",
    };
    for (prim_ops) |op| {
        assertTextFact(store, vocab_kb, slot, op);
        slot += 1;
    }

    const builtin_names = [_][]const u8{
        "arith_add",           "arith_sub",        "arith_mul",          "arith_div",
        "arith_pow",           "arith_reciprocal", "arith_compare",      "arith_equal",
        "arith_min",           "arith_max",        "arith_sign",         "arith_is_zero",
        "arith_floor",         "arith_ceil",       "arith_round",        "arith_abs",
        "arith_negate",        "arith_clamp",      "arith_lerp",         "arith_midpoint",
        "arith_distance",      "arith_from_int",   "arith_to_int",       "text_reverse",
        "text_split",          "text_contains",    "text_replace",       "text_join",
        "text_trim",           "text_upper",       "text_lower",         "text_starts_with",
        "text_ends_with",      "text_index_of",    "text_substring",     "text_repeat",
        "text_pad_left",       "text_pad_right",   "text_char_at",       "text_length",
        "sort",                "sort_by",          "filter",             "map",
        "reduce",              "group_by",         "frequencies",        "distinct",
        "flatten",             "chunk",            "zip",                "unzip",
        "reverse",             "rotate",           "take_first",         "take_last",
        "drop_first",          "drop_last",        "partition",          "interleave",
        "enumerate",           "min_by",           "max_by",             "scan",
        "all",                 "any",              "none",               "count",
        "find_first",          "find_last",        "find_all",           "binary_search",
        "merge",               "deduplicate",      "window",             "cartesian",
        "set_union",           "set_intersection", "set_difference",     "set_symmetric_diff",
        "set_is_subset",       "set_is_superset",  "set_is_disjoint",    "set_contains",
        "set_add",             "set_remove",       "set_equal",          "set_power_set",
        "set_from_array",      "map_get",          "map_set",            "map_delete",
        "map_contains_key",    "map_keys",         "map_values",         "map_size",
        "map_merge",           "map_filter_keys",  "map_filter_values",  "map_map_values",
        "map_invert",          "map_clear",        "map_equal",          "map_from_arrays",
        "parse_json",          "parse_csv",        "parse_xml",          "parse_yaml",
        "to_json",             "to_csv",           "to_fraction",        "from_fraction",
        "vdr_to_decimal",      "decimal_to_vdr",   "base_convert",       "timestamp_fields",
        "mat_vec_mul",         "transpose",        "gaussian_elim",      "inverse",
        "determinant",         "gram_schmidt",     "eigenvalues",        "svd",
        "stats_mean",          "stats_variance",   "stats_median",       "stats_bayes",
        "stats_normalize",     "stats_histogram",  "stats_correlation",  "stats_covariance",
        "graph_create",        "graph_add_node",   "graph_add_edge",     "graph_remove_node",
        "graph_remove_edge",   "graph_bfs",        "graph_dfs",          "graph_shortest_path",
        "graph_topo_sort",     "graph_components", "graph_cycle_detect", "graph_pagerank",
        "graph_markov_steady", "int_add",          "int_sub",            "int_mul",
        "int_div",             "int_mod",          "int_abs",            "int_sign",
        "int_min",             "int_max",          "int_clamp",          "int_pow",
        "int_factorial",       "int_choose",       "bit_and",            "bit_or",
        "bit_xor",             "bit_not",          "bit_shl",            "bit_shr",
        "bit_popcount",        "bit_reverse",      "timestamp_now",      "timestamp_diff",
        "timestamp_add",       "duration_seconds", "duration_minutes",   "duration_hours",
        "duration_days",       "duration_compare", "duration_format",    "timestamp_fields_ext",
    };
    for (builtin_names) |name| {
        assertTextFact(store, vocab_kb, slot, name);
        slot += 1;
    }

    const output_ops = [_][]const u8{
        "kb://",
        "FINDING",
        "SUMMARY",
        "TABLE",
        "LIST",
        "STATUS",
        "ERROR",
        "END_TURN",
    };
    for (output_ops) |op| {
        assertTextFact(store, vocab_kb, slot, op);
        slot += 1;
    }

    const scope_ops = [_][]const u8{
        "root",
        "root.system",
        "root.ops",
        "root.public",
        "parent",
        "children",
        "scope",
    };
    for (scope_ops) |op| {
        assertTextFact(store, vocab_kb, slot, op);
        slot += 1;
    }
}
