// ============================================================
// src/seed/builtin_declarations.zig
// ============================================================

pub fn loadBuiltinDeclarations(store: *KBStore, builtins_kb: i32) void {
    const decls = [_]struct { id: i32, name: []const u8, pure: bool, inputs: i32 }{
        .{ .id = 0, .name = "arith_add", .pure = true, .inputs = 2 },
        .{ .id = 1, .name = "arith_sub", .pure = true, .inputs = 2 },
        .{ .id = 2, .name = "arith_mul", .pure = true, .inputs = 2 },
        .{ .id = 3, .name = "arith_div", .pure = true, .inputs = 2 },
        .{ .id = 4, .name = "arith_pow", .pure = true, .inputs = 2 },
        .{ .id = 5, .name = "arith_reciprocal", .pure = true, .inputs = 1 },
        .{ .id = 6, .name = "arith_compare", .pure = true, .inputs = 2 },
        .{ .id = 7, .name = "arith_equal", .pure = true, .inputs = 2 },
        .{ .id = 100, .name = "text_reverse", .pure = true, .inputs = 1 },
        .{ .id = 101, .name = "text_split", .pure = true, .inputs = 1 },
        .{ .id = 102, .name = "text_contains", .pure = true, .inputs = 1 },
        .{ .id = 200, .name = "map_get", .pure = true, .inputs = 1 },
        .{ .id = 201, .name = "map_set", .pure = true, .inputs = 2 },
        .{ .id = 300, .name = "parse_json", .pure = false, .inputs = 1 },
        .{ .id = 301, .name = "parse_csv", .pure = false, .inputs = 1 },
        .{ .id = 302, .name = "parse_xml", .pure = false, .inputs = 1 },
        .{ .id = 303, .name = "parse_yaml", .pure = false, .inputs = 1 },
        .{ .id = 400, .name = "mat_vec_mul", .pure = true, .inputs = 0 },
        .{ .id = 401, .name = "transpose", .pure = true, .inputs = 0 },
        .{ .id = 402, .name = "gaussian_elim", .pure = true, .inputs = 0 },
        .{ .id = 403, .name = "inverse", .pure = true, .inputs = 0 },
        .{ .id = 404, .name = "determinant", .pure = true, .inputs = 0 },
        .{ .id = 420, .name = "stats_mean", .pure = true, .inputs = 1 },
        .{ .id = 421, .name = "stats_variance", .pure = true, .inputs = 1 },
        .{ .id = 422, .name = "stats_median", .pure = true, .inputs = 1 },
        .{ .id = 423, .name = "stats_bayes", .pure = true, .inputs = 0 },
        .{ .id = 440, .name = "graph_create", .pure = true, .inputs = 0 },
        .{ .id = 445, .name = "graph_bfs", .pure = true, .inputs = 1 },
        .{ .id = 446, .name = "graph_dfs", .pure = true, .inputs = 1 },
        .{ .id = 447, .name = "graph_shortest_path", .pure = true, .inputs = 0 },
        .{ .id = 451, .name = "graph_pagerank", .pure = true, .inputs = 0 },
        .{ .id = 460, .name = "int_add", .pure = true, .inputs = 0 },
        .{ .id = 470, .name = "int_pow", .pure = true, .inputs = 0 },
        .{ .id = 473, .name = "bit_and", .pure = true, .inputs = 0 },
        .{ .id = 490, .name = "timestamp_now", .pure = false, .inputs = 0 },
        .{ .id = 491, .name = "timestamp_diff", .pure = true, .inputs = 0 },
    };

    for (decls, 0..) |decl, i| {
        const slot: i32 = @intCast(i * 3);
        assertIntFact(store, builtins_kb, slot, decl.id);
        assertTextFact(store, builtins_kb, slot + 1, decl.name);
        assertIntFact(store, builtins_kb, slot + 2, if (decl.pure) 1 else 0);
    }
}
