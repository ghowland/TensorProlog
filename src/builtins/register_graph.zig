// ============================================================
// src/builtins/register_graph.zig
// ============================================================

const dispatch_reg = @import("dispatch.zig");
const BuiltinTable = dispatch_reg.BuiltinTable;
const graph_mod = @import("graph.zig");
const int_ops = @import("integer_ops.zig");
const time_ops = @import("time_ops.zig");

pub fn registerGraphBuiltins(table: *BuiltinTable) void {
    table.register(440, "graph_create", graph_mod.builtinGraphCreate, true, 0, .empty);
    table.register(441, "graph_add_node", graph_mod.builtinGraphAddNode, true, 1, .boolean);
    table.register(442, "graph_add_edge", graph_mod.builtinGraphAddEdge, true, 0, .boolean);
    table.register(443, "graph_remove_node", graph_mod.builtinGraphRemoveNode, true, 1, .boolean);
    table.register(444, "graph_remove_edge", graph_mod.builtinGraphRemoveEdge, true, 0, .boolean);
    table.register(445, "graph_bfs", graph_mod.builtinGraphBfs, true, 1, .value);
    table.register(446, "graph_dfs", graph_mod.builtinGraphDfs, true, 1, .value);
    table.register(447, "graph_shortest_path", graph_mod.builtinGraphShortestPath, true, 0, .value);
    table.register(448, "graph_topo_sort", graph_mod.builtinGraphTopoSort, true, 0, .value);
    table.register(449, "graph_components", graph_mod.builtinGraphComponents, true, 0, .value);
    table.register(450, "graph_cycle_detect", graph_mod.builtinGraphCycleDetect, true, 0, .boolean);
    table.register(451, "graph_pagerank", graph_mod.builtinGraphPageRank, true, 0, .value);
    table.register(452, "graph_markov_steady", graph_mod.builtinGraphMarkov, true, 0, .value);
}

pub fn registerIntegerOpsBuiltins(table: *BuiltinTable) void {
    table.register(460, "int_add", int_ops.builtinIntAdd, true, 0, .value);
    table.register(461, "int_sub", int_ops.builtinIntSub, true, 0, .value);
    table.register(462, "int_mul", int_ops.builtinIntMul, true, 0, .value);
    table.register(463, "int_div", int_ops.builtinIntDiv, true, 0, .value);
    table.register(464, "int_mod", int_ops.builtinIntMod, true, 0, .value);
    table.register(465, "int_abs", int_ops.builtinIntAbs, true, 1, .value);
    table.register(466, "int_sign", int_ops.builtinIntSign, true, 1, .value);
    table.register(467, "int_min", int_ops.builtinIntMin, true, 0, .value);
    table.register(468, "int_max", int_ops.builtinIntMax, true, 0, .value);
    table.register(469, "int_clamp", int_ops.builtinIntClamp, true, 0, .value);
    table.register(470, "int_pow", int_ops.builtinIntPow, true, 0, .value);
    table.register(471, "int_factorial", int_ops.builtinIntFactorial, true, 1, .value);
    table.register(472, "int_choose", int_ops.builtinIntChoose, true, 0, .value);
    table.register(473, "bit_and", int_ops.builtinBitAnd, true, 0, .value);
    table.register(474, "bit_or", int_ops.builtinBitOr, true, 0, .value);
    table.register(475, "bit_xor", int_ops.builtinBitXor, true, 0, .value);
    table.register(476, "bit_not", int_ops.builtinBitNot, true, 1, .value);
    table.register(477, "bit_shl", int_ops.builtinBitShl, true, 0, .value);
    table.register(478, "bit_shr", int_ops.builtinBitShr, true, 0, .value);
    table.register(479, "bit_popcount", int_ops.builtinBitPopcount, true, 1, .value);
    table.register(480, "bit_reverse", int_ops.builtinBitReverse, true, 1, .value);
}

pub fn registerTimeBuiltins(table: *BuiltinTable) void {
    table.register(490, "timestamp_now", time_ops.builtinTimestampNow, false, 0, .value);
    table.register(491, "timestamp_diff", time_ops.builtinTimestampDiff, true, 0, .value);
    table.register(492, "timestamp_add", time_ops.builtinTimestampAdd, true, 0, .value);
    table.register(493, "duration_seconds", time_ops.builtinDurationSeconds, true, 1, .value);
    table.register(494, "duration_minutes", time_ops.builtinDurationMinutes, true, 1, .value);
    table.register(495, "duration_hours", time_ops.builtinDurationHours, true, 1, .value);
    table.register(496, "duration_days", time_ops.builtinDurationDays, true, 1, .value);
    table.register(497, "duration_compare", time_ops.builtinDurationCompare, true, 0, .value);
    table.register(498, "duration_format", time_ops.builtinDurationFormat, true, 1, .text);
    table.register(499, "timestamp_fields", time_ops.builtinTimestampFields, true, 1, .value);
}
