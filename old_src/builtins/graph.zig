// ============================================================
// src/builtins/graph.zig
// ============================================================

const std = @import("std");
const q16_mod = @import("../vdr/q16.zig");
const types = @import("../vdr/types.zig");
const dispatch_mod = @import("dispatch.zig");

const Q16 = q16_mod.Q16;
const VlpStatus = types.VlpStatus;
const BuiltinArgs = dispatch_mod.BuiltinArgs;
const BuiltinResult = dispatch_mod.BuiltinResult;

pub const Edge = struct {
    from: i32,
    to: i32,
    weight: Q16,
};

pub const Graph = struct {
    nodes: []i32,
    node_count: i32,
    node_capacity: i32,
    edges: []Edge,
    edge_count: i32,
    edge_capacity: i32,

    pub fn init(node_buf: []i32, edge_buf: []Edge) Graph {
        return .{
            .nodes = node_buf,
            .node_count = 0,
            .node_capacity = @intCast(node_buf.len),
            .edges = edge_buf,
            .edge_count = 0,
            .edge_capacity = @intCast(edge_buf.len),
        };
    }

    pub fn addNode(self: *Graph, node_id: i32) bool {
        for (0..@as(usize, @intCast(self.node_count))) |i| {
            if (self.nodes[i] == node_id) return false;
        }
        if (self.node_count >= self.node_capacity) return false;
        self.nodes[@intCast(self.node_count)] = node_id;
        self.node_count += 1;
        return true;
    }

    pub fn removeNode(self: *Graph, node_id: i32) bool {
        var found = false;
        const nc: usize = @intCast(self.node_count);
        for (0..nc) |i| {
            if (self.nodes[i] == node_id) {
                self.nodes[i] = self.nodes[nc - 1];
                self.node_count -= 1;
                found = true;
                break;
            }
        }
        if (!found) return false;
        var w: usize = 0;
        const ec: usize = @intCast(self.edge_count);
        for (0..ec) |i| {
            if (self.edges[i].from != node_id and self.edges[i].to != node_id) {
                self.edges[w] = self.edges[i];
                w += 1;
            }
        }
        self.edge_count = @intCast(w);
        return true;
    }

    pub fn addEdge(self: *Graph, from: i32, to: i32, weight: Q16) bool {
        if (self.edge_count >= self.edge_capacity) return false;
        self.edges[@intCast(self.edge_count)] = .{ .from = from, .to = to, .weight = weight };
        self.edge_count += 1;
        return true;
    }

    pub fn removeEdge(self: *Graph, from: i32, to: i32) bool {
        const ec: usize = @intCast(self.edge_count);
        for (0..ec) |i| {
            if (self.edges[i].from == from and self.edges[i].to == to) {
                self.edges[i] = self.edges[ec - 1];
                self.edge_count -= 1;
                return true;
            }
        }
        return false;
    }

    fn nodeIndex(self: *const Graph, node_id: i32) ?usize {
        const nc: usize = @intCast(self.node_count);
        for (0..nc) |i| {
            if (self.nodes[i] == node_id) return i;
        }
        return null;
    }
};

pub fn bfs(graph: *const Graph, start: i32, visited: []i32) i32 {
    const nc: usize = @intCast(graph.node_count);
    var seen: [1024]bool = [_]bool{false} ** 1024;
    var queue: [1024]i32 = undefined;
    var q_head: usize = 0;
    var q_tail: usize = 0;
    var count: usize = 0;

    const start_idx = graph.nodeIndex(start) orelse return 0;
    if (start_idx >= seen.len) return 0;
    seen[start_idx] = true;
    queue[q_tail] = start;
    q_tail += 1;

    while (q_head < q_tail) {
        const current = queue[q_head];
        q_head += 1;

        if (count < visited.len) {
            visited[count] = current;
            count += 1;
        }

        const ec: usize = @intCast(graph.edge_count);
        for (0..ec) |e| {
            if (graph.edges[e].from == current) {
                const to = graph.edges[e].to;
                const to_idx = graph.nodeIndex(to) orelse continue;
                if (to_idx < seen.len and !seen[to_idx]) {
                    seen[to_idx] = true;
                    if (q_tail < queue.len) {
                        queue[q_tail] = to;
                        q_tail += 1;
                    }
                }
            }
        }
    }

    _ = nc;
    return @intCast(count);
}

pub fn dfs(graph: *const Graph, start: i32, visited: []i32) i32 {
    var seen: [1024]bool = [_]bool{false} ** 1024;
    var stack: [1024]i32 = undefined;
    var sp: usize = 0;
    var count: usize = 0;

    const start_idx = graph.nodeIndex(start) orelse return 0;
    if (start_idx >= seen.len) return 0;

    stack[sp] = start;
    sp += 1;

    while (sp > 0) {
        sp -= 1;
        const current = stack[sp];
        const cur_idx = graph.nodeIndex(current) orelse continue;
        if (cur_idx >= seen.len) continue;
        if (seen[cur_idx]) continue;
        seen[cur_idx] = true;

        if (count < visited.len) {
            visited[count] = current;
            count += 1;
        }

        const ec: usize = @intCast(graph.edge_count);
        for (0..ec) |e| {
            if (graph.edges[e].from == current) {
                const to = graph.edges[e].to;
                const to_idx = graph.nodeIndex(to) orelse continue;
                if (to_idx < seen.len and !seen[to_idx] and sp < stack.len) {
                    stack[sp] = to;
                    sp += 1;
                }
            }
        }
    }

    return @intCast(count);
}

pub fn shortestPath(graph: *const Graph, from: i32, to: i32, path: []i32, distance: *Q16) i32 {
    const nc: usize = @intCast(graph.node_count);
    if (nc == 0 or nc > 1024) {
        distance.* = Q16.zero();
        return 0;
    }

    var dist: [1024]i64 = undefined;
    var prev: [1024]i32 = undefined;
    var visited_flags: [1024]bool = [_]bool{false} ** 1024;
    const max_dist: i64 = 0x7FFFFFFFFFFFFFFF;

    for (0..nc) |i| {
        dist[i] = max_dist;
        prev[i] = -1;
    }

    const from_idx = graph.nodeIndex(from) orelse {
        distance.* = Q16.zero();
        return 0;
    };
    dist[from_idx] = 0;

    for (0..nc) |_| {
        var u: usize = nc;
        var u_dist: i64 = max_dist;
        for (0..nc) |i| {
            if (!visited_flags[i] and dist[i] < u_dist) {
                u_dist = dist[i];
                u = i;
            }
        }
        if (u >= nc) break;
        visited_flags[u] = true;

        const u_node = graph.nodes[u];
        const ec: usize = @intCast(graph.edge_count);
        for (0..ec) |e| {
            if (graph.edges[e].from == u_node) {
                const v_node = graph.edges[e].to;
                const v_idx = graph.nodeIndex(v_node) orelse continue;
                const w: i64 = @intCast(graph.edges[e].weight.v);
                const alt = dist[u] + w;
                if (alt < dist[v_idx]) {
                    dist[v_idx] = alt;
                    prev[v_idx] = @intCast(u);
                }
            }
        }
    }

    const to_idx = graph.nodeIndex(to) orelse {
        distance.* = Q16.zero();
        return 0;
    };

    if (dist[to_idx] == max_dist) {
        distance.* = Q16.zero();
        return 0;
    }

    distance.* = .{ .v = @intCast(dist[to_idx]), .r0 = 0 };

    var rev_path: [1024]i32 = undefined;
    var rev_len: usize = 0;
    var cur: usize = to_idx;
    while (cur < nc) {
        if (rev_len < rev_path.len) {
            rev_path[rev_len] = graph.nodes[cur];
            rev_len += 1;
        }
        if (prev[cur] < 0) break;
        cur = @intCast(prev[cur]);
    }

    const copy_len = @min(rev_len, path.len);
    for (0..copy_len) |i| {
        path[i] = rev_path[rev_len - 1 - i];
    }

    return @intCast(copy_len);
}

pub fn topologicalSort(graph: *const Graph, sorted: []i32) i32 {
    const nc: usize = @intCast(graph.node_count);
    if (nc == 0 or nc > 1024) return 0;

    var in_degree: [1024]i32 = [_]i32{0} ** 1024;
    const ec: usize = @intCast(graph.edge_count);
    for (0..ec) |e| {
        const to_idx = graph.nodeIndex(graph.edges[e].to) orelse continue;
        if (to_idx < in_degree.len) in_degree[to_idx] += 1;
    }

    var queue: [1024]usize = undefined;
    var q_head: usize = 0;
    var q_tail: usize = 0;

    for (0..nc) |i| {
        if (in_degree[i] == 0) {
            queue[q_tail] = i;
            q_tail += 1;
        }
    }

    var count: usize = 0;
    while (q_head < q_tail) {
        const u = queue[q_head];
        q_head += 1;
        if (count < sorted.len) {
            sorted[count] = graph.nodes[u];
            count += 1;
        }
        const u_node = graph.nodes[u];
        for (0..ec) |e| {
            if (graph.edges[e].from == u_node) {
                const v_idx = graph.nodeIndex(graph.edges[e].to) orelse continue;
                if (v_idx < in_degree.len) {
                    in_degree[v_idx] -= 1;
                    if (in_degree[v_idx] == 0 and q_tail < queue.len) {
                        queue[q_tail] = v_idx;
                        q_tail += 1;
                    }
                }
            }
        }
    }

    return @intCast(count);
}

pub fn connectedComponents(graph: *const Graph, components: []i32) i32 {
    const nc: usize = @intCast(graph.node_count);
    if (nc == 0 or nc > 1024) return 0;

    var comp_id: [1024]i32 = [_]i32{-1} ** 1024;
    var current_comp: i32 = 0;

    for (0..nc) |i| {
        if (comp_id[i] >= 0) continue;
        var stack: [1024]usize = undefined;
        var sp: usize = 0;
        stack[sp] = i;
        sp += 1;
        while (sp > 0) {
            sp -= 1;
            const u = stack[sp];
            if (comp_id[u] >= 0) continue;
            comp_id[u] = current_comp;
            const u_node = graph.nodes[u];
            const ec: usize = @intCast(graph.edge_count);
            for (0..ec) |e| {
                if (graph.edges[e].from == u_node) {
                    const v_idx = graph.nodeIndex(graph.edges[e].to) orelse continue;
                    if (v_idx < comp_id.len and comp_id[v_idx] < 0 and sp < stack.len) {
                        stack[sp] = v_idx;
                        sp += 1;
                    }
                }
                if (graph.edges[e].to == u_node) {
                    const v_idx = graph.nodeIndex(graph.edges[e].from) orelse continue;
                    if (v_idx < comp_id.len and comp_id[v_idx] < 0 and sp < stack.len) {
                        stack[sp] = v_idx;
                        sp += 1;
                    }
                }
            }
        }
        current_comp += 1;
    }

    const copy_len = @min(nc, components.len);
    for (0..copy_len) |i| {
        components[i] = comp_id[i];
    }

    return current_comp;
}

pub fn cycleDetect(graph: *const Graph) bool {
    const nc: usize = @intCast(graph.node_count);
    if (nc == 0 or nc > 1024) return false;

    const WHITE: i8 = 0;
    const GRAY: i8 = 1;
    const BLACK: i8 = 2;
    var color: [1024]i8 = [_]i8{WHITE} ** 1024;

    for (0..nc) |i| {
        if (color[i] == WHITE) {
            if (cycleDfsVisit(graph, i, &color)) return true;
        }
    }
    return false;
}

fn cycleDfsVisit(graph: *const Graph, u: usize, color: *[1024]i8) bool {
    color[u] = 1;
    const u_node = graph.nodes[u];
    const ec: usize = @intCast(graph.edge_count);
    for (0..ec) |e| {
        if (graph.edges[e].from == u_node) {
            const v_idx = graph.nodeIndex(graph.edges[e].to) orelse continue;
            if (v_idx >= 1024) continue;
            if (color[v_idx] == 1) return true;
            if (color[v_idx] == 0) {
                if (cycleDfsVisit(graph, v_idx, color)) return true;
            }
        }
    }
    color[u] = 2;
    return false;
}

pub fn pageRankExact(graph: *const Graph, ranks: []Q16, n_iterations: i32) void {
    const nc: usize = @intCast(graph.node_count);
    if (nc == 0 or nc > 1024) return;
    const ec: usize = @intCast(graph.edge_count);

    const initial_rank: i32 = @intCast(@divTrunc(@as(i64, Q16.D), @as(i64, @intCast(nc))));
    for (0..nc) |i| {
        ranks[i] = .{ .v = initial_rank, .r0 = 0 };
    }

    var sum: i64 = 0;
    for (0..nc) |i| {
        sum += @intCast(ranks[i].v);
    }
    const diff_init: i64 = @as(i64, Q16.D) - sum;
    if (diff_init != 0 and nc > 0) {
        ranks[0].v += @intCast(diff_init);
    }

    const damping_v: i64 = 55705;
    const one_minus_d: i64 = @as(i64, Q16.D) - damping_v;
    var new_ranks: [1024]Q16 = undefined;

    var out_degree: [1024]i32 = [_]i32{0} ** 1024;
    for (0..ec) |e| {
        const from_idx = graph.nodeIndex(graph.edges[e].from) orelse continue;
        if (from_idx < out_degree.len) out_degree[from_idx] += 1;
    }

    const iters: usize = @intCast(@max(n_iterations, 1));
    for (0..iters) |_| {
        const base: i64 = @divTrunc(one_minus_d, @as(i64, @intCast(nc)));
        for (0..nc) |i| {
            new_ranks[i] = .{ .v = @intCast(base), .r0 = 0 };
        }

        for (0..ec) |e| {
            const from_idx = graph.nodeIndex(graph.edges[e].from) orelse continue;
            const to_idx = graph.nodeIndex(graph.edges[e].to) orelse continue;
            if (from_idx >= 1024 or to_idx >= 1024) continue;
            const od = out_degree[from_idx];
            if (od == 0) continue;
            const contrib: i64 = @divTrunc(@as(i64, @intCast(ranks[from_idx].v)) * damping_v, @as(i64, Q16.D) * @as(i64, @intCast(od)));
            new_ranks[to_idx].v += @intCast(contrib);
        }

        var rank_sum: i64 = 0;
        for (0..nc) |i| {
            rank_sum += @intCast(new_ranks[i].v);
        }
        const rank_diff: i64 = @as(i64, Q16.D) - rank_sum;
        if (rank_diff != 0 and nc > 0) {
            var max_idx: usize = 0;
            var max_v: i32 = new_ranks[0].v;
            for (1..nc) |i| {
                if (new_ranks[i].v > max_v) {
                    max_v = new_ranks[i].v;
                    max_idx = i;
                }
            }
            new_ranks[max_idx].v += @intCast(rank_diff);
        }

        for (0..nc) |i| {
            ranks[i] = new_ranks[i];
        }
    }
}

pub fn markovSteady(transition: []const Q16, steady: []Q16, n: i32) void {
    const sz: usize = @intCast(n);
    if (sz == 0 or sz > 256) return;

    const initial: i32 = @intCast(@divTrunc(@as(i64, Q16.D), @as(i64, @intCast(sz))));
    for (0..sz) |i| {
        steady[i] = .{ .v = initial, .r0 = 0 };
    }
    var s: i64 = 0;
    for (0..sz) |i| s += @intCast(steady[i].v);
    if (s != @as(i64, Q16.D) and sz > 0) {
        steady[0].v += @intCast(@as(i64, Q16.D) - s);
    }

    var next: [256]Q16 = undefined;
    const max_iters: usize = 200;

    for (0..max_iters) |_| {
        for (0..sz) |i| {
            var acc: i64 = 0;
            for (0..sz) |j| {
                const sv: i64 = @intCast(steady[j].v);
                const tv: i64 = @intCast(transition[j * sz + i].v);
                acc += @divTrunc(sv * tv, @as(i64, Q16.D));
            }
            next[i] = .{ .v = @intCast(acc), .r0 = 0 };
        }

        var norm_sum: i64 = 0;
        for (0..sz) |i| norm_sum += @intCast(next[i].v);
        if (norm_sum != @as(i64, Q16.D) and norm_sum > 0 and sz > 0) {
            var max_idx: usize = 0;
            var max_v: i32 = next[0].v;
            for (1..sz) |i| {
                if (next[i].v > max_v) {
                    max_v = next[i].v;
                    max_idx = i;
                }
            }
            next[max_idx].v += @intCast(@as(i64, Q16.D) - norm_sum);
        }

        var converged = true;
        for (0..sz) |i| {
            if (steady[i].v != next[i].v) {
                converged = false;
                break;
            }
        }

        for (0..sz) |i| steady[i] = next[i];
        if (converged) break;
    }
}

pub fn builtinGraphCreate(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
pub fn builtinGraphAddNode(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
pub fn builtinGraphAddEdge(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
pub fn builtinGraphRemoveNode(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
pub fn builtinGraphRemoveEdge(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
pub fn builtinGraphBfs(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
pub fn builtinGraphDfs(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
pub fn builtinGraphShortestPath(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
pub fn builtinGraphTopoSort(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
pub fn builtinGraphComponents(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
pub fn builtinGraphCycleDetect(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
pub fn builtinGraphPageRank(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
pub fn builtinGraphMarkov(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}
