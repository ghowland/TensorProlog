// ============================================================
// src/llm/forward.zig
// ============================================================

const q16_mod = @import("../vdr/q16.zig");
const model_mod = @import("model.zig");
const softmax_mod = @import("softmax.zig");

const Q16 = q16_mod.Q16;
const Model = model_mod.Model;
const ModelConfig = model_mod.ModelConfig;

pub const ForwardArena = struct {
    buf: []Q16,
    next: i32,

    pub fn init(backing: []Q16) ForwardArena {
        return .{ .buf = backing, .next = 0 };
    }

    pub fn alloc(self: *ForwardArena, count: i32) []Q16 {
        const start: usize = @intCast(self.next);
        self.next += count;
        if (self.next > @as(i32, @intCast(self.buf.len))) {
            self.next -= count;
            return self.buf[0..0];
        }
        return self.buf[start..@intCast(self.next)];
    }

    pub fn reset(self: *ForwardArena) void {
        self.next = 0;
    }
};

pub fn forward(
    model: *const Model,
    input_ids: []const i32,
    arena: *ForwardArena,
    logits: []Q16,
) void {
    arena.reset();
    const cfg = model.config;
    const seq: i32 = @intCast(input_ids.len);
    const dm = cfg.d_model;

    const hidden = arena.alloc(seq * dm);
    embed(model.embedding, input_ids, dm, hidden);

    var scratch1 = arena.alloc(seq * dm * 3);
    const scratch2 = arena.alloc(seq * dm);
    const scratch3 = arena.alloc(seq * cfg.mlp_dim);
    var scratch_norm = arena.alloc(seq * dm);

    for (model.layers) |*layer| {
        layerNorm(hidden, layer.ln1_gamma, layer.ln1_beta, seq, dm, scratch_norm);
        matmul(scratch_norm, layer.qkv, scratch1, seq, dm * 3, dm);
        const q_buf = scratch1[0..@intCast(seq * dm)];
        const k_buf = scratch1[@intCast(seq * dm)..@intCast(seq * dm * 2)];
        const v_buf = scratch1[@intCast(seq * dm * 2)..@intCast(seq * dm * 3)];
        selfAttention(q_buf, k_buf, v_buf, seq, dm, cfg.n_heads, scratch2);
        matmul(scratch2, layer.out_proj, scratch_norm, seq, dm, dm);
        addInPlace(hidden, scratch_norm);
        layerNorm(hidden, layer.ln2_gamma, layer.ln2_beta, seq, dm, scratch_norm);
        matmul(scratch_norm, layer.mlp_up, scratch3, seq, cfg.mlp_dim, dm);
        activationInPlace(scratch3);
        matmul(scratch3, layer.mlp_down, scratch_norm, seq, dm, cfg.mlp_dim);
        addInPlace(hidden, scratch_norm);
    }

    layerNorm(hidden, model.ln_final_gamma, model.ln_final_beta, seq, dm, scratch_norm);

    const last_pos = @as(usize, @intCast((seq - 1) * dm));
    const last_hidden = scratch_norm[last_pos .. last_pos + @as(usize, @intCast(dm))];
    matmulVec(last_hidden, model.lm_head, logits, cfg.vocab_size, dm);
}

fn embed(table: []const Q16, ids: []const i32, dm: i32, out: []Q16) void {
    for (ids, 0..) |id, si| {
        const row_start: usize = @intCast(id * dm);
        const out_start: usize = si * @as(usize, @intCast(dm));
        var di: i32 = 0;
        while (di < dm) : (di += 1) {
            out[out_start + @as(usize, @intCast(di))] = table[row_start + @as(usize, @intCast(di))];
        }
    }
}

pub fn layerNorm(input: []const Q16, gamma: []const Q16, beta: []const Q16, seq: i32, dm: i32, output: []Q16) void {
    var si: i32 = 0;
    while (si < seq) : (si += 1) {
        const row_start: usize = @intCast(si * dm);
        var sum: i64 = 0;
        var di: i32 = 0;
        while (di < dm) : (di += 1) {
            sum += @as(i64, input[row_start + @as(usize, @intCast(di))].v);
        }
        const mean: i32 = @intCast(@divTrunc(sum, @as(i64, dm)));

        var var_sum: i64 = 0;
        di = 0;
        while (di < dm) : (di += 1) {
            const diff: i64 = @as(i64, input[row_start + @as(usize, @intCast(di))].v) - @as(i64, mean);
            var_sum += @divTrunc(diff * diff, Q16.D_i64);
        }
        var variance: i64 = @divTrunc(var_sum, @as(i64, dm));
        if (variance <= 0) variance = 1;

        var rsqrt: i64 = Q16.D_i64;
        var iter: i32 = 0;
        while (iter < 5) : (iter += 1) {
            rsqrt = @divTrunc(rsqrt * (3 * Q16.D_i64 - @divTrunc(variance * rsqrt * rsqrt, Q16.D_i64 * Q16.D_i64)), 2 * Q16.D_i64);
        }

        di = 0;
        while (di < dm) : (di += 1) {
            const idx = row_start + @as(usize, @intCast(di));
            const centered: i64 = @as(i64, input[idx].v) - @as(i64, mean);
            const normed: i64 = @divTrunc(centered * rsqrt, Q16.D_i64);
            const scaled: i64 = @divTrunc(normed * @as(i64, gamma[@intCast(di)].v), Q16.D_i64);
            output[idx] = Q16{ .v = @intCast(scaled + @as(i64, beta[@intCast(di)].v)), .r0 = 0 };
        }
    }
}

fn matmul(a: []const Q16, b: []const Q16, out: []Q16, m: i32, n: i32, k: i32) void {
    var mi: i32 = 0;
    while (mi < m) : (mi += 1) {
        var ni: i32 = 0;
        while (ni < n) : (ni += 1) {
            var acc: i64 = 0;
            var ki: i32 = 0;
            while (ki < k) : (ki += 1) {
                const ai: usize = @intCast(mi * k + ki);
                const bi: usize = @intCast(ki * n + ni);
                acc += @as(i64, a[ai].v) * @as(i64, b[bi].v);
            }
            const oi: usize = @intCast(mi * n + ni);
            out[oi] = Q16{
                .v = @intCast(@divTrunc(acc, Q16.D_i64)),
                .r0 = @intCast(@rem(acc, Q16.D_i64)),
            };
        }
    }
}

fn matmulVec(vec: []const Q16, mat: []const Q16, out: []Q16, rows: i32, cols: i32) void {
    var ri: i32 = 0;
    while (ri < rows) : (ri += 1) {
        var acc: i64 = 0;
        var ci: i32 = 0;
        while (ci < cols) : (ci += 1) {
            const mi: usize = @intCast(ri * cols + ci);
            acc += @as(i64, vec[@intCast(ci)].v) * @as(i64, mat[mi].v);
        }
        out[@intCast(ri)] = Q16{
            .v = @intCast(@divTrunc(acc, Q16.D_i64)),
            .r0 = @intCast(@rem(acc, Q16.D_i64)),
        };
    }
}

fn selfAttention(q: []const Q16, k: []const Q16, v: []const Q16, seq: i32, dm: i32, n_heads: i32, out: []Q16) void {
    const dh = @divTrunc(dm, n_heads);
    var hi: i32 = 0;
    while (hi < n_heads) : (hi += 1) {
        var qi: i32 = 0;
        while (qi < seq) : (qi += 1) {
            var scores: [128]Q16 = undefined;
            var si: i32 = 0;
            while (si < seq) : (si += 1) {
                var dot: i64 = 0;
                var di: i32 = 0;
                while (di < dh) : (di += 1) {
                    const q_idx: usize = @intCast(qi * dm + hi * dh + di);
                    const k_idx: usize = @intCast(si * dm + hi * dh + di);
                    dot += @as(i64, q[q_idx].v) * @as(i64, k[k_idx].v);
                }
                scores[@intCast(si)] = Q16{
                    .v = @intCast(@divTrunc(dot, Q16.D_i64)),
                    .r0 = 0,
                };
                if (si > qi) scores[@intCast(si)] = Q16.zero();
            }

            var probs: [128]Q16 = undefined;
            softmax_mod.softmaxSurrogate(scores[0..@intCast(seq)], probs[0..@intCast(seq)]);

            var di: i32 = 0;
            while (di < dh) : (di += 1) {
                var acc: i64 = 0;
                si = 0;
                while (si < seq) : (si += 1) {
                    const v_idx: usize = @intCast(si * dm + hi * dh + di);
                    acc += @as(i64, probs[@intCast(si)].v) * @as(i64, v[v_idx].v);
                }
                const o_idx: usize = @intCast(qi * dm + hi * dh + di);
                out[o_idx] = Q16{
                    .v = @intCast(@divTrunc(acc, Q16.D_i64)),
                    .r0 = 0,
                };
            }
        }
    }
}

fn addInPlace(a: []Q16, b: []const Q16) void {
    for (a, 0..) |*av, i| {
        av.* = Q16.add(av.*, b[i]);
    }
}

fn activationInPlace(data: []Q16) void {
    for (data) |*v| {
        if (v.v < 0) v.* = Q16.zero();
    }
}
