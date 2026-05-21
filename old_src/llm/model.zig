// ============================================================
// src/llm/model.zig
// ============================================================

const q16_mod = @import("../vdr/q16.zig");
const vdr_types = @import("../vdr/types.zig");

const Q16 = q16_mod.Q16;
const VlpStatus = vdr_types.VlpStatus;

pub const ModelConfig = struct {
    n_layers: i32 = 1,
    d_model: i32 = 4,
    n_heads: i32 = 1,
    d_head: i32 = 4,
    vocab_size: i32 = 5,
    mlp_dim: i32 = 8,
};

pub const LayerWeights = struct {
    ln1_gamma: []Q16,
    ln1_beta: []Q16,
    qkv: []Q16,
    out_proj: []Q16,
    ln2_gamma: []Q16,
    ln2_beta: []Q16,
    mlp_up: []Q16,
    mlp_down: []Q16,
};

pub const Model = struct {
    config: ModelConfig,
    embedding: []Q16,
    layers: []LayerWeights,
    ln_final_gamma: []Q16,
    ln_final_beta: []Q16,
    lm_head: []Q16,

    pub fn totalParams(self: *const Model) i64 {
        var total: i64 = @as(i64, self.config.vocab_size) * @as(i64, self.config.d_model);
        const dm: i64 = @as(i64, self.config.d_model);
        const mlp: i64 = @as(i64, self.config.mlp_dim);
        const n: i64 = @as(i64, self.config.n_layers);
        total += n * (dm + dm);
        total += n * (dm * dm * 3);
        total += n * (dm * dm);
        total += n * (dm + dm);
        total += n * (dm * mlp);
        total += n * (mlp * dm);
        total += dm + dm;
        total += @as(i64, self.config.vocab_size) * dm;
        return total;
    }
};

pub const ModelArena = struct {
    weights: []Q16,
    next: i32,

    pub fn init(backing: []Q16) ModelArena {
        for (backing) |*w| w.* = Q16.zero();
        return .{ .weights = backing, .next = 0 };
    }

    pub fn alloc(self: *ModelArena, count: i32) ?[]Q16 {
        if (self.next + count > @as(i32, @intCast(self.weights.len))) return null;
        const start: usize = @intCast(self.next);
        self.next += count;
        return self.weights[start..@intCast(self.next)];
    }

    pub fn allocModel(self: *ModelArena, cfg: ModelConfig) ?Model {
        const dm = cfg.d_model;
        const vs = cfg.vocab_size;
        const mlp = cfg.mlp_dim;

        const embedding = self.alloc(vs * dm) orelse return null;
        const ln_final_gamma = self.alloc(dm) orelse return null;
        const ln_final_beta = self.alloc(dm) orelse return null;
        const lm_head = self.alloc(vs * dm) orelse return null;

        var layer_buf: [64]LayerWeights = undefined;
        var li: i32 = 0;
        while (li < cfg.n_layers) : (li += 1) {
            layer_buf[@intCast(li)] = .{
                .ln1_gamma = self.alloc(dm) orelse return null,
                .ln1_beta = self.alloc(dm) orelse return null,
                .qkv = self.alloc(dm * dm * 3) orelse return null,
                .out_proj = self.alloc(dm * dm) orelse return null,
                .ln2_gamma = self.alloc(dm) orelse return null,
                .ln2_beta = self.alloc(dm) orelse return null,
                .mlp_up = self.alloc(dm * mlp) orelse return null,
                .mlp_down = self.alloc(mlp * dm) orelse return null,
            };
        }

        return Model{
            .config = cfg,
            .embedding = embedding,
            .layers = layer_buf[0..@intCast(cfg.n_layers)],
            .ln_final_gamma = ln_final_gamma,
            .ln_final_beta = ln_final_beta,
            .lm_head = lm_head,
        };
    }
};

pub fn initWeightsSmall(model: *Model, seed: i32) void {
    var s: i64 = @as(i64, seed);
    fillDeterministic(model.embedding, &s);
    for (model.layers) |*layer| {
        fillOnes(layer.ln1_gamma);
        fillZeros(layer.ln1_beta);
        fillDeterministic(layer.qkv, &s);
        fillDeterministic(layer.out_proj, &s);
        fillOnes(layer.ln2_gamma);
        fillZeros(layer.ln2_beta);
        fillDeterministic(layer.mlp_up, &s);
        fillDeterministic(layer.mlp_down, &s);
    }
    fillOnes(model.ln_final_gamma);
    fillZeros(model.ln_final_beta);
    fillDeterministic(model.lm_head, &s);
}

fn fillDeterministic(buf: []Q16, s: *i64) void {
    for (buf) |*v| {
        s.* = s.* *% 6364136223846793005 +% 1442695040888963407;
        const val: i32 = @intCast(@mod(@divTrunc(s.*, 65536), 1000) - 500);
        v.* = Q16{ .v = val, .r0 = 0 };
    }
}

fn fillOnes(buf: []Q16) void {
    for (buf) |*v| v.* = Q16.one();
}

fn fillZeros(buf: []Q16) void {
    for (buf) |*v| v.* = Q16.zero();
}
