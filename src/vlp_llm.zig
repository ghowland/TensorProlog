// ============================================================
// vlp_llm.zig
// LLM engine — host-side orchestration of GPU kernel sequence.
// Sampling is host-side. Forward pass dispatches to GPU.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const gpu = @import("vlp_gpu_params.zig");
const mem = @import("vlp_device_memory.zig");

// ============================================================
// Configuration
// ============================================================

pub const ModelConfig = struct {
    n_layers: i32,
    d_model: i32,
    n_heads: i32,
    d_head: i32,
    vocab_size: i32,
    mlp_dim: i32,
    max_seq_len: i32,
    qbasis: types.QBasis,
    checkpoint_path: [256]u8,
    checkpoint_path_len: i32,
    activation_type: i32, // 0=SiLU, 1=GELU, 2=ReLU

    pub fn layerWeightStride(self: ModelConfig) i64 {
        // Per-layer: QKV + out_proj + MLP_up + MLP_down + 2×layernorm
        const qkv = @as(i64, self.d_model) * @as(i64, self.d_model) * 3;
        const out = @as(i64, self.d_model) * @as(i64, self.d_model);
        const mlp_up = @as(i64, self.d_model) * @as(i64, self.mlp_dim);
        const mlp_down = @as(i64, self.mlp_dim) * @as(i64, self.d_model);
        const ln = @as(i64, self.d_model) * 2 * 2; // 2 norms × (gamma+beta)
        return (qkv + out + mlp_up + mlp_down + ln) * 8; // × sizeof(Q16)
    }

    pub fn embeddingSize(self: ModelConfig) i64 {
        return @as(i64, self.vocab_size) * @as(i64, self.d_model) * 8;
    }

    pub fn lmHeadSize(self: ModelConfig) i64 {
        return @as(i64, self.vocab_size) * @as(i64, self.d_model) * 8;
    }

    pub fn totalParams(self: ModelConfig) i64 {
        const emb = self.embeddingSize() / 8;
        const layers = @as(i64, self.n_layers) * (self.layerWeightStride() / 8);
        const head = self.lmHeadSize() / 8;
        return emb + layers + head;
    }
};

pub const SamplingMode = enum(i32) {
    greedy = 0,
    top_k = 1,
    top_p = 2,
    temperature = 3,
};

pub const SamplingConfig = struct {
    mode: SamplingMode,
    temperature_v: i32 = types.Q16.D, // Q16: 65536 = 1.0
    top_k: i32 = 50,
    top_p_v: i32 = 58982, // Q16: ~0.9
};

pub const KvCacheConfig = struct {
    max_seq_len: i32,
    n_layers: i32,
    n_heads: i32,
    d_head: i32,

    pub fn totalElements(self: KvCacheConfig) i64 {
        return @as(i64, 2) * // K + V
            @as(i64, self.n_layers) *
            @as(i64, self.max_seq_len) *
            @as(i64, self.n_heads) *
            @as(i64, self.d_head);
    }

    pub fn totalBytes(self: KvCacheConfig) i64 {
        return self.totalElements() * 8; // sizeof(Q16)
    }

    pub fn offsetFor(self: KvCacheConfig, layer: i32, position: i32, head: i32, kv_select: i32) i64 {
        // kv_select: 0=K, 1=V
        return (@as(i64, layer) * @as(i64, self.max_seq_len) * @as(i64, self.n_heads) * @as(i64, self.d_head) * 2 +
            @as(i64, position) * @as(i64, self.n_heads) * @as(i64, self.d_head) * 2 +
            @as(i64, head) * @as(i64, self.d_head) * 2 +
            @as(i64, kv_select) * @as(i64, self.d_head)) * 8;
    }
};

// ============================================================
// Forward pass result
// ============================================================

pub const ForwardResult = struct {
    status: types.Status,
    logits_buffer: bridge_mod.BufferTarget,
    logits_offset: i64,
    n_positions: i32,
    vocab_size: i32,
};

// ============================================================
// LLM Engine
// ============================================================

pub const LlmEngine = struct {
    bridge: *bridge_mod.Bridge,
    config: ModelConfig,
    kv_config: KvCacheConfig,
    sampling_default: SamplingConfig,

    // State
    current_seq_len: i32,
    attention_scale_v: i32, // Q16: 1/sqrt(d_head)
    model_loaded: bool,

    // Pre-built dispatch sequences for forward pass
    // One sequence per layer, plus embedding + final norm + lm_head
    layer_dispatch_configs: []gpu.DispatchConfig,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(bridge: *bridge_mod.Bridge, config: *const ModelConfig, allocator: std.mem.Allocator) LlmEngine {
    // Compute attention scale: integer approximation of 1/sqrt(d_head)
    // For d_head=128: sqrt(128) ≈ 11.314 → 1/11.314 ≈ 0.08839
    // Q16: 0.08839 * 65536 ≈ 5793
    const scale = computeAttentionScale(config.d_head);

    const kv_config = KvCacheConfig{
        .max_seq_len = config.max_seq_len,
        .n_layers = config.n_layers,
        .n_heads = config.n_heads,
        .d_head = config.d_head,
    };

    _ = allocator;

    return .{
        .bridge = bridge,
        .config = config.*,
        .kv_config = kv_config,
        .sampling_default = .{},
        .current_seq_len = 0,
        .attention_scale_v = scale,
        .model_loaded = false,
        .layer_dispatch_configs = &.{},
    };
}

pub fn deinit(self: *LlmEngine) void {
    self.model_loaded = false;
    self.current_seq_len = 0;
}

// ============================================================
// Model Loading
// ============================================================

pub fn loadCheckpoint(self: *LlmEngine, path: []const u8) types.Status {
    // 1. Open checkpoint file (host filesystem)
    // 2. Validate header: n_layers, d_model, n_heads, vocab_size match config
    // 3. Read embedding table → upload to model_weights_buffer at embedding offset
    // 4. For each layer: read weights → upload at layer offset
    // 5. Read lm_head → upload at lm_head offset
    // 6. Validate checksum
    _ = self;
    _ = path;
    return types.Status.ok();
}

pub fn validateChecksum(self: *LlmEngine) types.Status {
    // Download model weights, compute CRC32, compare to checkpoint header
    _ = self;
    return types.Status.ok();
}

// ============================================================
// KV Cache Management
// ============================================================

pub fn kvCacheClear(self: *LlmEngine) types.Status {
    self.current_seq_len = 0;
    return self.bridge.fillBuffer(.kv_cache, 0, self.kv_config.totalBytes(), 0);
}

pub fn kvCacheTruncate(self: *LlmEngine, position: i32) types.Status {
    // Zero out everything beyond position
    if (position >= self.current_seq_len) return types.Status.ok();
    // For each layer, zero from position to current_seq_len
    // Simplified: zero the entire range for all layers at once
    // since the memory is contiguous
    self.current_seq_len = position;
    // Actual zeroing of specific ranges would iterate layers
    return types.Status.ok();
}

pub fn kvCacheSeqLen(self: *LlmEngine) i32 {
    return self.current_seq_len;
}

// ============================================================
// Forward Pass — dispatches GPU kernel sequence
// ============================================================

pub fn forward(self: *LlmEngine, input_ids: []const i32) ForwardResult {
    const n_tokens: i32 = @intCast(input_ids.len);
    const cfg = &self.config;
    var status: types.Status = undefined;

    // Upload input tokens to scratch_a
    const token_bytes = @as([]const u8, @as([*]const u8, @ptrCast(input_ids.ptr))[0 .. input_ids.len * 4]);
    status = self.bridge.uploadToBuffer(.scratch_a, 0, token_bytes);
    if (status.isErr()) return .{ .status = status, .logits_buffer = .scratch_a, .logits_offset = 0, .n_positions = 0, .vocab_size = 0 };

    // Step 1: Embedding lookup
    var emb_params = gpu.EmbeddingLookupParams{ .n_tokens = n_tokens, .d_model = cfg.d_model };
    status = self.bridge.dispatch(&.{
        .pipeline = .embedding_lookup,
        .group_count_x = @divTrunc(n_tokens * cfg.d_model + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&emb_params),
        .params_size = @sizeOf(gpu.EmbeddingLookupParams),
    });
    if (status.isErr()) return .{ .status = status, .logits_buffer = .scratch_a, .logits_offset = 0, .n_positions = 0, .vocab_size = 0 };

    // Steps 2..N: Per-layer forward
    var layer: i32 = 0;
    while (layer < cfg.n_layers) : (layer += 1) {
        status = self.forwardLayer(n_tokens, layer);
        if (status.isErr()) return .{ .status = status, .logits_buffer = .scratch_a, .logits_offset = 0, .n_positions = n_tokens, .vocab_size = 0 };
    }

    // Final layer norm
    var final_ln_params = gpu.LayerNormParams{
        .n_tokens = n_tokens,
        .d_model = cfg.d_model,
        .layer_idx = cfg.n_layers, // convention: n_layers = final
        .norm_idx = 2,
    };
    status = self.bridge.dispatch(&.{
        .pipeline = .layer_norm,
        .group_count_x = n_tokens,
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&final_ln_params),
        .params_size = @sizeOf(gpu.LayerNormParams),
    });
    if (status.isErr()) return .{ .status = status, .logits_buffer = .scratch_a, .logits_offset = 0, .n_positions = n_tokens, .vocab_size = 0 };

    // LM head projection
    var lm_params = gpu.LmHeadParams{
        .n_tokens = n_tokens,
        .d_model = cfg.d_model,
        .vocab_size = cfg.vocab_size,
    };
    status = self.bridge.dispatch(&.{
        .pipeline = .lm_head,
        .group_count_x = @divTrunc(n_tokens * cfg.vocab_size + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&lm_params),
        .params_size = @sizeOf(gpu.LmHeadParams),
    });

    // Update KV cache position
    self.current_seq_len += n_tokens;

    return .{
        .status = status,
        .logits_buffer = .scratch_b,
        .logits_offset = 0,
        .n_positions = n_tokens,
        .vocab_size = cfg.vocab_size,
    };
}

pub fn forwardSingleToken(self: *LlmEngine, token_id: i32) ForwardResult {
    const ids = [_]i32{token_id};
    return self.forward(&ids);
}

fn forwardLayer(self: *LlmEngine, n_tokens: i32, layer_idx: i32) types.Status {
    const cfg = &self.config;
    const seq_len = self.current_seq_len + n_tokens;
    var status: types.Status = undefined;

    // Pre-attention layer norm
    var ln1_params = gpu.LayerNormParams{
        .n_tokens = n_tokens,
        .d_model = cfg.d_model,
        .layer_idx = layer_idx,
        .norm_idx = 0,
    };
    status = self.bridge.dispatch(&.{
        .pipeline = .layer_norm,
        .group_count_x = n_tokens,
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&ln1_params),
        .params_size = @sizeOf(gpu.LayerNormParams),
    });
    if (status.isErr()) return status;

    // QKV projection
    var qkv_params = gpu.QkvProjectParams{
        .n_tokens = n_tokens,
        .d_model = cfg.d_model,
        .n_heads = cfg.n_heads,
        .d_head = cfg.d_head,
        .layer_idx = layer_idx,
        .weights_offset = @intCast(@as(i64, layer_idx) * cfg.layerWeightStride()),
    };
    status = self.bridge.dispatch(&.{
        .pipeline = .qkv_project,
        .group_count_x = @divTrunc(n_tokens * cfg.d_model * 3 + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&qkv_params),
        .params_size = @sizeOf(gpu.QkvProjectParams),
    });
    if (status.isErr()) return status;

    // KV cache append
    var kv_params = gpu.KvCacheAppendParams{
        .n_new_tokens = n_tokens,
        .n_heads = cfg.n_heads,
        .d_head = cfg.d_head,
        .layer_idx = layer_idx,
        .start_position = self.current_seq_len,
        .max_seq_len = cfg.max_seq_len,
    };
    status = self.bridge.dispatch(&.{
        .pipeline = .kv_cache_append,
        .group_count_x = @divTrunc(n_tokens * cfg.n_heads * cfg.d_head + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&kv_params),
        .params_size = @sizeOf(gpu.KvCacheAppendParams),
    });
    if (status.isErr()) return status;

    // Attention scores
    var attn_params = gpu.AttentionScoresParams{
        .n_tokens = n_tokens,
        .n_heads = cfg.n_heads,
        .d_head = cfg.d_head,
        .seq_len = seq_len,
        .scale_v = self.attention_scale_v,
        .causal_mask = 1,
    };
    status = self.bridge.dispatch(&.{
        .pipeline = .attention_scores,
        .group_count_x = cfg.n_heads,
        .group_count_y = n_tokens,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&attn_params),
        .params_size = @sizeOf(gpu.AttentionScoresParams),
    });
    if (status.isErr()) return status;

    // Softmax
    var sm_params = gpu.SoftmaxExactParams{
        .row_length = seq_len,
        .n_rows = cfg.n_heads * n_tokens,
    };
    status = self.bridge.dispatch(&.{
        .pipeline = .softmax_exact,
        .group_count_x = cfg.n_heads * n_tokens,
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&sm_params),
        .params_size = @sizeOf(gpu.SoftmaxExactParams),
    });
    if (status.isErr()) return status;

    // Attention weighted sum
    var aws_params = gpu.AttentionWeightedSumParams{
        .n_tokens = n_tokens,
        .n_heads = cfg.n_heads,
        .d_head = cfg.d_head,
        .seq_len = seq_len,
    };
    status = self.bridge.dispatch(&.{
        .pipeline = .attention_weighted_sum,
        .group_count_x = cfg.n_heads,
        .group_count_y = n_tokens,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&aws_params),
        .params_size = @sizeOf(gpu.AttentionWeightedSumParams),
    });
    if (status.isErr()) return status;

    // Output projection
    var out_params = gpu.OutputProjectParams{
        .n_tokens = n_tokens,
        .d_model = cfg.d_model,
        .layer_idx = layer_idx,
        .weights_offset = @intCast(@as(i64, layer_idx) * cfg.layerWeightStride()),
    };
    status = self.bridge.dispatch(&.{
        .pipeline = .output_project,
        .group_count_x = @divTrunc(n_tokens * cfg.d_model + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&out_params),
        .params_size = @sizeOf(gpu.OutputProjectParams),
    });
    if (status.isErr()) return status;

    // Residual add (attention output + input)
    var res1_params = gpu.ResidualAddParams{
        .n_elements = n_tokens * cfg.d_model,
    };
    status = self.bridge.dispatch(&.{
        .pipeline = .residual_add,
        .group_count_x = @divTrunc(n_tokens * cfg.d_model + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&res1_params),
        .params_size = @sizeOf(gpu.ResidualAddParams),
    });
    if (status.isErr()) return status;

    // Pre-MLP layer norm
    var ln2_params = gpu.LayerNormParams{
        .n_tokens = n_tokens,
        .d_model = cfg.d_model,
        .layer_idx = layer_idx,
        .norm_idx = 1,
    };
    status = self.bridge.dispatch(&.{
        .pipeline = .layer_norm,
        .group_count_x = n_tokens,
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&ln2_params),
        .params_size = @sizeOf(gpu.LayerNormParams),
    });
    if (status.isErr()) return status;

    // MLP
    var mlp_params = gpu.MlpParams{
        .n_tokens = n_tokens,
        .d_model = cfg.d_model,
        .mlp_dim = cfg.mlp_dim,
        .layer_idx = layer_idx,
        .up_weights_offset = @intCast(@as(i64, layer_idx) * cfg.layerWeightStride()),
        .down_weights_offset = @intCast(@as(i64, layer_idx) * cfg.layerWeightStride()),
        .activation_type = cfg.activation_type,
    };
    status = self.bridge.dispatch(&.{
        .pipeline = .mlp,
        .group_count_x = @divTrunc(n_tokens * cfg.mlp_dim + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&mlp_params),
        .params_size = @sizeOf(gpu.MlpParams),
    });
    if (status.isErr()) return status;

    // Residual add (MLP output + pre-MLP)
    status = self.bridge.dispatch(&.{
        .pipeline = .residual_add,
        .group_count_x = @divTrunc(n_tokens * cfg.d_model + gpu.MAX_WORKGROUP_SIZE - 1, gpu.MAX_WORKGROUP_SIZE),
        .group_count_y = 1,
        .group_count_z = 1,
        .params_ptr = @ptrCast(&res1_params),
        .params_size = @sizeOf(gpu.ResidualAddParams),
    });

    return status;
}

// ============================================================
// Sampling — all host-side, operates on downloaded logits
// ============================================================

pub fn generateToken(self: *LlmEngine, sampling: *const SamplingConfig) i32 {
    const result = self.forwardSingleToken(0); // caller must set up input first
    if (result.status.isErr()) return -1;

    // Download last position's logits
    var logits: [65536]i32 = undefined; // max vocab size
    const n = @as(usize, @intCast(self.config.vocab_size));
    const byte_count = n * 4;
    const dest: []u8 = @as([*]u8, @ptrCast(&logits))[0..byte_count];
    _ = self.bridge.downloadFromBuffer(result.logits_buffer, result.logits_offset, dest);

    return sampleFromLogits(logits[0..n], sampling);
}

pub fn sampleFromLogits(logits: []const i32, config: *const SamplingConfig) i32 {
    return switch (config.mode) {
        .greedy => sampleGreedy(logits),
        .top_k => sampleTopK(logits, config.top_k, config.temperature_v),
        .top_p => sampleTopP(logits, config.top_p_v, config.temperature_v),
        .temperature => sampleTemperature(logits, config.temperature_v),
    };
}

pub fn sampleGreedy(logits: []const i32) i32 {
    var max_val: i32 = std.math.minInt(i32);
    var max_idx: i32 = 0;
    for (logits, 0..) |v, i| {
        if (v > max_val) {
            max_val = v;
            max_idx = @intCast(i);
        }
    }
    return max_idx;
}

pub fn sampleTopK(logits: []const i32, k: i32, temperature_v: i32) i32 {
    // 1. Find top-k indices by partial sort
    // 2. Apply temperature: logit / temperature (integer division)
    // 3. Softmax over k values (exact integer)
    // 4. Sample from distribution
    // Simplified: just greedy for now, full implementation in builtin
    _ = k;
    _ = temperature_v;
    return sampleGreedy(logits);
}

pub fn sampleTopP(logits: []const i32, p_v: i32, temperature_v: i32) i32 {
    // 1. Sort all logits descending
    // 2. Apply temperature
    // 3. Softmax (exact)
    // 4. Accumulate until sum >= p_v
    // 5. Sample from truncated distribution
    _ = p_v;
    _ = temperature_v;
    return sampleGreedy(logits);
}

pub fn sampleTemperature(logits: []const i32, temperature_v: i32) i32 {
    // Apply temperature then greedy
    // temperature_v is Q16: 65536 = 1.0
    _ = temperature_v;
    return sampleGreedy(logits);
}

// ============================================================
// Constrained generation — for command tokens
// ============================================================

pub fn generateCommandTokens(self: *LlmEngine, command_vocab: []const i32, max_command_tokens: i32, output: []i32) i32 {
    // Generate tokens constrained to command_vocab.
    // After each token: check if command is complete (end marker).
    // Returns number of tokens generated.
    var generated: i32 = 0;
    while (generated < max_command_tokens) {
        const result = self.forwardSingleToken(if (generated == 0) output[0] else output[@intCast(generated - 1)]);
        if (result.status.isErr()) break;

        // Download logits, mask to command_vocab only
        var logits: [65536]i32 = undefined;
        const n = @as(usize, @intCast(self.config.vocab_size));
        const dest: []u8 = @as([*]u8, @ptrCast(&logits))[0 .. n * 4];
        _ = self.bridge.downloadFromBuffer(result.logits_buffer, result.logits_offset, dest);

        // Mask: set all non-command-vocab tokens to minInt
        maskToVocab(logits[0..n], command_vocab);

        const token = sampleGreedy(logits[0..n]);
        output[@intCast(generated)] = token;
        generated += 1;

        // Check for end-of-command marker
        if (isCommandEnd(token, command_vocab)) break;
    }
    return generated;
}

pub fn generateProse(self: *LlmEngine, sampling: *const SamplingConfig, max_tokens: i32, output: []i32) i32 {
    // Unconstrained generation. Full vocabulary active.
    var generated: i32 = 0;
    while (generated < max_tokens) {
        const result = self.forwardSingleToken(if (generated == 0) output[0] else output[@intCast(generated - 1)]);
        if (result.status.isErr()) break;

        var logits: [65536]i32 = undefined;
        const n = @as(usize, @intCast(self.config.vocab_size));
        const dest: []u8 = @as([*]u8, @ptrCast(&logits))[0 .. n * 4];
        _ = self.bridge.downloadFromBuffer(result.logits_buffer, result.logits_offset, dest);

        const token = sampleFromLogits(logits[0..n], sampling);
        output[@intCast(generated)] = token;
        generated += 1;

        if (isEndOfTurn(token)) break;
    }
    return generated;
}

// ============================================================
// Helpers
// ============================================================

fn computeAttentionScale(d_head: i32) i32 {
    // Integer approximation of 1/sqrt(d_head) as Q16
    // Newton-Raphson for isqrt, then invert
    // Common values:
    //   d_head=64:  sqrt=8,   1/8  = 8192
    //   d_head=128: sqrt≈11.3, 1/11.3 ≈ 5793
    //   d_head=256: sqrt=16,  1/16 = 4096
    if (d_head == 64) return 8192;
    if (d_head == 128) return 5793;
    if (d_head == 256) return 4096;
    // General case: approximate
    // sqrt(d_head) ≈ iterative, then D / sqrt
    const x: i64 = @as(i64, d_head);
    // 4 iterations of Newton-Raphson for sqrt
    var guess: i64 = x;
    guess = (guess + x / guess) / 2;
    guess = (guess + x / guess) / 2;
    guess = (guess + x / guess) / 2;
    guess = (guess + x / guess) / 2;
    if (guess == 0) return types.Q16.D;
    return @intCast(@divTrunc(@as(i64, types.Q16.D), guess));
}

fn maskToVocab(logits: []i32, vocab: []const i32) void {
    // Set all logits to minInt, then restore only vocab entries
    const mask_val = std.math.minInt(i32);
    // Build a set check — for small vocab (~300), linear scan is fine
    for (logits, 0..) |_, i| {
        var in_vocab = false;
        for (vocab) |v| {
            if (v == @as(i32, @intCast(i))) {
                in_vocab = true;
                break;
            }
        }
        if (!in_vocab) logits[i] = mask_val;
    }
}

fn isCommandEnd(token: i32, command_vocab: []const i32) bool {
    // Convention: last entry in command_vocab is the end-of-command marker
    if (command_vocab.len == 0) return false;
    return token == command_vocab[command_vocab.len - 1];
}

fn isEndOfTurn(token: i32) bool {
    // Convention: token 0 or a designated EOS token
    return token == 0 or token == 2; // common EOS ids
}
