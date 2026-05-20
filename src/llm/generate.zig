// ============================================================
// src/llm/generate.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");
const model_mod = @import("model.zig");
const forward_mod = @import("forward.zig");
const softmax_mod = @import("softmax.zig");
const kv_cache_mod = @import("kv_cache.zig");
const sampling_mod = @import("sampling.zig");
const kb_store_mod = @import("../kb/store.zig");
const command_parse = @import("../engine/command_parse.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;
const Model = model_mod.Model;
const KVCache = kv_cache_mod.KVCache;
const SamplingConfig = sampling_mod.SamplingConfig;
const KBStore = kb_store_mod.KBStore;
const VlpCommand = command_parse.VlpCommand;

pub const GenerateState = struct {
    model: *const Model,
    kv_cache: *KVCache,
    sampling_config: SamplingConfig,
    position: i32,
    max_seq_len: i32,
    logits_buf: []Q16,
    probs_buf: []Q16,
    hidden_buf: []Q16,
    rng: sampling_mod.RNG,

    pub fn init(
        model: *const Model,
        kv_cache: *KVCache,
        sampling_config: SamplingConfig,
        logits_buf: []Q16,
        probs_buf: []Q16,
        hidden_buf: []Q16,
        max_seq_len: i32,
        seed: i64,
    ) GenerateState {
        return .{
            .model = model,
            .kv_cache = kv_cache,
            .sampling_config = sampling_config,
            .position = 0,
            .max_seq_len = max_seq_len,
            .logits_buf = logits_buf,
            .probs_buf = probs_buf,
            .hidden_buf = hidden_buf,
            .rng = sampling_mod.RNG.init(seed),
        };
    }
};

pub fn prefill(
    state: *GenerateState,
    input_ids: []const i32,
) void {
    const vocab: usize = @intCast(state.model.config.vocab_size);
    const d_model: usize = @intCast(state.model.config.d_model);
    // const seq_len = input_ids.len;

    for (input_ids, 0..) |token_id, pos| {
        const tid: usize = @intCast(token_id);
        const embed_start = tid * d_model;
        const embed_end = embed_start + d_model;
        const embed_slice = state.model.embedding[embed_start..embed_end];

        forward_mod.forwardSinglePosition(
            state.model,
            embed_slice,
            state.kv_cache,
            @intCast(pos),
            state.hidden_buf,
            state.logits_buf,
        );

        state.position = @intCast(pos + 1);
    }

    _ = vocab;
}

pub fn generateToken(
    state: *GenerateState,
) i32 {
    const vocab: usize = @intCast(state.model.config.vocab_size);

    softmax_mod.softmaxSurrogate(
        state.logits_buf[0..vocab],
        state.probs_buf[0..vocab],
    );

    var token: i32 = undefined;

    if (state.sampling_config.greedy) {
        token = sampling_mod.sampleGreedy(state.probs_buf[0..vocab]);
    } else {
        var temp_buf: [65536]Q16 = undefined;

        if (state.sampling_config.temperature.v != Q16.D) {
            sampling_mod.applyTemperature(
                state.logits_buf[0..vocab],
                state.sampling_config.temperature,
                temp_buf[0..vocab],
            );
            softmax_mod.softmaxSurrogate(
                temp_buf[0..vocab],
                state.probs_buf[0..vocab],
            );
        }

        if (state.sampling_config.top_k > 0 and state.sampling_config.top_k < @as(i32, @intCast(vocab))) {
            token = sampling_mod.sampleTopK(
                state.probs_buf[0..vocab],
                state.sampling_config.top_k,
                &state.rng,
            );
        } else if (state.sampling_config.top_p.v < Q16.D) {
            token = sampling_mod.sampleTopP(
                state.probs_buf[0..vocab],
                state.sampling_config.top_p,
                &state.rng,
            );
        } else {
            token = sampling_mod.sampleGreedy(state.probs_buf[0..vocab]);
        }
    }

    if (state.position < state.max_seq_len) {
        const d_model: usize = @intCast(state.model.config.d_model);
        const tid: usize = @intCast(token);
        const embed_start = tid * d_model;
        const embed_end = embed_start + d_model;
        const embed_slice = state.model.embedding[embed_start..embed_end];

        forward_mod.forwardSinglePosition(
            state.model,
            embed_slice,
            state.kv_cache,
            state.position,
            state.hidden_buf,
            state.logits_buf,
        );

        state.position += 1;
    }

    return token;
}

pub fn generateCommand(
    state: *GenerateState,
    store: *KBStore,
    command_out: *VlpCommand,
    tokens_used: *i32,
) VlpStatus {
    var cmd_tokens: [32]i32 = undefined;
    var cmd_len: i32 = 0;
    const max_cmd_tokens: i32 = 32;

    while (cmd_len < max_cmd_tokens) {
        const vocab: usize = @intCast(state.model.config.vocab_size);

        softmax_mod.softmaxSurrogate(
            state.logits_buf[0..vocab],
            state.probs_buf[0..vocab],
        );

        maskNonCommandTokens(state.probs_buf[0..vocab], state.model.config.vocab_size);

        renormalize(state.probs_buf[0..vocab]);

        const token = sampling_mod.sampleGreedy(state.probs_buf[0..vocab]);

        const ci: usize = @intCast(cmd_len);
        cmd_tokens[ci] = token;
        cmd_len += 1;

        if (isCommandEnd(token)) {
            break;
        }

        if (state.position < state.max_seq_len) {
            const d_model: usize = @intCast(state.model.config.d_model);
            const tid: usize = @intCast(token);
            const embed_start = tid * d_model;
            const embed_end = embed_start + d_model;
            const embed_slice = state.model.embedding[embed_start..embed_end];

            forward_mod.forwardSinglePosition(
                state.model,
                embed_slice,
                state.kv_cache,
                state.position,
                state.hidden_buf,
                state.logits_buf,
            );

            state.position += 1;
        }
    }

    tokens_used.* = cmd_len;

    const cl: usize = @intCast(cmd_len);
    return command_parse.commandParse(&cmd_tokens, @intCast(cl), store, command_out);
}

pub fn generateProse(
    state: *GenerateState,
    max_tokens: i32,
    output_ids: []i32,
) i32 {
    var generated: i32 = 0;
    const mt: usize = @intCast(max_tokens);
    const limit = @min(mt, output_ids.len);

    for (0..limit) |i| {
        const token = generateToken(state);

        if (isEndOfTurn(token)) {
            break;
        }

        output_ids[i] = token;
        generated += 1;
    }

    return generated;
}

fn maskNonCommandTokens(probs: []Q16, vocab_size: i32) void {
    const cmd_vocab_start: i32 = 0;
    const cmd_vocab_end: i32 = @min(300, vocab_size);

    const vs: usize = @intCast(vocab_size);
    const ce: usize = @intCast(cmd_vocab_end);

    for (ce..vs) |i| {
        probs[i] = Q16.zero();
    }

    _ = cmd_vocab_start;
}

fn renormalize(probs: []Q16) void {
    var sum: i64 = 0;
    for (probs) |p| {
        sum += @intCast(p.v);
    }

    if (sum == 0) {
        if (probs.len > 0) {
            probs[0] = .{ .v = Q16.D, .r0 = 0 };
        }
        return;
    }

    if (sum == @as(i64, Q16.D)) {
        return;
    }

    for (probs) |*p| {
        const pv: i64 = @intCast(p.v);
        const scaled: i64 = pv * @as(i64, Q16.D);
        p.v = @intCast(@divTrunc(scaled, sum));
        p.r0 = @intCast(scaled - @as(i64, @intCast(p.v)) * sum);
    }

    var new_sum: i64 = 0;
    for (probs) |p| {
        new_sum += @intCast(p.v);
    }
    const diff: i64 = @as(i64, Q16.D) - new_sum;
    if (diff != 0 and probs.len > 0) {
        var max_idx: usize = 0;
        var max_val: i32 = probs[0].v;
        for (probs[1..], 1..) |p, i| {
            if (p.v > max_val) {
                max_val = p.v;
                max_idx = i;
            }
        }
        probs[max_idx].v += @intCast(diff);
    }
}

fn isCommandEnd(token: i32) bool {
    return token == 299;
}

fn isEndOfTurn(token: i32) bool {
    return token == 0;
}
