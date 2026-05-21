// ============================================================
// vlp_kernel.zig
// THE kernel. One entry point. 28 ops. Compiled to spirv32-vulkan.
// Imports vlp_gpu_shared.zig for all constants and field offsets.
// ============================================================

const std = @import("std");

comptime {
    std.gpu.executionMode(main, .{ .local_size = .{ .x = 256, .y = 1, .z = 1 } });
}

const shared = @import("vlp_gpu_shared.zig");
const gpu = @import("std").gpu;

// ============================================================
// Buffer declarations — fixed-size arrays, extern struct wrappers
// ============================================================

const ModelBuf = extern struct { data: [shared.MAX_BUFFER_INTS]i32 };
const KbBuf = extern struct { data: [shared.MAX_BUFFER_INTS]i32 };
const ScratchBuf = extern struct { data: [shared.MAX_BUFFER_INTS]i32 };
const KvBuf = extern struct { data: [shared.MAX_KV_CACHE_INTS]i32 };
const ParamsBuf = extern struct { data: [shared.PARAMS_INTS]i32 };
const StatusBuf = extern struct { data: [shared.MAX_STATUS_ENTRIES]i32 };
const ResultBuf = extern struct { data: [shared.MAX_RESULT_SLOTS]i32 };

// Set 0 — Model
extern var embedding_table: ModelBuf addrspace(.storage_buffer);
extern var layer_weights: ModelBuf addrspace(.storage_buffer);
extern var lm_head_weights: ModelBuf addrspace(.storage_buffer);
extern var ln_params: ModelBuf addrspace(.storage_buffer);

// Set 1 — KB Data
extern var kb_store: KbBuf addrspace(.storage_buffer);
extern var fact_store: KbBuf addrspace(.storage_buffer);
extern var rule_store: KbBuf addrspace(.storage_buffer);
extern var term_store: KbBuf addrspace(.storage_buffer);
extern var live_state: KbBuf addrspace(.storage_buffer);

// Set 2 — Scratch
extern var scratch_a: ScratchBuf addrspace(.storage_buffer);
extern var scratch_b: ScratchBuf addrspace(.storage_buffer);
extern var kv_cache: KvBuf addrspace(.storage_buffer);

// Set 3 — Control
extern var params: ParamsBuf addrspace(.uniform);
extern var status_buf: StatusBuf addrspace(.storage_buffer);
extern var result_counts: ResultBuf addrspace(.storage_buffer);

// ============================================================
// Shared memory for reductions and softmax
// ============================================================

var s_i32: [shared.MAX_WORKGROUP]i32 addrspace(.shared) = undefined;
var s_i64: [shared.MAX_WORKGROUP]i64 addrspace(.shared) = undefined;
var s_idx: [shared.MAX_WORKGROUP]i32 addrspace(.shared) = undefined;

// ============================================================
// Entry point
// ============================================================

export fn main() callconv(.spirv_kernel) void {
    // @setExecProperty(.local_size, .{ 256, 1, 1 });
    // comptime {
    //     gpu.executionMode(main, .{ .local_size = .{ .x = 256, .y = 1, .z = 1 } });
    // }

    // Bind all buffers to descriptor sets
    declareBinding(&embedding_table, shared.SET_MODEL, shared.BIND_EMBEDDING);
    declareBinding(&layer_weights, shared.SET_MODEL, shared.BIND_LAYER_WEIGHTS);
    declareBinding(&lm_head_weights, shared.SET_MODEL, shared.BIND_LM_HEAD);
    declareBinding(&ln_params, shared.SET_MODEL, shared.BIND_LN_PARAMS);

    declareBinding(&kb_store, shared.SET_KB_DATA, shared.BIND_KB_STORE);
    declareBinding(&fact_store, shared.SET_KB_DATA, shared.BIND_FACT_STORE);
    declareBinding(&rule_store, shared.SET_KB_DATA, shared.BIND_RULE_STORE);
    declareBinding(&term_store, shared.SET_KB_DATA, shared.BIND_TERM_STORE);
    declareBinding(&live_state, shared.SET_KB_DATA, shared.BIND_LIVE_STATE);

    declareBinding(&scratch_a, shared.SET_SCRATCH, shared.BIND_SCRATCH_A);
    declareBinding(&scratch_b, shared.SET_SCRATCH, shared.BIND_SCRATCH_B);
    declareBinding(&kv_cache, shared.SET_SCRATCH, shared.BIND_KV_CACHE);

    declareBinding(&params, shared.SET_CONTROL, shared.BIND_PARAMS);
    declareBinding(&status_buf, shared.SET_CONTROL, shared.BIND_STATUS);
    declareBinding(&result_counts, shared.SET_CONTROL, shared.BIND_RESULT_COUNTS);

    const op = params.data[shared.P_OP_CODE];

    switch (op) {
        0 => op_embedding_lookup(),
        1 => op_layer_norm(),
        2 => op_gemm(.layer_weights_qkv),
        3 => op_attention_scores(),
        4 => op_softmax_exact(),
        5 => op_attention_weighted_sum(),
        6 => op_gemm(.layer_weights_out),
        7 => op_mlp(),
        8 => op_gemm(.lm_head),
        9 => op_kv_cache_append(),
        10 => op_residual_add(),
        11 => op_fact_write_batch(),
        12 => op_fact_read_batch(),
        13 => op_fact_scan_by_tag(),
        14 => op_scoped_search(),
        15 => op_unify_candidates(),
        16 => op_rule_match_scan(),
        17 => op_rule_body_eval(),
        18 => op_rule_check_satisfied(),
        19 => op_builtin_unary(),
        20 => op_builtin_binary(),
        21 => op_builtin_reduction(),
        22 => op_builtin_sort(),
        23 => op_builtin_matmul(),
        24 => op_confidence_combine(),
        25 => op_confidence_chain(),
        26 => op_buffer_copy(),
        27 => op_buffer_fill(),
        else => {},
    }
}

fn workgroupBarrier() void {
    // OpControlBarrier Workgroup Workgroup AcquireRelease|WorkgroupMemory
    // Scope 2 = Workgroup, Semantics 264 = 0x108 = AcquireRelease(8) | WorkgroupMemory(256)
    asm volatile (
        \\OpControlBarrier $scope $scope $semantics
        :
        : [scope] "c" (@as(u32, 2)),
          [semantics] "c" (@as(u32, 264)),
    );
}

fn declareBinding(comptime ptr: anytype, comptime set: u32, comptime bind: u32) void {
    asm volatile (
        \\OpDecorate %ptr DescriptorSet $set
        \\OpDecorate %ptr Binding $bind
        :
        : [ptr] "" (ptr),
          [set] "c" (set),
          [bind] "c" (bind),
    );
}

// ============================================================
// Helpers
// ============================================================

fn gid() i32 {
    return @as(i32, gpu.global_invocation_id[0]);
}

fn lid() u32 {
    return gpu.local_invocation_id[0];
}

fn wid() u32 {
    return gpu.workgroup_id[0];
}

fn p(field: i32) i32 {
    return params.data[@intCast(field)];
}

fn q16_mul(a: i32, b: i32) i32 {
    return @intCast(@divTrunc(@as(i64, a) * @as(i64, b), shared.D));
}

fn q16_div(a: i32, b: i32) i32 {
    if (b == 0) return 0;
    return @intCast(@divTrunc(@as(i64, a) * shared.D, @as(i64, b)));
}

fn int_exp(x: i32) i32 {
    if (x >= 0) return shared.D;
    if (x < -720896) return 0;
    const abs_x: u32 = @intCast(-x);
    const segment: u32 = abs_x / @as(u32, @intCast(shared.D));
    const frac: i32 = @intCast(abs_x % @as(u32, @intCast(shared.D)));
    if (segment >= 10) return 0;
    const high = shared.exp_table[segment];
    const low = shared.exp_table[segment + 1];
    return high + @as(i32, @intCast(@divTrunc(@as(i64, low - high) * @as(i64, frac), shared.D)));
}

fn silu(x: i32) i32 {
    var sig: i32 = undefined;
    if (x < -262144) {
        sig = 0;
    } else if (x < -131072) {
        sig = @intCast(@divTrunc(@as(i64, x + 262144) * 4681, shared.D));
    } else if (x < 0) {
        sig = 4681 + @as(i32, @intCast(@divTrunc(@as(i64, x + 131072) * 28087, 131072)));
    } else if (x < 131072) {
        sig = 32768 + @as(i32, @intCast(@divTrunc(@as(i64, x) * 28087, 131072)));
    } else if (x < 262144) {
        sig = 60855 + @as(i32, @intCast(@divTrunc(@as(i64, x - 131072) * 4681, shared.D)));
    } else {
        sig = shared.D;
    }
    return q16_mul(x, sig);
}

fn clz32(x: i32) i32 {
    if (x == 0) return 32;
    var u: u32 = @bitCast(x);
    var n: i32 = 0;
    if ((u & 0xFFFF0000) == 0) {
        n += 16;
        u <<= 16;
    }
    if ((u & 0xFF000000) == 0) {
        n += 8;
        u <<= 8;
    }
    if ((u & 0xF0000000) == 0) {
        n += 4;
        u <<= 4;
    }
    if ((u & 0xC0000000) == 0) {
        n += 2;
        u <<= 2;
    }
    if ((u & 0x80000000) == 0) {
        n += 1;
    }
    return n;
}

fn popcount32(x: i32) i32 {
    var u: u32 = @bitCast(x);
    u = u - ((u >> 1) & 0x55555555);
    u = (u & 0x33333333) + ((u >> 2) & 0x33333333);
    u = (u + (u >> 4)) & 0x0F0F0F0F;
    return @intCast((u *% 0x01010101) >> 24);
}

fn gcd_iter(a_in: i32, b_in: i32) i32 {
    var a = if (a_in < 0) -a_in else a_in;
    var b = if (b_in < 0) -b_in else b_in;
    var i: i32 = 0;
    while (i < 64) : (i += 1) {
        if (b == 0) return a;
        const t = b;
        b = @mod(a, b);
        a = t;
    }
    return a;
}

// ============================================================
// GEMM source selector
// ============================================================

const GemmSource = enum { layer_weights_qkv, layer_weights_out, lm_head };

// ============================================================
// Op 0: embedding_lookup
// P: F0=n_tokens F1=d_model
// ============================================================

fn op_embedding_lookup() void {
    const idx = gid();
    const n_tokens = p(shared.P_FIELD_0);
    const d_model = p(shared.P_FIELD_1);
    const token_idx = @divTrunc(idx, d_model);
    const dim_idx = @mod(idx, d_model);
    if (token_idx >= n_tokens) return;
    const token_id = scratch_a.data[@intCast(token_idx)];
    scratch_b.data[@intCast(idx)] = embedding_table.data[@intCast(token_id * d_model + dim_idx)];
}

// ============================================================
// Op 1: layer_norm
// P: F0=n_tokens F1=d_model F2=layer_idx F3=norm_idx F4=epsilon_v
// ============================================================

fn op_layer_norm() void {
    const local = lid();
    const work = wid();
    const n_tokens = p(shared.P_FIELD_0);
    const d_model = p(shared.P_FIELD_1);
    const layer_idx = p(shared.P_FIELD_2);
    const norm_idx = p(shared.P_FIELD_3);

    if (@as(i32, @intCast(work)) >= n_tokens) return;

    const base: i32 = @as(i32, @intCast(work)) * d_model;

    // Sum of squares for RMSNorm
    var local_sum_sq: i64 = 0;
    var i: i32 = @intCast(local);
    while (i < d_model) : (i += 256) {
        const val: i64 = scratch_a.data[@intCast(base + i)];
        local_sum_sq += val * val;
    }
    s_i64[local] = local_sum_sq;
    workgroupBarrier();

    // Tree reduction
    var stride: u32 = 128;
    while (stride > 0) : (stride >>= 1) {
        if (local < stride) s_i64[local] += s_i64[local + stride];
        workgroupBarrier();
    }

    const mean_sq = @divTrunc(s_i64[0], @as(i64, d_model));
    var guess: i64 = if (mean_sq > 0) mean_sq else 1;
    guess = @divTrunc(guess + @divTrunc(mean_sq, guess), 2);
    guess = @divTrunc(guess + @divTrunc(mean_sq, guess), 2);
    guess = @divTrunc(guess + @divTrunc(mean_sq, guess), 2);
    guess = @divTrunc(guess + @divTrunc(mean_sq, guess), 2);

    const inv_rms: i32 = if (guess == 0) shared.D else @intCast(@divTrunc(@as(i64, shared.D) * shared.D, guess));

    if (local == 0) s_i32[0] = inv_rms;
    workgroupBarrier();
    const final_inv = s_i32[0];

    const ln_base = (layer_idx * 2 + norm_idx) * d_model;
    i = @intCast(local);
    while (i < d_model) : (i += 256) {
        const val: i64 = scratch_a.data[@intCast(base + i)];
        const normed = @divTrunc(val * @as(i64, final_inv), shared.D);
        const gamma: i64 = ln_params.data[@intCast(ln_base + i)];
        scratch_b.data[@intCast(base + i)] = @intCast(@divTrunc(normed * gamma, shared.D));
    }
}

// ============================================================
// Op 2/6/8: GEMM (unified for QKV, output proj, LM head)
// P: F0=n_tokens F1=d_model F2 F3 F4 F5 vary by GemmSource
// ============================================================

fn op_gemm(source: GemmSource) void {
    const idx = gid();
    const n_tokens = p(shared.P_FIELD_0);
    const d_model = p(shared.P_FIELD_1);

    var out_cols: i32 = undefined;
    var w_offset: i32 = undefined;

    switch (source) {
        .layer_weights_qkv => {
            out_cols = d_model * 3;
            w_offset = p(shared.P_FIELD_5);
        },
        .layer_weights_out => {
            out_cols = d_model;
            w_offset = p(shared.P_FIELD_3);
        },
        .lm_head => {
            out_cols = p(shared.P_FIELD_2); // vocab_size
            w_offset = 0;
        },
    }

    const row = @divTrunc(idx, out_cols);
    const col = @mod(idx, out_cols);
    if (row >= n_tokens) return;

    const in_base = row * d_model;
    var acc: i64 = 0;
    var k: i32 = 0;
    while (k < d_model) : (k += 1) {
        const a_val: i64 = scratch_a.data[@intCast(in_base + k)];
        const w_val: i64 = switch (source) {
            .layer_weights_qkv, .layer_weights_out => layer_weights.data[@intCast(w_offset + k * out_cols + col)],
            .lm_head => lm_head_weights.data[@intCast(k * out_cols + col)],
        };
        acc += a_val * w_val;
    }
    scratch_b.data[@intCast(row * out_cols + col)] = @intCast(@divTrunc(acc, shared.D));
}

// ============================================================
// Op 3: attention_scores
// P: F0=n_tokens F1=n_heads F2=d_head F3=seq_len F4=scale_v F5=causal
// ============================================================

fn op_attention_scores() void {
    const head_idx: i32 = @intCast(gpu.workgroup_id[0]);
    const query_pos: i32 = @intCast(gpu.workgroup_id[1]);
    const local = lid();
    const n_tokens = p(shared.P_FIELD_0);
    const n_heads = p(shared.P_FIELD_1);
    const d_head = p(shared.P_FIELD_2);
    const seq_len = p(shared.P_FIELD_3);
    const scale_v = p(shared.P_FIELD_4);
    const causal = p(shared.P_FIELD_5);

    if (head_idx >= n_heads or query_pos >= n_tokens) return;

    const q_base = query_pos * n_heads * d_head * 3 + head_idx * d_head;
    const out_base = (head_idx * n_tokens + query_pos) * seq_len;
    const actual_query = query_pos + (seq_len - n_tokens);

    var key_pos: i32 = @intCast(local);
    while (key_pos < seq_len) : (key_pos += 256) {
        if (causal != 0 and key_pos > actual_query) {
            scratch_a.data[@intCast(out_base + key_pos)] = -2147483647;
            continue;
        }
        const k_base = key_pos * n_heads * d_head * 2 + head_idx * d_head * 2;
        var dot: i64 = 0;
        var d: i32 = 0;
        while (d < d_head) : (d += 1) {
            dot += @as(i64, scratch_b.data[@intCast(q_base + d)]) *
                @as(i64, kv_cache.data[@intCast(k_base + d)]);
        }
        scratch_a.data[@intCast(out_base + key_pos)] = @intCast(@divTrunc(dot * @as(i64, scale_v), shared.D));
    }
}

// ============================================================
// Op 4: softmax_exact
// P: F0=row_length F1=n_rows F2=denominator
// ============================================================

fn op_softmax_exact() void {
    const local = lid();
    const work = wid();
    const row_len = p(shared.P_FIELD_0);
    const n_rows = p(shared.P_FIELD_1);
    const denom: i64 = p(shared.P_FIELD_2);

    if (@as(i32, @intCast(work)) >= n_rows) return;

    const base: i32 = @as(i32, @intCast(work)) * row_len;

    // Find max
    var local_max: i32 = -2147483647;
    var i: i32 = @intCast(local);
    while (i < row_len) : (i += 256) {
        const v = scratch_a.data[@intCast(base + i)];
        if (v > local_max) local_max = v;
    }
    s_i32[local] = local_max;
    workgroupBarrier();
    var stride: u32 = 128;
    while (stride > 0) : (stride >>= 1) {
        if (local < stride and s_i32[local + stride] > s_i32[local])
            s_i32[local] = s_i32[local + stride];
        workgroupBarrier();
    }
    const row_max = s_i32[0];

    // Compute exp, accumulate sum
    var local_sum: i64 = 0;
    i = @intCast(local);
    while (i < row_len) : (i += 256) {
        const e = int_exp(scratch_a.data[@intCast(base + i)] - row_max);
        scratch_a.data[@intCast(base + i)] = e;
        local_sum += e;
    }
    s_i64[local] = local_sum;
    workgroupBarrier();
    stride = 128;
    while (stride > 0) : (stride >>= 1) {
        if (local < stride) s_i64[local] += s_i64[local + stride];
        workgroupBarrier();
    }
    var total: i64 = s_i64[0];
    if (total == 0) total = 1;

    // Normalize + track max remainder
    var local_prob_sum: i64 = 0;
    var max_rem: i32 = 0;
    var max_rem_i: i32 = -1;
    i = @intCast(local);
    while (i < row_len) : (i += 256) {
        const e: i64 = scratch_a.data[@intCast(base + i)];
        const num = e * denom;
        const prob: i32 = @intCast(@divTrunc(num, total));
        const rem: i32 = @intCast(@mod(num, total));
        scratch_a.data[@intCast(base + i)] = prob;
        local_prob_sum += prob;
        if (rem > max_rem) {
            max_rem = rem;
            max_rem_i = i;
        }
    }

    // Reduce sum and find global max remainder holder
    s_i64[local] = local_prob_sum;
    s_i32[local] = max_rem;
    s_idx[local] = max_rem_i;
    workgroupBarrier();
    stride = 128;
    while (stride > 0) : (stride >>= 1) {
        if (local < stride) {
            s_i64[local] += s_i64[local + stride];
            if (s_i32[local + stride] > s_i32[local]) {
                s_i32[local] = s_i32[local + stride];
                s_idx[local] = s_idx[local + stride];
            }
        }
        workgroupBarrier();
    }

    // FRU: adjust so sum == D exactly
    if (local == 0 and s_idx[0] >= 0) {
        const deficit: i32 = @intCast(denom - s_i64[0]);
        scratch_a.data[@intCast(base + s_idx[0])] += deficit;
    }
}

// ============================================================
// Op 5: attention_weighted_sum
// P: F0=n_tokens F1=n_heads F2=d_head F3=seq_len
// ============================================================

fn op_attention_weighted_sum() void {
    const head_idx: i32 = @intCast(gpu.workgroup_id[0]);
    const query_pos: i32 = @intCast(gpu.workgroup_id[1]);
    const local = lid();
    const n_tokens = p(shared.P_FIELD_0);
    const n_heads = p(shared.P_FIELD_1);
    const d_head = p(shared.P_FIELD_2);
    const seq_len = p(shared.P_FIELD_3);

    if (head_idx >= n_heads or query_pos >= n_tokens) return;

    const prob_base = (head_idx * n_tokens + query_pos) * seq_len;
    const out_offset = n_heads * n_tokens * seq_len;
    const out_base = out_offset + (query_pos * n_heads + head_idx) * d_head;

    var d: i32 = @intCast(local);
    while (d < d_head) : (d += 256) {
        var acc: i64 = 0;
        var kp: i32 = 0;
        while (kp < seq_len) : (kp += 1) {
            const prob: i64 = scratch_a.data[@intCast(prob_base + kp)];
            const v_base = kp * n_heads * d_head * 2 + head_idx * d_head * 2 + d_head;
            acc += prob * @as(i64, kv_cache.data[@intCast(v_base + d)]);
        }
        scratch_a.data[@intCast(out_base + d)] = @intCast(@divTrunc(acc, shared.D));
    }
}

// ============================================================
// Op 7: MLP
// P: F0=n_tokens F1=d_model F2=mlp_dim F3=layer_idx F4=up_off F5=down_off F6=act
// ============================================================

fn op_mlp() void {
    const idx = gid();
    const n_tokens = p(shared.P_FIELD_0);
    const d_model = p(shared.P_FIELD_1);
    const mlp_dim = p(shared.P_FIELD_2);
    const up_off = p(shared.P_FIELD_4);
    const down_off = p(shared.P_FIELD_5);
    const act_type = p(shared.P_FIELD_6);

    const row = @divTrunc(idx, d_model);
    const col = @mod(idx, d_model);
    if (row >= n_tokens) return;

    const in_base = row * d_model;
    var acc: i64 = 0;
    var m: i32 = 0;
    while (m < mlp_dim) : (m += 1) {
        var up_acc: i64 = 0;
        var k: i32 = 0;
        while (k < d_model) : (k += 1) {
            up_acc += @as(i64, scratch_a.data[@intCast(in_base + k)]) *
                @as(i64, layer_weights.data[@intCast(up_off + k * mlp_dim + m)]);
        }
        var up_val: i32 = @intCast(@divTrunc(up_acc, shared.D));
        if (act_type == 0) {
            up_val = silu(up_val);
        } else if (act_type == 2) {
            up_val = if (up_val > 0) up_val else 0;
        }
        acc += @as(i64, up_val) *
            @as(i64, layer_weights.data[@intCast(down_off + m * d_model + col)]);
    }
    scratch_b.data[@intCast(row * d_model + col)] = @intCast(@divTrunc(acc, shared.D));
}

// ============================================================
// Op 9: kv_cache_append
// P: F0=n_new F1=n_heads F2=d_head F3=layer_idx F4=start_pos F5=max_seq
// ============================================================

fn op_kv_cache_append() void {
    const idx = gid();
    const n_new = p(shared.P_FIELD_0);
    const n_heads = p(shared.P_FIELD_1);
    const d_head = p(shared.P_FIELD_2);
    const start_pos = p(shared.P_FIELD_4);
    const max_seq = p(shared.P_FIELD_5);
    const total = n_new * n_heads * d_head * 2;
    if (idx >= total) return;

    var rem = idx;
    const kv_sel = @mod(rem, 2);
    rem = @divTrunc(rem, 2);
    const dim = @mod(rem, d_head);
    rem = @divTrunc(rem, d_head);
    const head = @mod(rem, n_heads);
    rem = @divTrunc(rem, n_heads);
    const token = rem;
    const pos = start_pos + token;
    if (pos >= max_seq) return;

    const src = token * 3 * n_heads * d_head + (kv_sel + 1) * n_heads * d_head + head * d_head + dim;
    const dst = pos * n_heads * d_head * 2 + head * d_head * 2 + kv_sel * d_head + dim;
    kv_cache.data[@intCast(dst)] = scratch_b.data[@intCast(src)];
}

// ============================================================
// Op 10: residual_add
// P: F0=n_elements
// ============================================================

fn op_residual_add() void {
    const idx = gid();
    if (idx >= p(shared.P_FIELD_0)) return;
    scratch_a.data[@intCast(idx)] += scratch_b.data[@intCast(idx)];
}

// ============================================================
// Op 11: fact_write_batch
// P: F0=n_facts F1=base_offset F2=capacity
// ============================================================

fn op_fact_write_batch() void {
    const idx = gid();
    if (idx >= p(shared.P_FIELD_0)) return;
    const slot = scratch_b.data[@intCast(idx)];
    const target = p(shared.P_FIELD_1) + slot;
    if (target < 0 or target >= p(shared.P_FIELD_2)) {
        status_buf.data[@intCast(idx)] = 201;
        return;
    }
    const src = idx * shared.FACT_INTS;
    const dst = target * shared.FACT_INTS;
    var i: i32 = 0;
    while (i < shared.FACT_INTS) : (i += 1) {
        fact_store.data[@intCast(dst + i)] = scratch_a.data[@intCast(src + i)];
    }
    status_buf.data[@intCast(idx)] = 0;
}

// ============================================================
// Op 12: fact_read_batch
// P: F0=n_reads
// ============================================================

fn op_fact_read_batch() void {
    const idx = gid();
    if (idx >= p(shared.P_FIELD_0)) return;
    const offset = scratch_a.data[@intCast(idx)];
    const src = offset * shared.FACT_INTS;
    const dst = idx * shared.FACT_INTS;
    var i: i32 = 0;
    while (i < shared.FACT_INTS) : (i += 1) {
        scratch_b.data[@intCast(dst + i)] = fact_store.data[@intCast(src + i)];
    }
}

// ============================================================
// Op 13: fact_scan_by_tag
// P: F0=base_offset F1=scan_length F2=target_tag F3=max_results
// ============================================================

fn op_fact_scan_by_tag() void {
    const idx = gid();
    if (idx >= p(shared.P_FIELD_1)) return;
    const fact_base = (p(shared.P_FIELD_0) + idx) * shared.FACT_INTS;
    const tag = fact_store.data[@intCast(fact_base + shared.FACT_TAG)];
    if (tag == p(shared.P_FIELD_2)) {
        const slot = @atomicRmw(i32, &result_counts.data[0], .Add, 1, .seq_cst);
        if (slot < p(shared.P_FIELD_3)) {
            scratch_a.data[@intCast(slot)] = idx;
        }
    }
}

// ============================================================
// Op 14: scoped_search
// P: F0=n_chain F1=total_facts F2=target_tag F3=max_results
// ============================================================

fn op_scoped_search() void {
    const idx = gid();
    if (idx >= p(shared.P_FIELD_1)) return;
    const abs_offset = scratch_b.data[@intCast(idx)];
    const fact_base = abs_offset * shared.FACT_INTS;
    const tag = fact_store.data[@intCast(fact_base + shared.FACT_TAG)];
    if (tag == p(shared.P_FIELD_2)) {
        const slot = @atomicRmw(i32, &result_counts.data[0], .Add, 1, .seq_cst);
        if (slot < p(shared.P_FIELD_3)) {
            scratch_a.data[@intCast(slot)] = abs_offset;
        }
    }
}

// ============================================================
// Op 15: unify_candidates
// P: F0=n_cand F1=q_type F2=q_atom F3=q_int F4=q_vdr_v F5=q_vdr_r0
//    F6=q_func F7=q_argc F8=q_argoff F9=max_bind
// ============================================================

fn op_unify_candidates() void {
    const idx = gid();
    const n_cand = p(shared.P_FIELD_0);
    if (idx >= n_cand) return;

    const abs_offset = scratch_a.data[@intCast(idx)];
    const fb = abs_offset * shared.FACT_INTS;
    const fact_tag = fact_store.data[@intCast(fb + shared.FACT_TAG)];
    if (fact_tag == @intFromEnum(shared.FactTag.empty)) {
        status_buf.data[@intCast(idx)] = 0;
        return;
    }

    const fact_v = fact_store.data[@intCast(fb + shared.FACT_VALUE_V)];
    const fact_r0 = fact_store.data[@intCast(fb + shared.FACT_VALUE_R0)] & 0xFFFF;
    const q_type = p(shared.P_FIELD_1);
    var matched = false;

    if (q_type == @intFromEnum(shared.TermType.variable)) {
        matched = true;
        const bind_base = idx * p(shared.P_FIELD_9) * shared.BINDING_INTS;
        scratch_b.data[@intCast(bind_base)] = p(shared.P_FIELD_2);
        scratch_b.data[@intCast(bind_base + 1)] = abs_offset;
    } else if (q_type == @intFromEnum(shared.TermType.atom)) {
        matched = (fact_v == p(shared.P_FIELD_2));
    } else if (q_type == @intFromEnum(shared.TermType.integer)) {
        matched = (fact_v == p(shared.P_FIELD_3));
    } else if (q_type == @intFromEnum(shared.TermType.vdr)) {
        matched = (fact_v == p(shared.P_FIELD_4)) and
            (fact_r0 == (p(shared.P_FIELD_5) & 0xFFFF));
    } else if (q_type == @intFromEnum(shared.TermType.compound)) {
        matched = (fact_v == p(shared.P_FIELD_6));
    }

    if (matched) {
        status_buf.data[@intCast(idx)] = 1;
        _ = @atomicRmw(i32, &result_counts.data[0], .Add, 1, .seq_cst);
    } else {
        status_buf.data[@intCast(idx)] = 0;
    }
}

// ============================================================
// Op 16: rule_match_scan
// P: F0=n_rules F1=rules_base F2=q_type F3=q_atom F4=q_func F5=q_argc F6=max_matches
// ============================================================

fn op_rule_match_scan() void {
    const idx = gid();
    if (idx >= p(shared.P_FIELD_0)) return;

    const rb = (p(shared.P_FIELD_1) + idx) * shared.RULE_INTS;
    const rule_id = rule_store.data[@intCast(rb + shared.RULE_ID)];
    if (rule_id == -1) {
        status_buf.data[@intCast(idx)] = 0;
        return;
    }

    const head_off = rule_store.data[@intCast(rb + shared.RULE_HEAD)];
    const tb = head_off * shared.TERM_INTS;
    const head_type = term_store.data[@intCast(tb + shared.TERM_TYPE)] & 0xFF;
    const head_primary = term_store.data[@intCast(tb + shared.TERM_PRIMARY)];
    const head_aux = term_store.data[@intCast(tb + shared.TERM_AUX)];
    const q_type = p(shared.P_FIELD_2);

    var matched = false;
    if (q_type == @intFromEnum(shared.TermType.variable) or head_type == @intFromEnum(shared.TermType.variable)) {
        matched = true;
    } else if (q_type == @intFromEnum(shared.TermType.atom) and head_type == @intFromEnum(shared.TermType.atom)) {
        matched = (head_primary == p(shared.P_FIELD_3));
    } else if (q_type == @intFromEnum(shared.TermType.compound) and head_type == @intFromEnum(shared.TermType.compound)) {
        matched = (head_primary == p(shared.P_FIELD_4)) and (head_aux == p(shared.P_FIELD_5));
    }

    if (matched) {
        const slot = @atomicRmw(i32, &result_counts.data[0], .Add, 1, .seq_cst);
        if (slot < p(shared.P_FIELD_6)) scratch_a.data[@intCast(slot)] = rule_id;
        status_buf.data[@intCast(idx)] = 1;
    } else {
        status_buf.data[@intCast(idx)] = 0;
    }
}

// ============================================================
// Op 17: rule_body_eval
// P: F0=n_matched F1=max_body F2=facts_base F3=facts_count
// ============================================================

fn op_rule_body_eval() void {
    const idx = gid();
    const n_matched = p(shared.P_FIELD_0);
    const max_body = p(shared.P_FIELD_1);
    const facts_base = p(shared.P_FIELD_2);
    const facts_count = p(shared.P_FIELD_3);
    const total = n_matched * max_body;
    if (idx >= total) return;

    const rule_idx = @divTrunc(idx, max_body);
    const body_idx = @mod(idx, max_body);
    if (rule_idx >= n_matched) {
        scratch_b.data[@intCast(idx)] = 1;
        return;
    }

    const rule_id = scratch_a.data[@intCast(rule_idx)];
    const rb = rule_id * shared.RULE_INTS;
    const body_count = rule_store.data[@intCast(rb + shared.RULE_BODY_COUNT)] & 0xFFFF;
    if (body_idx >= body_count) {
        scratch_b.data[@intCast(idx)] = 1;
        return;
    }

    const body_off = rule_store.data[@intCast(rb + shared.RULE_BODY_OFF)];
    const tb = (body_off + body_idx) * shared.TERM_INTS;
    const t_type = term_store.data[@intCast(tb + shared.TERM_TYPE)] & 0xFF;
    const t_primary = term_store.data[@intCast(tb + shared.TERM_PRIMARY)];
    const t_vdr_v = term_store.data[@intCast(tb + shared.TERM_VDR_V)];

    if (t_type == @intFromEnum(shared.TermType.variable)) {
        scratch_b.data[@intCast(idx)] = 1;
        return;
    }

    var found: i32 = 0;
    const scan_limit = if (facts_count < 4096) facts_count else 4096;
    var fi: i32 = 0;
    while (fi < scan_limit) : (fi += 1) {
        const fb = (facts_base + fi) * shared.FACT_INTS;
        const f_tag = fact_store.data[@intCast(fb + shared.FACT_TAG)];
        if (f_tag == @intFromEnum(shared.FactTag.empty)) continue;
        const f_v = fact_store.data[@intCast(fb + shared.FACT_VALUE_V)];

        if (t_type == @intFromEnum(shared.TermType.atom) and f_v == t_primary) {
            found = 1;
            break;
        }
        if (t_type == @intFromEnum(shared.TermType.integer) and f_v == t_primary) {
            found = 1;
            break;
        }
        if (t_type == @intFromEnum(shared.TermType.vdr) and f_v == t_vdr_v) {
            found = 1;
            break;
        }
        if (t_type == @intFromEnum(shared.TermType.compound) and f_v == t_primary) {
            found = 1;
            break;
        }
    }
    scratch_b.data[@intCast(idx)] = found;
}

// ============================================================
// Op 18: rule_check_satisfied
// P: F0=n_matched F1=max_body F2=max_fires
// ============================================================

fn op_rule_check_satisfied() void {
    const idx = gid();
    const n_matched = p(shared.P_FIELD_0);
    const max_body = p(shared.P_FIELD_1);
    if (idx >= n_matched) return;

    const rule_id = scratch_a.data[@intCast(idx)];
    const rb = rule_id * shared.RULE_INTS;
    const body_count = rule_store.data[@intCast(rb + shared.RULE_BODY_COUNT)] & 0xFFFF;

    var all: bool = true;
    const eval_base = idx * max_body;
    var bi: i32 = 0;
    while (bi < body_count and bi < max_body) : (bi += 1) {
        if (scratch_b.data[@intCast(eval_base + bi)] == 0) {
            all = false;
            break;
        }
    }

    if (all) {
        const slot = @atomicRmw(i32, &result_counts.data[1], .Add, 1, .seq_cst);
        if (slot < p(shared.P_FIELD_2)) {
            scratch_a.data[@intCast(n_matched + slot)] = rule_id;
        }
    }
}

// ============================================================
// Op 19: builtin_unary
// P: F0=n_elements F1=sub_op F2=input_off F3=output_off
// ============================================================

fn op_builtin_unary() void {
    const idx = gid();
    if (idx >= p(shared.P_FIELD_0)) return;
    const val = scratch_a.data[@intCast(p(shared.P_FIELD_2) + idx)];
    const sub = p(shared.P_FIELD_1);
    const result = switch (sub) {
        0 => if (val < 0) -val else val,
        1 => -val,
        2 => if (val > 0) shared.D else if (val < 0) -shared.D else 0,
        3 => ~val,
        4 => clz32(val),
        5 => @as(i32, 32) - clz32(val & -val) - 1, // ctz via clz of lowest set bit
        6 => popcount32(val),
        7 => @as(i32, if (val == 0) 1 else 0),
        8 => @as(i32, if (val > 0) 1 else 0),
        9 => @as(i32, if (val < 0) 1 else 0),
        10 => q16_mul(val, val),
        11 => val * 2,
        12 => @divTrunc(val, 2),
        else => val,
    };
    scratch_b.data[@intCast(p(shared.P_FIELD_3) + idx)] = result;
}

// ============================================================
// Op 20: builtin_binary
// P: F0=n_elements F1=sub_op F2=in_a_off F3=in_b_off F4=output_off
// ============================================================

fn op_builtin_binary() void {
    const idx = gid();
    if (idx >= p(shared.P_FIELD_0)) return;
    const a = scratch_a.data[@intCast(p(shared.P_FIELD_2) + idx)];
    const b = scratch_a.data[@intCast(p(shared.P_FIELD_3) + idx)];
    const sub = p(shared.P_FIELD_1);
    const result = switch (sub) {
        0 => a + b,
        1 => a - b,
        2 => q16_mul(a, b),
        3 => q16_div(a, b),
        4 => if (b != 0) @mod(a, b) else 0,
        5 => if (a < b) a else b,
        6 => if (a > b) a else b,
        7 => gcd_iter(a, b),
        8 => blk: {
            const g = gcd_iter(a, b);
            break :blk if (g != 0) @as(i32, @intCast(@divTrunc(@as(i64, a), g) * b)) else 0;
        },
        9 => a & b,
        10 => a | b,
        11 => a ^ b,
        12 => if (b >= 0 and b < 32) a << @intCast(b) else 0,
        13 => if (b >= 0 and b < 32) a >> @intCast(b) else 0,
        14 => if (a < b) @as(i32, -1) else if (a > b) @as(i32, 1) else 0,
        else => a,
    };
    scratch_b.data[@intCast(p(shared.P_FIELD_4) + idx)] = result;
}

// ============================================================
// Op 21: builtin_reduction
// P: F0=n_elements F1=sub_op F2=input_off
// ============================================================

fn op_builtin_reduction() void {
    const local = lid();
    const n = p(shared.P_FIELD_0);
    const sub = p(shared.P_FIELD_1);
    const off = p(shared.P_FIELD_2);

    var val: i64 = switch (sub) {
        0, 4, 5, 6 => 0,
        1 => @as(i64, shared.D),
        2, 9 => 2147483647,
        3, 10 => -2147483647,
        7 => 0,
        8 => 1,
        else => 0,
    };
    var best_idx: i32 = -1;

    var i: i32 = @intCast(local);
    while (i < n) : (i += 256) {
        const v: i64 = scratch_a.data[@intCast(off + i)];
        switch (sub) {
            0, 4 => val += v,
            1 => val = @divTrunc(val * v, shared.D),
            2, 9 => {
                if (v < val) {
                    val = v;
                    best_idx = i;
                }
            },
            3, 10 => {
                if (v > val) {
                    val = v;
                    best_idx = i;
                }
            },
            5 => val += v * v,
            6 => {
                if (v != 0) val += 1;
            },
            7 => {
                if (v > 0) val = 1;
            },
            8 => {
                if (v <= 0) val = 0;
            },
            else => {},
        }
    }

    s_i64[local] = val;
    s_idx[local] = best_idx;
    workgroupBarrier();

    var stride: u32 = 128;
    while (stride > 0) : (stride >>= 1) {
        if (local < stride) {
            switch (sub) {
                0, 4, 5, 6 => s_i64[local] += s_i64[local + stride],
                1 => s_i64[local] = @divTrunc(s_i64[local] * s_i64[local + stride], shared.D),
                2, 9 => {
                    if (s_i64[local + stride] < s_i64[local]) {
                        s_i64[local] = s_i64[local + stride];
                        s_idx[local] = s_idx[local + stride];
                    }
                },
                3, 10 => {
                    if (s_i64[local + stride] > s_i64[local]) {
                        s_i64[local] = s_i64[local + stride];
                        s_idx[local] = s_idx[local + stride];
                    }
                },
                7 => {
                    if (s_i64[local + stride] != 0) s_i64[local] = 1;
                },
                8 => {
                    if (s_i64[local + stride] == 0) s_i64[local] = 0;
                },
                else => {},
            }
        }
        workgroupBarrier();
    }

    if (local == 0) {
        var final_val = s_i64[0];
        if (sub == 4 and n > 0) final_val = @divTrunc(final_val, n);
        if (sub == 5 and n > 0) final_val = @divTrunc(final_val, n);
        scratch_b.data[0] = @intCast(final_val);
        scratch_b.data[1] = s_idx[0];
    }
}

// ============================================================
// Op 22: builtin_sort (bitonic, single workgroup, max 256)
// P: F0=n_elements F1=ascending F2=input_off F3=output_off
// ============================================================

fn op_builtin_sort() void {
    const local = lid();
    const n = p(shared.P_FIELD_0);
    const ascending = p(shared.P_FIELD_1);

    const sentinel: i32 = if (ascending != 0) 2147483647 else -2147483647;
    s_i32[local] = if (@as(i32, @intCast(local)) < n) scratch_a.data[@intCast(p(shared.P_FIELD_2) + @as(i32, @intCast(local)))] else sentinel;
    workgroupBarrier();

    var stage: u32 = 2;
    while (stage <= 256) : (stage <<= 1) {
        var step: u32 = stage >> 1;
        while (step > 0) : (step >>= 1) {
            const partner = local ^ step;
            if (partner > local) {
                const dir = ((@divTrunc(local, stage)) % 2 == 0) != (ascending == 0);
                const swap = if (dir) (s_i32[local] > s_i32[partner]) else (s_i32[local] < s_i32[partner]);
                if (swap) {
                    const tmp = s_i32[local];
                    s_i32[local] = s_i32[partner];
                    s_i32[partner] = tmp;
                }
            }
            workgroupBarrier();
        }
    }

    if (@as(i32, @intCast(local)) < n) {
        scratch_b.data[@intCast(p(shared.P_FIELD_3) + @as(i32, @intCast(local)))] = s_i32[local];
    }
}

// ============================================================
// Op 23: builtin_matmul
// P: F0=m F1=n F2=k F3=a_off F4=b_off F5=c_off
// ============================================================

fn op_builtin_matmul() void {
    const idx = gid();
    const m = p(shared.P_FIELD_0);
    const n = p(shared.P_FIELD_1);
    const k = p(shared.P_FIELD_2);
    if (idx >= m * n) return;

    const row = @divTrunc(idx, n);
    const col = @mod(idx, n);
    const a_off = p(shared.P_FIELD_3);
    const b_off = p(shared.P_FIELD_4);

    var acc: i64 = 0;
    var i: i32 = 0;
    while (i < k) : (i += 1) {
        acc += @as(i64, scratch_a.data[@intCast(a_off + row * k + i)]) *
            @as(i64, scratch_a.data[@intCast(b_off + i * n + col)]);
    }
    scratch_b.data[@intCast(p(shared.P_FIELD_5) + row * n + col)] = @intCast(@divTrunc(acc, shared.D));
}

// ============================================================
// Op 24: confidence_combine
// P: F0=n_sources F1=mode F2=penalty_v F3=input_off
// ============================================================

fn op_confidence_combine() void {
    const local = lid();
    const n = p(shared.P_FIELD_0);
    const mode = p(shared.P_FIELD_1);
    const off = p(shared.P_FIELD_3);

    var prod: i64 = @as(i64, shared.D);
    var i: i32 = @intCast(local);
    while (i < n) : (i += 256) {
        const c: i64 = scratch_a.data[@intCast(off + i)];
        prod = @divTrunc(prod * (@as(i64, shared.D) - c), shared.D);
    }
    s_i64[local] = prod;
    workgroupBarrier();

    var stride: u32 = 128;
    while (stride > 0) : (stride >>= 1) {
        if (local < stride) {
            s_i64[local] = @divTrunc(s_i64[local] * s_i64[local + stride], shared.D);
        }
        workgroupBarrier();
    }

    if (local == 0) {
        var result: i64 = @as(i64, shared.D) - s_i64[0];
        if (mode == 1) {
            const penalty: i64 = p(shared.P_FIELD_2);
            const pairs = @divTrunc(n * (n - 1), 2);
            var pi: i32 = 0;
            while (pi < pairs and pi < 100) : (pi += 1) {
                result = @divTrunc(result * penalty, shared.D);
            }
        }
        if (result < 0) result = 0;
        if (result > shared.D) result = shared.D;
        scratch_b.data[0] = @intCast(result);
    }
}

// ============================================================
// Op 25: confidence_chain
// P: F0=n_links F1=per_link_v
// ============================================================

fn op_confidence_chain() void {
    if (lid() != 0) return;
    const n = p(shared.P_FIELD_0);
    const per: i64 = p(shared.P_FIELD_1);
    var result: i64 = per;
    const links = if (n < 100) n else 100;
    var i: i32 = 1;
    while (i < links) : (i += 1) {
        result = @divTrunc(result * per, shared.D);
    }
    if (result < 0) result = 0;
    if (result > shared.D) result = shared.D;
    scratch_b.data[0] = @intCast(result);
}

// ============================================================
// Op 26: buffer_copy
// P: F0=src_off F1=dst_off F2=n_elements F3=elem_size
// ============================================================

fn op_buffer_copy() void {
    const idx = gid();
    if (idx >= p(shared.P_FIELD_2)) return;
    const es = p(shared.P_FIELD_3);
    const src = p(shared.P_FIELD_0) + idx * es;
    const dst = p(shared.P_FIELD_1) + idx * es;
    var i: i32 = 0;
    while (i < es) : (i += 1) {
        scratch_b.data[@intCast(dst + i)] = scratch_a.data[@intCast(src + i)];
    }
}

// ============================================================
// Op 27: buffer_fill
// P: F0=dst_off F1=n_elements F2=fill_value F3=elem_size
// ============================================================

fn op_buffer_fill() void {
    const idx = gid();
    if (idx >= p(shared.P_FIELD_1)) return;
    const es = p(shared.P_FIELD_3);
    const dst = p(shared.P_FIELD_0) + idx * es;
    const fill = p(shared.P_FIELD_2);
    var i: i32 = 0;
    while (i < es) : (i += 1) {
        scratch_b.data[@intCast(dst + i)] = fill;
    }
}
