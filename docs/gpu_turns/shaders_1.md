```glsl
// ============================================================
// embedding_lookup.comp
// Maps token IDs to embedding vectors.
// Input: scratch_a = token_ids [n_tokens] i32
// Output: scratch_b = embedded [n_tokens * d_model] i32 (Q16 .v)
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer EmbeddingTable { int data[]; } embedding;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;

layout(set = 3, binding = 0) uniform Params {
    int n_tokens;
    int d_model;
    int _pad0;
    int _pad1;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    int token_idx = int(gid) / params.d_model;
    int dim_idx = int(gid) % params.d_model;
    if (token_idx >= params.n_tokens) return;
    int token_id = scratch_a.data[token_idx];
    scratch_b.data[token_idx * params.d_model + dim_idx] =
        embedding.data[token_id * params.d_model + dim_idx];
}
```

```glsl
// ============================================================
// layer_norm.comp
// RMSNorm: one workgroup per token position.
// Uses shared memory for parallel mean/variance reduction.
// Input: scratch_a [n_tokens * d_model] Q16 .v
// Output: scratch_b [n_tokens * d_model] Q16 .v
// Reads gamma/beta from layer_norm_params buffer.
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 3) buffer LayerNormParams { int data[]; } ln_params;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;

layout(set = 3, binding = 0) uniform Params {
    int n_tokens;
    int d_model;
    int layer_idx;
    int norm_idx;
    int epsilon_v;
    int _pad0;
    int _pad1;
    int _pad2;
} params;

shared int s_data[256];
shared int64_t s_accum[256];

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint wid = gl_WorkGroupID.x;
    if (int(wid) >= params.n_tokens) return;

    int base = int(wid) * params.d_model;

    // Phase 1: compute sum of squares for RMSNorm
    int64_t local_sum_sq = 0;
    for (int i = int(lid); i < params.d_model; i += 256) {
        int val = scratch_a.data[base + i];
        local_sum_sq += int64_t(val) * int64_t(val);
    }
    s_accum[lid] = local_sum_sq;
    barrier();

    // Tree reduction for sum of squares
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) {
            s_accum[lid] += s_accum[lid + stride];
        }
        barrier();
    }

    // s_accum[0] = sum of squares
    // RMS = sqrt(sum_sq / d_model)
    // We need 1/RMS = isqrt(sum_sq / d_model)
    // Work in Q16: sum_sq is in Q16^2 units (D^2 scale)

    int64_t mean_sq = s_accum[0] / int64_t(params.d_model);

    // Newton-Raphson integer inverse sqrt: 4 iterations
    // Start with rough estimate
    int64_t x = mean_sq;
    if (x <= 0) x = 1;
    int64_t guess = x;
    // sqrt via Newton
    guess = (guess + x / guess) / 2;
    guess = (guess + x / guess) / 2;
    guess = (guess + x / guess) / 2;
    guess = (guess + x / guess) / 2;
    // inv_rms = D / sqrt(mean_sq) = 65536 / guess
    int inv_rms;
    if (guess == 0) {
        inv_rms = 65536;
    } else {
        inv_rms = int(int64_t(65536) * int64_t(65536) / guess);
    }

    // Broadcast inv_rms via shared memory
    if (lid == 0) s_data[0] = inv_rms;
    barrier();
    inv_rms = s_data[0];

    // Phase 2: normalize and apply gamma
    // gamma/beta layout: layer_idx * 2 * d_model + norm_idx * d_model
    int ln_base = (params.layer_idx * 2 + params.norm_idx) * params.d_model;

    for (int i = int(lid); i < params.d_model; i += 256) {
        int val = scratch_a.data[base + i];
        // normalized = val * inv_rms / D
        int64_t normed = int64_t(val) * int64_t(inv_rms) / int64_t(65536);
        // apply gamma: normed * gamma / D
        int gamma = ln_params.data[ln_base + i];
        int64_t scaled = normed * int64_t(gamma) / int64_t(65536);
        scratch_b.data[base + i] = int(scaled);
    }
}
```

```glsl
// ============================================================
// qkv_project.comp
// Tiled integer GEMM: [n_tokens × d_model] × [d_model × 3*d_model]
// Output: [n_tokens × 3 * d_model] in scratch_b
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 1) buffer LayerWeights { int data[]; } weights;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;

layout(set = 3, binding = 0) uniform Params {
    int n_tokens;
    int d_model;
    int n_heads;
    int d_head;
    int layer_idx;
    int weights_offset;
    int _pad0;
    int _pad1;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    int out_cols = params.d_model * 3;
    int row = int(gid) / out_cols;
    int col = int(gid) % out_cols;
    if (row >= params.n_tokens) return;

    // Dot product: input[row, :] · weights[:, col]
    int64_t acc = 0;
    int in_base = row * params.d_model;
    int w_base = params.weights_offset + col;

    for (int k = 0; k < params.d_model; k++) {
        acc += int64_t(scratch_a.data[in_base + k]) *
               int64_t(weights.data[w_base + k * out_cols]);
    }

    // Q16: result = acc / D
    scratch_b.data[row * out_cols + col] = int(acc / int64_t(65536));
}
```

```glsl
// ============================================================
// attention_scores.comp
// Q × K^T for one (head, query_position) per workgroup.
// Output: score matrix in scratch_a.
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;
layout(set = 2, binding = 2) buffer KvCache  { int data[]; } kv_cache;

layout(set = 3, binding = 0) uniform Params {
    int n_tokens;
    int n_heads;
    int d_head;
    int seq_len;
    int scale_v;
    int causal_mask;
    int _pad0;
    int _pad1;
} params;

void main() {
    int head_idx = int(gl_WorkGroupID.x);
    int query_pos = int(gl_WorkGroupID.y);
    uint lid = gl_LocalInvocationID.x;

    if (head_idx >= params.n_heads || query_pos >= params.n_tokens) return;

    // Q vector for this head and position is in scratch_b
    // Layout: [n_tokens × 3 × n_heads × d_head]
    // Q starts at offset 0 within the 3*d_model block
    int q_base = query_pos * params.n_heads * params.d_head * 3 +
                 head_idx * params.d_head;

    // Output offset in scratch_a: [n_heads × n_tokens × seq_len]
    int out_base = (head_idx * params.n_tokens + query_pos) * params.seq_len;

    // Each invocation handles a subset of key positions
    for (int key_pos = int(lid); key_pos < params.seq_len; key_pos += 256) {
        // Causal mask
        int actual_query = query_pos + (params.seq_len - params.n_tokens);
        if (params.causal_mask != 0 && key_pos > actual_query) {
            scratch_a.data[out_base + key_pos] = -2147483647; // INT_MIN+1
            continue;
        }

        // K vector from KV cache
        // KV cache layout: [layer × max_seq × n_heads × d_head × 2]
        // K is kv_select=0
        int k_base = key_pos * params.n_heads * params.d_head * 2 +
                     head_idx * params.d_head * 2;

        // Dot product Q · K
        int64_t dot = 0;
        for (int d = 0; d < params.d_head; d++) {
            dot += int64_t(scratch_b.data[q_base + d]) *
                   int64_t(kv_cache.data[k_base + d]);
        }

        // Apply scale: score = dot * scale_v / D
        int score = int((dot * int64_t(params.scale_v)) / int64_t(65536));
        scratch_a.data[out_base + key_pos] = score;
    }
}
```

```glsl
// ============================================================
// softmax_exact.comp
// Exact integer softmax. Output sums to D (65536). Exactly.
// One workgroup per row. Uses FRU remainder redistribution.
// Input/output: scratch_a (in-place).
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;

layout(set = 3, binding = 0) uniform Params {
    int row_length;
    int n_rows;
    int denominator; // D = 65536
    int _pad;
} params;

shared int s_data[256];
shared int64_t s_accum[256];
shared int s_max;
shared int64_t s_sum;
shared int s_remainder;
shared int s_max_rem_idx;

// Integer exp approximation: piecewise linear
// Input: x (shifted so max=0, all values <= 0)
// Output: positive integer proportional to exp(x)
// Uses 5-segment piecewise approximation
int int_exp(int x) {
    // Clamp very negative values to 0
    if (x < -720896) return 0; // ~-11 * D
    // Shift to positive range for table
    // exp(x/D) approximated in integer
    // For x=0: return D (65536)
    // For x=-D: return ~24109 (e^-1 * D)
    // Linear interpolation between breakpoints
    if (x >= 0) return params.denominator;
    int abs_x = -x;
    int segment = abs_x / 65536;
    int frac = abs_x % 65536;
    // Precomputed: exp(-k) * D for k=0..10
    // 65536, 24109, 8874, 3263, 1201, 442, 162, 60, 22, 8, 3
    int table[11] = int[11](65536, 24109, 8874, 3263, 1201, 442, 162, 60, 22, 8, 3);
    if (segment >= 10) return 0;
    int high = table[segment];
    int low = table[segment + 1];
    // Linear interp: high + (low - high) * frac / D
    return high + int(int64_t(low - high) * int64_t(frac) / int64_t(65536));
}

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint wid = gl_WorkGroupID.x;
    if (int(wid) >= params.n_rows) return;

    int base = int(wid) * params.row_length;

    // Phase 1: find max (parallel reduction)
    int local_max = -2147483647;
    for (int i = int(lid); i < params.row_length; i += 256) {
        int val = scratch_a.data[base + i];
        if (val > local_max) local_max = val;
    }
    s_data[lid] = local_max;
    barrier();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (s_data[lid + stride] > s_data[lid])
                s_data[lid] = s_data[lid + stride];
        }
        barrier();
    }
    if (lid == 0) s_max = s_data[0];
    barrier();
    int row_max = s_max;

    // Phase 2: compute exp(x - max) for each element, accumulate sum
    int64_t local_exp_sum = 0;
    for (int i = int(lid); i < params.row_length; i += 256) {
        int shifted = scratch_a.data[base + i] - row_max;
        int e = int_exp(shifted);
        scratch_a.data[base + i] = e; // temporarily store exp values
        local_exp_sum += int64_t(e);
    }
    s_accum[lid] = local_exp_sum;
    barrier();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) {
            s_accum[lid] += s_accum[lid + stride];
        }
        barrier();
    }
    if (lid == 0) {
        s_sum = s_accum[0];
        if (s_sum == 0) s_sum = 1; // prevent division by zero
    }
    barrier();
    int64_t total_sum = s_sum;

    // Phase 3: normalize — prob[i] = exp[i] * D / sum
    // Track remainder for FRU redistribution
    int local_max_rem = 0;
    int local_max_rem_i = -1;
    int64_t local_prob_sum = 0;

    for (int i = int(lid); i < params.row_length; i += 256) {
        int e = scratch_a.data[base + i];
        int64_t numerator = int64_t(e) * int64_t(params.denominator);
        int prob = int(numerator / total_sum);
        int rem = int(numerator % total_sum);
        scratch_a.data[base + i] = prob;
        local_prob_sum += int64_t(prob);
        if (rem > local_max_rem) {
            local_max_rem = rem;
            local_max_rem_i = i;
        }
    }

    // Reduce to get actual sum and find global max remainder
    s_accum[lid] = local_prob_sum;
    s_data[lid] = local_max_rem;
    barrier();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) {
            s_accum[lid] += s_accum[lid + stride];
            if (s_data[lid + stride] > s_data[lid])
                s_data[lid] = s_data[lid + stride];
        }
        barrier();
    }

    // Phase 4: FRU — adjust so sum == D exactly
    if (lid == 0) {
        int actual_sum = int(s_accum[0]);
        s_remainder = params.denominator - actual_sum;
        s_max_rem_idx = -1;
    }
    barrier();

    // Find which invocation has the max remainder element
    if (local_max_rem == s_data[0] && local_max_rem_i >= 0) {
        s_max_rem_idx = local_max_rem_i;
    }
    barrier();

    // Add the deficit to the element with largest remainder
    if (lid == 0 && s_max_rem_idx >= 0) {
        scratch_a.data[base + s_max_rem_idx] += s_remainder;
    }
}
```

```glsl
// ============================================================
// attention_weighted_sum.comp
// attn_probs × V. One workgroup per (head, query_position).
// Output: scratch_a [n_tokens × n_heads × d_head]
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 2) buffer KvCache  { int data[]; } kv_cache;

layout(set = 3, binding = 0) uniform Params {
    int n_tokens;
    int n_heads;
    int d_head;
    int seq_len;
} params;

// scratch_a currently holds attention probs [n_heads × n_tokens × seq_len]
// kv_cache holds V values
// We need a separate output region — use end of scratch_a past the probs

void main() {
    int head_idx = int(gl_WorkGroupID.x);
    int query_pos = int(gl_WorkGroupID.y);
    uint lid = gl_LocalInvocationID.x;

    if (head_idx >= params.n_heads || query_pos >= params.n_tokens) return;

    // Attention probs for this head and query
    int prob_base = (head_idx * params.n_tokens + query_pos) * params.seq_len;

    // Output offset: past all probs
    int out_offset = params.n_heads * params.n_tokens * params.seq_len;
    int out_base = out_offset +
                   (query_pos * params.n_heads + head_idx) * params.d_head;

    // Each invocation computes a subset of d_head dimensions
    for (int d = int(lid); d < params.d_head; d += 256) {
        int64_t acc = 0;
        for (int key_pos = 0; key_pos < params.seq_len; key_pos++) {
            int prob = scratch_a.data[prob_base + key_pos];
            // V from kv_cache: kv_select=1
            int v_base = key_pos * params.n_heads * params.d_head * 2 +
                         head_idx * params.d_head * 2 +
                         params.d_head; // offset for V (after K)
            acc += int64_t(prob) * int64_t(kv_cache.data[v_base + d]);
        }
        // Q16: divide by D
        scratch_a.data[out_base + d] = int(acc / int64_t(65536));
    }
}
```

```glsl
// ============================================================
// output_project.comp
// GEMM: attention output [n_tokens × d_model] × weights [d_model × d_model]
// Input: scratch_a (attention output region), Output: scratch_b
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 1) buffer LayerWeights { int data[]; } weights;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;

layout(set = 3, binding = 0) uniform Params {
    int n_tokens;
    int d_model;
    int layer_idx;
    int weights_offset;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    int row = int(gid) / params.d_model;
    int col = int(gid) % params.d_model;
    if (row >= params.n_tokens) return;

    // Input is in scratch_a after attention weighted sum output region
    // Caller sets the correct base via input layout convention
    int in_base = row * params.d_model;

    // Output projection weights offset within layer
    // Convention: out_proj follows QKV weights
    int w_base = params.weights_offset + params.d_model * 3 * params.d_model +
                 col;

    int64_t acc = 0;
    for (int k = 0; k < params.d_model; k++) {
        acc += int64_t(scratch_a.data[in_base + k]) *
               int64_t(weights.data[w_base + k * params.d_model]);
    }
    scratch_b.data[row * params.d_model + col] = int(acc / int64_t(65536));
}
```

```glsl
// ============================================================
// mlp.comp
// Two-phase MLP: up-project + activation, then down-project.
// Phase selected by a secondary dispatch (host dispatches twice
// with different params, or this shader handles both in sequence
// for invocations that map to each phase).
// Here: single-pass approach — each invocation computes one output element.
// Input: scratch_a, Output: scratch_b
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 1) buffer LayerWeights { int data[]; } weights;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;

layout(set = 3, binding = 0) uniform Params {
    int n_tokens;
    int d_model;
    int mlp_dim;
    int layer_idx;
    int up_weights_offset;
    int down_weights_offset;
    int activation_type; // 0=SiLU, 1=GELU, 2=ReLU
    int _pad;
} params;

// Integer SiLU: x * sigmoid(x)
// sigmoid(x) ≈ piecewise linear in Q16
int silu(int x) {
    // sigmoid: 5-segment piecewise linear
    // For x < -4*D: sigmoid ≈ 0
    // For x > 4*D: sigmoid ≈ D
    int D = 65536;
    int sig;
    if (x < -262144) { sig = 0; }           // x < -4.0
    else if (x < -131072) {                   // -4.0 <= x < -2.0
        sig = int(int64_t(x + 262144) * int64_t(4681) / int64_t(D)); // slope ~0.07
    }
    else if (x < 0) {                         // -2.0 <= x < 0
        sig = 4681 + int(int64_t(x + 131072) * int64_t(28087) / int64_t(131072));
    }
    else if (x < 131072) {                    // 0 <= x < 2.0
        sig = 32768 + int(int64_t(x) * int64_t(28087) / int64_t(131072));
    }
    else if (x < 262144) {                    // 2.0 <= x < 4.0
        sig = 60855 + int(int64_t(x - 131072) * int64_t(4681) / int64_t(D));
    }
    else { sig = D; }

    // silu = x * sigmoid / D
    return int(int64_t(x) * int64_t(sig) / int64_t(D));
}

int relu(int x) {
    return x > 0 ? x : 0;
}

int activate(int x) {
    if (params.activation_type == 0) return silu(x);
    if (params.activation_type == 2) return relu(x);
    return silu(x); // default
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    int row = int(gid) / params.d_model;
    int col = int(gid) % params.d_model;
    if (row >= params.n_tokens) return;

    int in_base = row * params.d_model;

    // Up-project: input[row] × up_weights[:, intermediate]
    // Then activate, then down-project to output
    // For each output element, we must compute the full intermediate vector
    // and reduce. This is expensive per-invocation but avoids a second pass.

    int64_t acc = 0;
    for (int m = 0; m < params.mlp_dim; m++) {
        // Up-project: dot(input[row], up_weights[:, m])
        int64_t up_acc = 0;
        for (int k = 0; k < params.d_model; k++) {
            up_acc += int64_t(scratch_a.data[in_base + k]) *
                      int64_t(weights.data[params.up_weights_offset + k * params.mlp_dim + m]);
        }
        int up_val = int(up_acc / int64_t(65536));
        int act_val = activate(up_val);

        // Down-project: act_val * down_weights[m, col]
        acc += int64_t(act_val) *
               int64_t(weights.data[params.down_weights_offset + m * params.d_model + col]);
    }

    scratch_b.data[row * params.d_model + col] = int(acc / int64_t(65536));
}
```

```glsl
// ============================================================
// lm_head.comp
// Final projection: hidden [n_tokens × d_model] × lm_head [d_model × vocab]
// Output: logits [n_tokens × vocab_size] in scratch_b
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 2) buffer LmHead { int data[]; } lm_head;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;

layout(set = 3, binding = 0) uniform Params {
    int n_tokens;
    int d_model;
    int vocab_size;
    int _pad;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    int row = int(gid) / params.vocab_size;
    int col = int(gid) % params.vocab_size;
    if (row >= params.n_tokens) return;

    int in_base = row * params.d_model;
    int64_t acc = 0;
    for (int k = 0; k < params.d_model; k++) {
        acc += int64_t(scratch_a.data[in_base + k]) *
               int64_t(lm_head.data[k * params.vocab_size + col]);
    }
    scratch_b.data[row * params.vocab_size + col] = int(acc / int64_t(65536));
}
```

```glsl
// ============================================================
// kv_cache_append.comp
// Stores K and V vectors for new positions into KV cache.
// Each invocation writes one scalar element.
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;
layout(set = 2, binding = 2) buffer KvCache  { int data[]; } kv_cache;

layout(set = 3, binding = 0) uniform Params {
    int n_new_tokens;
    int n_heads;
    int d_head;
    int layer_idx;
    int start_position;
    int max_seq_len;
    int _pad0;
    int _pad1;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total_elems = params.n_new_tokens * params.n_heads * params.d_head * 2;
    if (int(gid) >= total_elems) return;

    // Decompose gid into (token, head, dim, kv_select)
    int rem = int(gid);
    int kv_select = rem % 2;          rem /= 2;
    int dim       = rem % params.d_head;   rem /= params.d_head;
    int head      = rem % params.n_heads;  rem /= params.n_heads;
    int token     = rem;

    int position = params.start_position + token;
    if (position >= params.max_seq_len) return;

    // Source: QKV output in scratch_b
    // Layout: [n_tokens × 3 × n_heads × d_head]
    // K = offset 1, V = offset 2 within the 3-block
    int qkv_offset = token * 3 * params.n_heads * params.d_head +
                     (kv_select + 1) * params.n_heads * params.d_head +
                     head * params.d_head + dim;

    // Destination in KV cache
    int cache_offset = position * params.n_heads * params.d_head * 2 +
                       head * params.d_head * 2 +
                       kv_select * params.d_head + dim;

    kv_cache.data[cache_offset] = scratch_b.data[qkv_offset];
}
```

```glsl
// ============================================================
// residual_add.comp
// Element-wise: scratch_a[i] = scratch_a[i] + scratch_b[i]
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;

layout(set = 3, binding = 0) uniform Params {
    int n_elements;
    int _pad0;
    int _pad1;
    int _pad2;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= params.n_elements) return;
    scratch_a.data[gid] = scratch_a.data[gid] + scratch_b.data[gid];
}
```

```glsl
// ============================================================
// fact_write_batch.comp
// Writes N facts to fact_store in parallel.
// Input: scratch_a = facts [n_facts × 10 ints (40 bytes)]
//        scratch_b = slot_ids [n_facts] i32
// Output: fact_store at base_offset + slot_ids[gid]
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 1, binding = 1) buffer FactStore { int data[]; } fact_store;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;

layout(set = 3, binding = 0) uniform Params {
    int n_facts;
    int base_offset;
    int fact_store_capacity;
    int _pad;
} params;

layout(set = 3, binding = 1) buffer StatusBuf { int data[]; } status_buf;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= params.n_facts) return;

    int slot_id = scratch_b.data[gid];
    int target = params.base_offset + slot_id;

    if (target < 0 || target >= params.fact_store_capacity) {
        status_buf.data[gid] = 201; // ERR_KB_FULL
        return;
    }

    // Copy 10 ints (40 bytes) per fact
    int src_base = int(gid) * 10;
    int dst_base = target * 10;
    for (int i = 0; i < 10; i++) {
        fact_store.data[dst_base + i] = scratch_a.data[src_base + i];
    }
    status_buf.data[gid] = 0;
}
```

```glsl
// ============================================================
// fact_read_batch.comp
// Reads N facts from fact_store in parallel.
// Input: scratch_a = read_offsets [n_reads] i32 (absolute offsets)
// Output: scratch_b = facts [n_reads × 10 ints]
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 1, binding = 1) buffer FactStore { int data[]; } fact_store;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;

layout(set = 3, binding = 0) uniform Params {
    int n_reads;
    int _pad0;
    int _pad1;
    int _pad2;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= params.n_reads) return;

    int offset = scratch_a.data[gid];
    int src_base = offset * 10;
    int dst_base = int(gid) * 10;

    for (int i = 0; i < 10; i++) {
        scratch_b.data[dst_base + i] = fact_store.data[src_base + i];
    }
}
```

```glsl
// ============================================================
// fact_scan_by_tag.comp
// Scans contiguous fact range for matching tag.
// Output: matching slot indices to scratch_a via atomic counter.
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 1, binding = 1) buffer FactStore { int data[]; } fact_store;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;

layout(set = 3, binding = 0) uniform Params {
    int base_offset;
    int scan_length;
    int target_tag;
    int max_results;
} params;

layout(set = 3, binding = 2) buffer ResultCounts { int data[]; } result_counts;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= params.scan_length) return;

    // Fact tag is the first int (offset 0) of the 10-int fact struct
    int fact_base = (params.base_offset + int(gid)) * 10;
    int tag = fact_store.data[fact_base];

    if (tag == params.target_tag) {
        int idx = atomicAdd(result_counts.data[0], 1);
        if (idx < params.max_results) {
            scratch_a.data[idx] = int(gid); // slot index within KB
        }
    }
}
```

```glsl
// ============================================================
// scoped_search.comp
// Same as fact_scan_by_tag but over pre-flattened absolute offsets.
// Host builds scan plan across multiple KBs.
// Input: scratch_b = scan_offsets [total_facts] i32 (absolute fact indices)
// Output: scratch_a = matching facts info via atomic.
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 1, binding = 1) buffer FactStore { int data[]; } fact_store;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;

layout(set = 3, binding = 0) uniform Params {
    int n_chain_entries;
    int total_facts;
    int target_tag;
    int max_results;
} params;

layout(set = 3, binding = 2) buffer ResultCounts { int data[]; } result_counts;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= params.total_facts) return;

    int abs_offset = scratch_b.data[gid];
    int fact_base = abs_offset * 10;
    int tag = fact_store.data[fact_base];

    if (tag == params.target_tag) {
        int idx = atomicAdd(result_counts.data[0], 1);
        if (idx < params.max_results) {
            // Store absolute offset so host can read the fact
            scratch_a.data[idx] = abs_offset;
        }
    }
}
```

```glsl
// ============================================================
// buffer_copy.comp
// Generic element copy from scratch_a to scratch_b.
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 2, binding = 0) buffer ScratchA { int data[]; } scratch_a;
layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;

layout(set = 3, binding = 0) uniform Params {
    int src_offset;
    int dst_offset;
    int n_elements;
    int element_size; // in ints (1 for i32, 2 for i64, 10 for Fact)
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= params.n_elements) return;

    int src = params.src_offset + int(gid) * params.element_size;
    int dst = params.dst_offset + int(gid) * params.element_size;

    for (int i = 0; i < params.element_size; i++) {
        scratch_b.data[dst + i] = scratch_a.data[src + i];
    }
}
```

```glsl
// ============================================================
// buffer_fill.comp
// Fills scratch_b region with a constant i32 value.
// ============================================================
#version 450
layout(local_size_x = 256) in;

layout(set = 2, binding = 1) buffer ScratchB { int data[]; } scratch_b;

layout(set = 3, binding = 0) uniform Params {
    int dst_offset;
    int n_elements;
    int fill_value;
    int element_size;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (int(gid) >= params.n_elements) return;

    int dst = params.dst_offset + int(gid) * params.element_size;
    for (int i = 0; i < params.element_size; i++) {
        scratch_b.data[dst + i] = params.fill_value;
    }
}
```

Shader turn 1 complete. 17 shaders: 11 LLM, 4 KB, 2 utility. Ready for turn 2.
