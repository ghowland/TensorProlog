# TensorProlog Workflow Cookbook

## Complete ML Practitioner Reference

---

## Part 1: LLM Workflows (70%)

---

### 1.1 Model Initialization — Loading a Pretrained VDR Model

The starting point for any LLM work. Weights arrive as integer checkpoints.

```
// Initialize device
TensorPrologInit(0);
TensorPrologDevice_t dev;
TensorPrologDeviceGet(&dev, 0);
TensorPrologDeviceSetCurrent(dev);

// Check capabilities
TensorPrologDeviceProps_t props;
TensorPrologDeviceGetProperties(&props, dev);
// props.max_q_basis tells you Q16/Q32/Q335 support
// props.has_fru tells you if transcendentals are hardware-accelerated
// props.kb_cache_size_bytes tells you how many KBs fit in shared memory

// Allocate model weights
i64 n_params = 7_000_000_000;  // 7B model
void* weights;
TensorPrologMallocTyped(&weights, TensorProlog_Q16, n_params);

// Load checkpoint — bit-identical to what was saved, always, on any device
TensorPrologTrainCheckpointLoad(weights, NULL, TensorProlog_Q16, n_params,
    "/models/llama-vdr-7b-q16.ckpt", stream);

// No precision negotiation. No mixed-precision config.
// No "should I use FP16 or BF16 or FP8 or TF32?"
// No quantization calibration. No GPTQ, AWQ, SmoothQuant.
// It's integers. Load them. Done.
```

**What's different from conventional:** No `torch.float16` vs `torch.bfloat16` decision. No `model.half()` or `model.to(dtype=...)`. No quantization pipeline. No calibration dataset. The checkpoint is the model. The model is integers. Loading is memcpy.

---

### 1.2 Single Forward Pass — Inference

The fundamental operation. Input tokens → output logits.

```
// Embed: lookup table, integer indices → VDR vectors
// embedding table is weights[0..vocab_size * d_model]
// input_ids is array of i32 token indices
// embedded is output: array of vdr_q16 vectors
TensorPrologVdrGemm(TensorProlog_Q16, NO_TRANS, NO_TRANS,
    seq_len, d_model, vocab_size,
    &one_q16, one_hot_input, seq_len,
    embedding_table, vocab_size,
    &zero_q16, embedded, seq_len, stream);
// Or direct index gather — simpler for sparse one-hot:
// for each token, copy row from embedding table. Pure memcpy.

// For each transformer layer:
for (i32 layer = 0; layer < n_layers; layer++) {

    // Layer norm — exact mean and variance
    TensorPrologVdrLayerNorm(TensorProlog_Q16, hidden, normed,
        gamma[layer], beta[layer], d_model, 0, stream);

    // QKV projection — single GEMM, split output
    TensorPrologVdrGemm(TensorProlog_Q16, NO_TRANS, NO_TRANS,
        seq_len, 3 * d_model, d_model,
        &one_q16, normed, seq_len,
        qkv_weights[layer], d_model,
        &zero_q16, qkv_out, seq_len, stream);
    // Split qkv_out into Q, K, V by pointer arithmetic — no copy

    // Attention — fused, exact
    TensorPrologAttentionConfig_t attn_cfg = {
        .qbasis = TensorProlog_Q16,
        .seq_len = seq_len,
        .d_model = d_model,
        .n_heads = n_heads,
        .d_head = d_model / n_heads,
        .causal_mask = true,
        .softmax_type = TensorProlog_SOFTMAX_QUADRATIC,
    };
    TensorPrologAttentionForward(&attn_cfg, Q, K, V, attn_out, NULL, stream);
    // attn_out attention weights sum to 65536 per row. Exactly.
    // Not 65535.99. Not 65536.01. 65536.

    // Output projection
    TensorPrologVdrGemm(TensorProlog_Q16, NO_TRANS, NO_TRANS,
        seq_len, d_model, d_model,
        &one_q16, attn_out, seq_len,
        out_proj_weights[layer], d_model,
        &zero_q16, projected, seq_len, stream);

    // Residual add — exact
    TensorPrologVdrAdd(TensorProlog_Q16, hidden, projected, hidden, seq_len * d_model, stream);

    // MLP: norm → up-project → activation → down-project → residual
    TensorPrologVdrLayerNorm(TensorProlog_Q16, hidden, normed,
        mlp_gamma[layer], mlp_beta[layer], d_model, 0, stream);

    TensorPrologVdrGemm(TensorProlog_Q16, NO_TRANS, NO_TRANS,
        seq_len, mlp_dim, d_model,
        &one_q16, normed, seq_len,
        mlp_up[layer], d_model,
        &zero_q16, mlp_hidden, seq_len, stream);

    // Activation: quadratic surrogate or exact GELU via FRU
    // Quadratic: element-wise square, normalized — same pattern as softmax
    // Or if FRU available:
    // TensorPrologFRUExp(TensorProlog_Q16, mlp_hidden, ...) for exact GELU

    TensorPrologVdrGemm(TensorProlog_Q16, NO_TRANS, NO_TRANS,
        seq_len, d_model, mlp_dim,
        &one_q16, mlp_activated, seq_len,
        mlp_down[layer], mlp_dim,
        &zero_q16, mlp_out, seq_len, stream);

    TensorPrologVdrAdd(TensorProlog_Q16, hidden, mlp_out, hidden, seq_len * d_model, stream);
}

// Final layer norm + logit projection
TensorPrologVdrLayerNorm(TensorProlog_Q16, hidden, normed_final,
    final_gamma, final_beta, d_model, 0, stream);

TensorPrologVdrGemm(TensorProlog_Q16, NO_TRANS, NO_TRANS,
    seq_len, vocab_size, d_model,
    &one_q16, normed_final, seq_len,
    lm_head, d_model,
    &zero_q16, logits, seq_len, stream);

// Softmax over vocabulary — exact
TensorPrologVdrSoftmax(TensorProlog_Q16, logits_last_pos, probs, vocab_size, stream);
// probs sums to 65536. Pick the highest. That's your next token.
```

**What you never do:** No `torch.no_grad()` context manager. No `model.eval()` mode switch. No `torch.cuda.amp.autocast()`. No mixed-precision forward. No KV-cache management (covered separately in 1.4). The forward pass is the forward pass. Same code path training and inference.

---

### 1.3 Training Loop — Pretraining from Scratch

```
// Create compute graph for backward pass
TensorPrologComputeGraph_t graph;
TensorPrologTrainComputeGraphCreate(&graph);

// Optimizer state
void *m_state, *v_state;  // Adam first/second moments
TensorPrologMallocTyped(&m_state, TensorProlog_Q16, n_params);
TensorPrologMallocTyped(&v_state, TensorProlog_Q16, n_params);
TensorPrologMemset(m_state, 0, n_params * 8);
TensorPrologMemset(v_state, 0, n_params * 8);

// Learning rate as exact fraction: 3/10000 at Q16
// V = 65536 * 3 / 10000 = 19, D = 65536
// This is not 0.0003 truncated to float16. It's 19/65536 = 0.000289916...
// If you want exactly 3/10000 at higher precision, use Q32.
vdr_q16 lr = { .v = 19, .r0 = 0 };

vdr_q16 beta1 = { .v = 58982, .r0 = 0 };  // ~0.9
vdr_q16 beta2 = { .v = 64880, .r0 = 0 };  // ~0.99

for (i32 epoch = 0; epoch < n_epochs; epoch++) {
    for (i32 batch = 0; batch < n_batches; batch++) {

        // Load batch — integer token IDs, nothing to convert
        load_batch(train_data, batch, input_ids, target_ids);

        // Forward pass with graph recording
        TensorPrologTrainComputeGraphRecord(graph, true);
        // ... (same forward pass as 1.2, operations recorded in graph)
        TensorPrologTrainComputeGraphRecord(graph, false);

        // Loss — exact cross-entropy
        TensorPrologTrainComputeLoss(TensorProlog_Q16, logits, target_ids, &loss,
            vocab_size, batch_size, stream);
        // loss is a VDR scalar. Compare to previous batch by integer comparison.
        // Monotonic decrease is verifiable by < operator. Not by "is it within epsilon."

        // Backward — exact gradients
        TensorPrologTrainBackwardPass(graph, &loss, stream);
        // Every gradient is exact. Chain rule on exact values = exact derivatives.
        // No gradient explosion from accumulated float error.
        // No gradient underflow from vanishing float precision.
        // No loss scaling needed. No gradient clipping needed.

        // Update — exact Adam
        TensorPrologTrainAdamUpdate(TensorProlog_Q16, weights, gradients, m_state, v_state,
            &lr, &beta1, &beta2, epoch * n_batches + batch, n_params, stream);
        // Parameters change by exactly lr * adjusted_gradient.
        // Not by approximately lr * approximately adjusted_gradient.
    }

    // Checkpoint — bit-identical save
    TensorPrologTrainCheckpointSave(weights, optimizer_state, TensorProlog_Q16, n_params,
        checkpoint_path, stream);
    // Resume on a different machine: identical training trajectory.
    // Not "similar." Identical. Every parameter, every gradient, every update.
}
```

**What you never do:** No `scaler = GradScaler()`. No `scaler.scale(loss).backward()`. No `scaler.step(optimizer)`. No `scaler.update()`. No `torch.nn.utils.clip_grad_norm_`. No warmup scheduler for float stability. No NaN detection and recovery. No "training diverged, try lower learning rate." If training diverges, the cause is the learning rate or the data, not arithmetic instability. Debuggable.

---

### 1.4 KV-Cache for Autoregressive Generation

In conventional LLMs, KV-cache avoids recomputing attention for previous tokens. In TensorProlog, the cache is KB-resident.

```
// Create session for generation
TensorPrologSessionConfig_t sess_cfg = {
    .kb_root_id = model_kb_root,
    .user_id = user_id,
    .visibility_level = TensorProlog_VIS_INTERNAL,
    .max_kb_count = 1000,
    .max_live_memory_bytes = 64 * 1024 * 1024,  // 64MB for KV cache
};
TensorPrologSession_t session;
TensorPrologSessionCreate(&session, &sess_cfg);

TensorPrologStream_t gen_stream;
TensorPrologStreamCreateWithSession(&gen_stream, session);

// KV cache lives in a KB — addressed by layer_id + position
TensorPrologKBConfig_t kv_cfg = {
    .name = "kv_cache",
    .parent_id = model_kb_root,
    .visibility = TensorProlog_VIS_INTERNAL,
    .max_facts = n_layers * max_seq_len * 2,  // K and V per layer per position
};
i32 kv_kb_id;
TensorPrologKBCreate(kb_store, &kv_kb_id, &kv_cfg);

// Generate tokens autoregressively
for (i32 step = 0; step < max_gen_len; step++) {

    // Forward pass for current position only
    // Q: current position (1 × d_model)
    // K, V: compute for current position, append to cache

    for (i32 layer = 0; layer < n_layers; layer++) {
        // Compute K, V for this position
        // ... (projection GEMM on single position)

        // Store in KB
        i32 k_slot = layer * max_seq_len * 2 + step * 2;
        i32 v_slot = k_slot + 1;
        TensorPrologKBFactAssert(kb_store, kv_kb_id, k_slot, &k_current);
        TensorPrologKBFactAssert(kb_store, kv_kb_id, v_slot, &v_current);

        // Attention: Q against all cached K, V up to current position
        // Load cached K, V from KB into device buffer
        for (i32 pos = 0; pos <= step; pos++) {
            TensorPrologKBFactQuery(kb_store, kv_kb_id,
                layer * max_seq_len * 2 + pos * 2, &cached_k[pos]);
            TensorPrologKBFactQuery(kb_store, kv_kb_id,
                layer * max_seq_len * 2 + pos * 2 + 1, &cached_v[pos]);
        }
        // Or batch-load the range — same O(1) per fact

        // Attention for this layer, this position
        TensorPrologAttentionConfig_t step_cfg = {
            .qbasis = TensorProlog_Q16,
            .seq_len = step + 1,
            .d_model = d_model,
            .n_heads = n_heads,
            .d_head = d_model / n_heads,
            .causal_mask = false,  // all positions visible (we only have up to current)
            .softmax_type = TensorProlog_SOFTMAX_QUADRATIC,
        };
        TensorPrologAttentionForward(&step_cfg, q_current, cached_k, cached_v,
            attn_out, NULL, gen_stream);
    }

    // Final logits → softmax → sample
    TensorPrologVdrSoftmax(TensorProlog_Q16, logits, probs, vocab_size, gen_stream);

    // Greedy: find max
    // Or sampling: use probs as exact distribution for weighted random
    i32 next_token = sample_from_exact_distribution(probs, vocab_size);

    if (next_token == EOS_TOKEN) break;
}

// KV cache is in the KB. If you snapshot the session, the cache is included.
// Restore the snapshot later → generation continues from exactly where it stopped.
// Not "approximately where." Exactly where. Same attention weights. Same cache state.
```

**Key difference:** KV-cache in KB means it survives session boundaries. Snapshot mid-generation, restore next week, continue generating. Conventional systems lose the KV-cache when the session ends. Also: the cache values are exact. No KV-cache quantization needed (a whole subfield of optimization that doesn't exist here).

---

### 1.5 Multi-GPU Distributed Training

```
// Initialize across 8 GPUs
TensorPrologComm_t comm;
TensorPrologDistCommCreate(&comm, 8, rank);

// Each rank loads the same checkpoint — bit-identical on all ranks
TensorPrologTrainCheckpointLoad(weights, opt_state, TensorProlog_Q16, n_params, path, stream);
// Verification: allreduce the checksum, confirm identical
i32 local_checksum = compute_integer_checksum(weights, n_params);
i32 global_checksum;
TensorPrologDistAllReduce(&local_checksum, &global_checksum, 1, TensorProlog_Q16,
    TensorProlog_SUM, comm, stream);
// global_checksum == local_checksum * 8 if all ranks identical. Integer check.

for (i32 step = 0; step < total_steps; step++) {

    // Each rank processes different data shard
    load_shard(train_data, rank, step, input_ids, target_ids);

    // Forward + backward (same as 1.3)
    forward_pass(weights, input_ids, logits, graph, stream);
    TensorPrologTrainComputeLoss(TensorProlog_Q16, logits, target_ids, &loss,
        vocab_size, batch_size, stream);
    TensorPrologTrainBackwardPass(graph, &loss, stream);

    // AllReduce gradients — THIS IS THE CRITICAL DIFFERENCE
    TensorPrologDistAllReduce(gradients, gradients_reduced, n_params,
        TensorProlog_Q16, TensorProlog_SUM, comm, stream);
    // Integer sum is associative. Ring reduce, tree reduce, butterfly reduce —
    // ALL produce bit-identical results. ALWAYS.
    // Conventional float allreduce is non-deterministic because float addition
    // is non-associative. Different reduction orders → different sums.
    // This is why distributed float training is non-reproducible.
    // This is why TensorProlog distributed training IS reproducible.

    // All ranks apply identical update to identical parameters
    TensorPrologTrainAdamUpdate(TensorProlog_Q16, weights, gradients_reduced, m, v,
        &lr, &beta1, &beta2, step, n_params, stream);
    // After update, all 8 ranks have bit-identical weights.
    // Not "approximately the same weights." Identical.
    // No gradient sync bugs hiding in float noise.
}
```

**What's eliminated:** The entire class of "distributed training non-determinism" bugs. In conventional systems, running the same training job twice on the same hardware can produce different models because allreduce ordering varies. In TensorProlog, same data + same hyperparameters = same model. Every time. This also means: if rank 3 produces a different loss than rank 0 at any step, it's a real bug, not float noise. Binary search to find it.

---

### 1.6 Fine-Tuning with LoRA

```
// Load base model (frozen)
TensorPrologTrainCheckpointLoad(base_weights, NULL, TensorProlog_Q16, n_base_params, base_path, stream);

// Allocate LoRA adapters — small exact integer matrices
i32 lora_rank = 16;
void *lora_A, *lora_B;  // per layer
for (i32 layer = 0; layer < n_layers; layer++) {
    TensorPrologMallocTyped(&lora_A[layer], TensorProlog_Q16, d_model * lora_rank);
    TensorPrologMallocTyped(&lora_B[layer], TensorProlog_Q16, lora_rank * d_model);
    // Initialize A: small random integers
    // Initialize B: zeros (standard LoRA init)
    TensorPrologMemset(lora_B[layer], 0, lora_rank * d_model * 8);
}

for (i32 step = 0; step < finetune_steps; step++) {

    // Forward: base_out + lora_B @ lora_A @ input
    for (i32 layer = 0; layer < n_layers; layer++) {
        // Base path (frozen, no gradient)
        TensorPrologVdrGemm(TensorProlog_Q16, NO_TRANS, NO_TRANS,
            seq_len, d_model, d_model,
            &one_q16, hidden, seq_len,
            base_weights_layer[layer], d_model,
            &zero_q16, base_out, seq_len, stream);

        // LoRA path (trainable)
        TensorPrologVdrGemm(TensorProlog_Q16, NO_TRANS, NO_TRANS,
            seq_len, lora_rank, d_model,
            &one_q16, hidden, seq_len,
            lora_A[layer], d_model,
            &zero_q16, lora_mid, seq_len, stream);

        TensorPrologVdrGemm(TensorProlog_Q16, NO_TRANS, NO_TRANS,
            seq_len, d_model, lora_rank,
            &one_q16, lora_mid, seq_len,
            lora_B[layer], lora_rank,
            &zero_q16, lora_out, seq_len, stream);

        // Combine: exact addition
        TensorPrologVdrAdd(TensorProlog_Q16, base_out, lora_out, combined, seq_len * d_model, stream);
    }

    // Loss + backward only through LoRA parameters
    // Gradients for base_weights are not computed (not in graph)
    // Gradients for lora_A, lora_B are exact
    TensorPrologTrainAdamUpdate(TensorProlog_Q16, lora_params, lora_grads, ...);
}

// Save LoRA adapter — tiny checkpoint, bit-identical
TensorPrologTrainCheckpointSave(lora_all, NULL, TensorProlog_Q16, total_lora_params,
    "lora-adapter-q16.ckpt", stream);

// Merge LoRA into base (optional, for deployment):
// W_merged = W_base + B @ A
// Exact addition. Merged model is exactly W_base + the learned delta.
// Not approximately. The fine-tuning delta is preserved without any
// quantization loss from merging.
```

**What's different:** LoRA in conventional systems often trains in FP16/BF16, then quantizes back to INT8 for deployment, losing precision in the merge. TensorProlog LoRA trains in Q16, merges in Q16, deploys in Q16. The merged model is exactly base + delta. No merge artifacts.

---

### 1.7 Serving — Continuous Batching with Sessions

```
// Model weights in global memory, shared across all sessions
void* model_weights;  // loaded once

// Per-request: create session, generate, destroy
void handle_request(i32 user_id, i32* prompt_tokens, i32 prompt_len) {

    // Session isolates this request's KV-cache and state
    TensorPrologSessionConfig_t cfg = {
        .kb_root_id = serving_root,
        .user_id = user_id,
        .visibility_level = TensorProlog_VIS_OWNER_ONLY,
        .max_kb_count = 100,
        .max_live_memory_bytes = 16 * 1024 * 1024,
    };
    TensorPrologSession_t session;
    TensorPrologSessionCreate(&session, &cfg);

    TensorPrologStream_t stream;
    TensorPrologStreamCreateWithSession(&stream, session);

    // Safety: user_id determines KB visibility
    // This user can only see their own KBs and public KBs
    // Other users' sessions are structurally invisible
    // Not filtered — absent from this session's scope

    // Prefill: process all prompt tokens at once
    forward_pass_prefill(model_weights, prompt_tokens, prompt_len, session, stream);

    // Decode: generate one token at a time
    i32 output_tokens[MAX_GEN_LEN];
    for (i32 step = 0; step < MAX_GEN_LEN; step++) {
        i32 next = forward_pass_decode_step(model_weights, session, stream);
        output_tokens[step] = next;
        if (next == EOS_TOKEN) break;
    }

    // Cleanup
    TensorPrologStreamDestroy(stream);
    TensorPrologSessionDestroy(session);
    // KV-cache gone. User state gone. Clean isolation.
}

// Continuous batching: multiple sessions share the GPU
// Each session has its own KV-cache KB, its own stream
// Scheduler interleaves sessions across SMs
// No cross-contamination because session credentials are integers
// that the hardware enforces

// For persistent conversations (chatbot):
// Don't destroy the session. Snapshot it.
TensorPrologSessionSnapshot(session, &snapshot);
TensorPrologSnapshotSave(&snapshot, user_session_path);
// Next conversation turn: restore and continue
TensorPrologSnapshotLoad(&snapshot, user_session_path);
TensorPrologSessionRestore(session, &snapshot);
// KV-cache restored. Conversation continues from exact prior state.
```

---

### 1.8 Evaluation and Benchmarking — Deterministic

```
// Run eval suite — SAME result every time
void run_eval(void* model_weights, eval_dataset_t* eval) {
    i32 total_correct = 0;
    i32 total = eval->n_examples;

    for (i32 i = 0; i < total; i++) {
        TensorPrologSession_t session;
        TensorPrologSessionCreate(&session, &eval_cfg);
        TensorPrologStream_t stream;
        TensorPrologStreamCreateWithSession(&stream, session);

        // Forward pass
        forward_pass(model_weights, eval->examples[i].tokens,
            eval->examples[i].len, logits, stream);

        TensorPrologVdrSoftmax(TensorProlog_Q16, logits, probs, vocab_size, stream);

        // Compare prediction to target
        i32 predicted = argmax(probs, vocab_size);
        if (predicted == eval->examples[i].target) total_correct++;

        TensorPrologStreamDestroy(stream);
        TensorPrologSessionDestroy(session);
    }

    // Accuracy as exact fraction
    // total_correct / total — exact. Not 0.847. It's 847/1000.
    // Run this eval on a different machine: 847/1000. Same examples, same model,
    // same result. Always.
    // Run it twice on the same machine: 847/1000. Both times.
    // This is why determinism matters for eval: you can trust the number.

    // Verify determinism explicitly
    bool identical;
    TensorPrologProfileVerifyDeterminism(forward_kernel, args, grid, block, 5, &identical);
    // identical == true. Guaranteed. Or your hardware is broken.
}
```

**What this enables:** Regression testing for model changes. "Did this change improve accuracy?" becomes an integer comparison between two exact fractions, not a statistical test with confidence intervals over noisy float evaluations.

---

### 1.9 The VDR-LLM-Prolog Session — Full System

Everything above was raw TensorProlog for model computation. This workflow shows the full system from the paper: LLM + KB + Prolog + grammars + runners.

```
// ========================================
// Session Setup: SRE Investigation System
// ========================================

TensorPrologSession_t sre_session;
TensorPrologSessionCreate(&sre_session, &sre_cfg);
TensorPrologStream_t sre_stream;
TensorPrologStreamCreateWithSession(&sre_stream, sre_session);

// Create KB tree for SRE domain
i32 root_kb, services_kb, incidents_kb, rules_kb, grammars_kb;
TensorPrologKBCreate(kb_store, &root_kb, &(TensorPrologKBConfig_t){
    .name = "sre_root", .parent_id = -1, .visibility = TensorProlog_VIS_INTERNAL });
TensorPrologKBCreate(kb_store, &services_kb, &(TensorPrologKBConfig_t){
    .name = "services", .parent_id = root_kb });
TensorPrologKBCreate(kb_store, &incidents_kb, &(TensorPrologKBConfig_t){
    .name = "incidents", .parent_id = root_kb });
TensorPrologKBCreate(kb_store, &rules_kb, &(TensorPrologKBConfig_t){
    .name = "triage_rules", .parent_id = root_kb });
TensorPrologKBCreate(kb_store, &grammars_kb, &(TensorPrologKBConfig_t){
    .name = "grammars", .parent_id = root_kb });

// Load seed rules (the 150+ accumulated triage rules)
load_seed_rules(kb_store, rules_kb, "sre_rules.snapshot");

// Create SRE findings grammar
TensorPrologGrammar_t findings_grammar;
TensorPrologGrammarCreate(&findings_grammar,
    "| {service:text} | {error_rate:vdr_value} | {confidence:vdr_value} "
    "| {cause:text} | {status:enum(investigating|confirmed|resolved)} |");
TensorPrologGrammarStoreInKB(kb_store, grammars_kb, 0, findings_grammar);

// ========================================
// Processor Runner: Continuous Prometheus Ingest
// ========================================

TensorPrologRunner_t prom_runner;
TensorPrologRunnerCreateProcessor(&prom_runner, &(TensorPrologProcessorConfig_t){
    .session = prom_session,  // separate session for isolation
    .source_type = TensorProlog_SOURCE_PROMETHEUS,
    .connection_params = prom_connection_bytes,
    .ingest_fn = prometheus_ingest,    // compacts metrics to KB facts
    .max_turns_before_recycle = 200,   // fresh LLM every 200 turns
});
TensorPrologRunnerStart(prom_runner);
// Now running continuously. Every metric lands as an exact VDR fraction
// at an integer KB address. No LLM involved. No tokens spent.

// ========================================
// Polling Runner: Rule Evaluation Every 60s
// ========================================

TensorPrologRunner_t poll_runner;
TensorPrologRunnerCreatePoller(&poll_runner, &(TensorPrologPollerConfig_t){
    .interval_ms = 60000,
    .session = poll_session,
    .poll_fn = triage_poll,
    .max_consecutive_errors = 5,
});
TensorPrologRunnerStart(poll_runner);

// triage_poll implementation:
TensorPrologStatus_t triage_poll(TensorPrologSession_t session) {
    // Fire all triage rules against current metric state
    TensorPrologPrologFired_t fired[256];
    i32 n_fired;
    TensorPrologPrologFireAndCommit(kb_store, rules_kb, &n_fired, stream);
    // n_fired rules matched. Results committed as KB facts.
    // Zero LLM tokens. Pure Prolog unification over exact integers.

    // Check if any fired rule produced an incident
    for (i32 i = 0; i < n_fired; i++) {
        if (fired[i].action == ACTION_CREATE_INCIDENT) {
            // Create incident KB
            i32 inc_kb;
            TensorPrologKBCreate(kb_store, &inc_kb, &(TensorPrologKBConfig_t){
                .name = fired[i].incident_name,
                .parent_id = incidents_kb,
            });
            // Assert findings
            TensorPrologKBFactAssert(kb_store, inc_kb, SLOT_SERVICE, &fired[i].service);
            TensorPrologKBFactAssert(kb_store, inc_kb, SLOT_ERROR_RATE, &fired[i].rate);
            TensorPrologKBFactAssert(kb_store, inc_kb, SLOT_CONFIDENCE, &fired[i].confidence);
            // Confidence is exact VDR fraction from propagation:
            // Prometheus metric confidence 95/100, rule derivation 1/1,
            // combined: 95/100. Not "approximately 0.95."

            // Render findings table via grammar — zero LLM tokens
            TensorPrologGrammar_t grammar;
            TensorPrologGrammarLoadFromKB(kb_store, grammars_kb, 0, &grammar);
            TensorPrologGrammarFill_t fills[] = {
                { .slot = "service", .text = fired[i].service_name },
                { .slot = "error_rate", .vdr_value = fired[i].rate },
                { .slot = "confidence", .vdr_value = fired[i].confidence },
                { .slot = "cause", .text = fired[i].cause_name },
                { .slot = "status", .enum_value = "investigating" },
            };
            TensorPrologGrammarRender(grammar, fills, 5, output_buf, OUT_CAP, &out_len);
            // Every pipe, every space, every header came from the grammar.
            // LLM contribution: zero.
        }
    }
    return TensorProlog_OK;
}

// ========================================
// When an engineer opens a chat, the investigation is already done.
// LLM generates ~40 tokens of prose framing around pre-computed findings.
// ========================================

// ========================================
// Rule Hygiene: Monthly Pruning
// ========================================

TensorPrologRunner_t hygiene_runner;
TensorPrologRunnerCreateInternal(&hygiene_runner, &(TensorPrologInternalConfig_t){
    .session = hygiene_session,
    .interval_ms = 86400000 * 30,  // monthly
    .compute_fn = rule_hygiene,
});

TensorPrologStatus_t rule_hygiene(TensorPrologSession_t session) {
    i32 candidates[100];
    i32 n_candidates;
    TensorPrologPrologHygiene(kb_store, rules_kb,
        90,       // stale_days
        20, 100,  // min_success_rate = 20/100
        candidates, 100, &n_candidates, stream);
    // candidates contains rule_ids that are stale or failing.
    // Log them. Optionally retract:
    for (i32 i = 0; i < n_candidates; i++) {
        TensorPrologPrologRuleRetract(kb_store, rules_kb, candidates[i]);
    }
    return TensorProlog_OK;
}
```

---

### 1.10 Speculative Decoding

Draft model generates candidates fast, main model verifies in parallel.

```
// Draft model: smaller, same architecture, Q16
// Main model: full size, Q16
// Both produce exact softmax distributions.

void speculative_generate(void* main_weights, void* draft_weights,
                           TensorPrologSession_t session, TensorPrologStream_t stream) {

    i32 K = 5;  // speculate 5 tokens ahead

    while (!done) {
        // Draft generates K tokens greedily
        i32 draft_tokens[K];
        for (i32 i = 0; i < K; i++) {
            forward_pass_decode_step(draft_weights, draft_session, draft_stream);
            TensorPrologVdrSoftmax(TensorProlog_Q16, draft_logits, draft_probs, vocab_size, draft_stream);
            draft_tokens[i] = argmax(draft_probs, vocab_size);
        }

        // Main model verifies all K in parallel (single forward pass)
        forward_pass_prefill(main_weights, draft_tokens, K, session, stream);
        // Now have main model's exact probability for each draft token

        // Accept/reject by exact comparison
        for (i32 i = 0; i < K; i++) {
            TensorPrologVdrSoftmax(TensorProlog_Q16, main_logits[i], main_probs, vocab_size, stream);

            // main_probs[draft_tokens[i]] is the main model's exact probability
            // for the draft token. Compare exactly.
            vdr_q16 main_prob, draft_prob;
            main_prob = main_probs[draft_tokens[i]];
            draft_prob = draft_probs_saved[i][draft_tokens[i]];

            // Accept if main_prob >= draft_prob (or sampling criterion)
            i32 cmp;
            TensorPrologVdrCompare(TensorProlog_Q16, &main_prob, &draft_prob, &cmp, 1, stream);
            if (cmp >= 0) {
                accept(draft_tokens[i]);
            } else {
                // Reject: sample from adjusted distribution
                // adjusted = (main - draft) / (1 - draft)
                // All exact VDR arithmetic. No rounding error in rejection sampling.
                reject_and_resample(main_probs, draft_probs_saved[i], vocab_size);
                break;
            }
        }
    }
}
```

**What's different:** The acceptance criterion is exact comparison between exact probabilities. No float rounding means the speculative decoding acceptance/rejection boundary is mathematically precise. Conventional speculative decoding can accept or reject incorrectly due to float comparison being approximate.

---

### 1.11 Mixture of Experts (MoE) Routing

```
// Router: small network that produces gating scores per expert
// Top-K experts activated per token

void moe_layer(void* router_weights, void** expert_weights, i32 n_experts,
               i32 top_k, void* input, void* output, i32 seq_len,
               TensorPrologStream_t stream) {

    // Router forward: input → gating logits
    void* gate_logits;
    TensorPrologMallocTyped(&gate_logits, TensorProlog_Q16, seq_len * n_experts);
    TensorPrologVdrGemm(TensorProlog_Q16, NO_TRANS, NO_TRANS,
        seq_len, n_experts, d_model,
        &one_q16, input, seq_len,
        router_weights, d_model,
        &zero_q16, gate_logits, seq_len, stream);

    // Softmax per token over experts — exact
    for (i32 pos = 0; pos < seq_len; pos++) {
        TensorPrologVdrSoftmax(TensorProlog_Q16,
            gate_logits + pos * n_experts,
            gate_probs + pos * n_experts,
            n_experts, stream);
        // Each token's expert probabilities sum to 65536 exactly.
        // Top-K selection by exact integer comparison — no ambiguity
        // at decision boundaries. If two experts are tied, they're
        // exactly tied. Not "within epsilon of tied."
    }

    // Select top-K experts per token
    for (i32 pos = 0; pos < seq_len; pos++) {
        select_top_k(gate_probs + pos * n_experts, n_experts, top_k,
            selected_experts[pos], selected_weights[pos]);
    }

    // Dispatch to selected experts, weighted combine
    TensorPrologMemset(output, 0, seq_len * d_model * 8);
    for (i32 e = 0; e < top_k; e++) {
        // Expert forward pass
        TensorPrologVdrGemm(TensorProlog_Q16, ...);  // expert MLP

        // Weighted add to output — exact
        TensorPrologVdrScale(TensorProlog_Q16, expert_out, &selected_weights[e],
            scaled, seq_len * d_model, stream);
        TensorPrologVdrAdd(TensorProlog_Q16, output, scaled, output,
            seq_len * d_model, stream);
    }
    // Output is exact weighted sum of exactly top-K experts.
    // Load balancing loss computed on exact gating fractions.
}
```

**Key insight:** MoE routing is notoriously sensitive to float precision because expert selection happens at decision boundaries. Tokens near the boundary between expert 3 and expert 4 can be routed differently on different hardware due to float rounding in the softmax. TensorProlog routing is deterministic: same input always selects the same experts.

---

### 1.12 RLHF / DPO Training

```
// Direct Preference Optimization in exact arithmetic

void dpo_training_step(void* policy_weights, void* ref_weights,
                        i32* chosen_tokens, i32 chosen_len,
                        i32* rejected_tokens, i32 rejected_len,
                        vdr_q16 beta, TensorPrologStream_t stream) {

    // Forward pass: policy log-probs for chosen and rejected
    forward_pass(policy_weights, chosen_tokens, chosen_len, chosen_logits, stream);
    forward_pass(policy_weights, rejected_tokens, rejected_len, rejected_logits, stream);

    // Forward pass: reference log-probs (frozen)
    forward_pass(ref_weights, chosen_tokens, chosen_len, ref_chosen_logits, stream);
    forward_pass(ref_weights, rejected_tokens, rejected_len, ref_rejected_logits, stream);

    // Compute log probability ratios — exact
    // log_ratio_chosen = sum(log_policy_chosen - log_ref_chosen)
    // log_ratio_rejected = sum(log_policy_rejected - log_ref_rejected)

    // With FRU:
    TensorPrologFRULog(TensorProlog_Q16, policy_chosen_probs, log_policy_chosen, chosen_len, 8, stream);
    TensorPrologFRULog(TensorProlog_Q16, ref_chosen_probs, log_ref_chosen, chosen_len, 8, stream);
    TensorPrologVdrSub(TensorProlog_Q16, log_policy_chosen, log_ref_chosen,
        log_ratio_chosen_per_token, chosen_len, stream);
    TensorPrologVdrReduce(TensorProlog_Q16, log_ratio_chosen_per_token, TensorProlog_SUM,
        &zero_q16, &log_ratio_chosen, chosen_len, stream);
    // Same for rejected

    // DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    // All exact VDR operations. No float noise in the preference signal.
    // The model learns from exact preference gradients, not noisy approximations.

    TensorPrologVdrSub(TensorProlog_Q16, &log_ratio_chosen, &log_ratio_rejected, &margin, 1, stream);
    TensorPrologVdrMul(TensorProlog_Q16, &beta, &margin, &scaled_margin, 1, stream);
    // sigmoid via FRU or rational approximation
    // loss = -log(sigmoid_result)
    // backward through exact computation graph
    TensorPrologTrainBackwardPass(graph, &loss, stream);
    TensorPrologTrainAdamUpdate(TensorProlog_Q16, policy_weights, gradients, ...);
}
```

**What matters:** DPO/RLHF is extremely sensitive to the precision of log-probability differences. Small float errors in the policy vs reference log-prob ratio can flip the sign of the gradient — telling the model to move toward the rejected response instead of the chosen one. Exact arithmetic eliminates this failure mode entirely.

---

### 1.13 Model Merging

```
// Merge two fine-tuned models by exact weighted average
void merge_models(void* model_a, void* model_b, vdr_q16 weight_a,
                   vdr_q16 weight_b, void* merged, i64 n_params,
                   TensorPrologStream_t stream) {

    // merged = weight_a * model_a + weight_b * model_b
    TensorPrologVdrScale(TensorProlog_Q16, model_a, &weight_a, scaled_a, n_params, stream);
    TensorPrologVdrScale(TensorProlog_Q16, model_b, &weight_b, scaled_b, n_params, stream);
    TensorPrologVdrAdd(TensorProlog_Q16, scaled_a, scaled_b, merged, n_params, stream);

    // That's it. The merged model is exactly the weighted average.
    // weight_a = 7/10, weight_b = 3/10:
    // merged[i] = 7/10 * a[i] + 3/10 * b[i] — exact for every parameter.
    // No float accumulation error across 7 billion parameters.
    // The merge is reversible: given merged and model_a and the weights,
    // you can recover model_b exactly. Try that with float merges.
}

// SLERP merge (spherical interpolation) — also exact
// The trig functions use FRU for exact evaluation.
// Every interpolation step is exact.
// The merged model lies exactly on the geodesic between the two source models.
```

---

### 1.14 Remainder Monitoring in Production

```
// Periodically check that your Q-basis is sufficient
void remainder_health_check(void* weights, i64 n_params, TensorPrologStream_t stream) {

    void* magnitudes;
    TensorPrologMallocTyped(&magnitudes, TensorProlog_Q16, n_params);

    TensorPrologVdrRemainderMagnitude(TensorProlog_Q16, weights, magnitudes, n_params, stream);

    // Compute statistics on remainder magnitudes
    vdr_q16 max_remainder, mean_remainder;
    TensorPrologBuiltinMaxBy(TensorProlog_Q16, magnitudes, magnitudes, &max_remainder, n_params);
    TensorPrologStatsMean(TensorProlog_Q16, magnitudes, &mean_remainder, n_params, stream);

    // If max_remainder is large relative to D, you need deeper R slots or higher Q-basis.
    // If mean_remainder is near zero, Q16 is sufficient for this model.
    // This is your precision budget dashboard. Exact numbers, not heuristics.

    // Compare to threshold
    vdr_q16 threshold = { .v = 655, .r0 = 0 };  // ~1% of D
    i32 cmp;
    TensorPrologVdrCompare(TensorProlog_Q16, &max_remainder, &threshold, &cmp, 1, stream);
    if (cmp > 0) {
        // Consider reprojecting to Q32 for this layer
        TensorPrologVdrReproject(TensorProlog_Q16, TensorProlog_Q32, weights_layer, weights_layer_q32,
            layer_params, stream);
    }

    TensorPrologFree(magnitudes);
}
```

---

## Part 2: Diffusion Workflows (20%)

---

### 2.1 Forward Diffusion — Exact Noise Addition

```
// The forward process adds noise at exact scheduled levels.
// In conventional diffusion: alpha_bar_t is a float, noise is float,
// and each step compounds float error. After 1000 steps, the cumulative
// drift means the reverse process doesn't exactly invert the forward.

// In VDR: every noise schedule value is an exact fraction.
// alpha_bar_t at step t is precomputed as exact VDR values.

void forward_diffusion(void* x_0, void* noise, void* x_t,
                        vdr_q16 alpha_bar_t, i32 n_pixels,
                        TensorPrologStream_t stream) {

    // x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    // sqrt via FRU — exact to declared depth

    vdr_q16 one_minus_alpha = { .v = 65536 - alpha_bar_t.v, .r0 = 0 };

    void *sqrt_alpha, *sqrt_one_minus;
    TensorPrologFRUSqrt(TensorProlog_Q16, &alpha_bar_t, sqrt_alpha, 6, stream);
    TensorPrologFRUSqrt(TensorProlog_Q16, &one_minus_alpha, sqrt_one_minus, 6, stream);

    // Scale and add — exact
    TensorPrologVdrScale(TensorProlog_Q16, x_0, sqrt_alpha, term1, n_pixels, stream);
    TensorPrologVdrScale(TensorProlog_Q16, noise, sqrt_one_minus, term2, n_pixels, stream);
    TensorPrologVdrAdd(TensorProlog_Q16, term1, term2, x_t, n_pixels, stream);

    // x_t is the exact noised version at timestep t.
    // Apply the exact inverse: x_0 = (x_t - sqrt(1-a)*noise) / sqrt(a)
    // You get back EXACTLY x_0. Not approximately. Exactly.
    // This is the zero-drift denoising from paper Section VDR-26.
}
```

---

### 2.2 Reverse Diffusion — Zero-Drift Denoising

```
void reverse_step(void* model_weights, void* x_t, void* x_t_minus_1,
                   vdr_q16 alpha_t, vdr_q16 alpha_bar_t, vdr_q16 alpha_bar_t_minus_1,
                   i32 n_pixels, TensorPrologSession_t session, TensorPrologStream_t stream) {

    // Predict noise using U-Net
    unet_forward(model_weights, x_t, t, predicted_noise, session, stream);
    // U-Net is integer convolutions + attention + exact layer norm.
    // Same architecture as LLM attention but spatial instead of sequential.

    // Compute mean of reverse distribution — exact
    // mu_t = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * predicted_noise)

    vdr_q16 one_minus_alpha_t, recip_sqrt_alpha;
    one_minus_alpha_t.v = 65536 - alpha_t.v;

    TensorPrologFRUSqrt(TensorProlog_Q16, &alpha_t, &sqrt_alpha, 6, stream);
    TensorPrologVdrDiv(TensorProlog_Q16, &one_q16, &sqrt_alpha, &recip_sqrt_alpha, 1, stream);

    vdr_q16 one_minus_alpha_bar;
    one_minus_alpha_bar.v = 65536 - alpha_bar_t.v;
    TensorPrologFRUSqrt(TensorProlog_Q16, &one_minus_alpha_bar, &sqrt_one_minus_abar, 6, stream);

    // coeff = (1-alpha_t) / sqrt(1-alpha_bar_t)
    TensorPrologVdrDiv(TensorProlog_Q16, &one_minus_alpha_t, &sqrt_one_minus_abar, &coeff, 1, stream);

    // x_t - coeff * predicted_noise
    TensorPrologVdrScale(TensorProlog_Q16, predicted_noise, &coeff, scaled_noise, n_pixels, stream);
    TensorPrologVdrSub(TensorProlog_Q16, x_t, scaled_noise, adjusted, n_pixels, stream);

    // mu = recip_sqrt_alpha * adjusted
    TensorPrologVdrScale(TensorProlog_Q16, adjusted, &recip_sqrt_alpha, mu, n_pixels, stream);

    // Add scheduled noise for steps > 0 (stochastic)
    // sigma_t = known exact schedule value
    // x_{t-1} = mu + sigma_t * z, where z is integer noise
    TensorPrologVdrScale(TensorProlog_Q16, random_noise, &sigma_t, noise_term, n_pixels, stream);
    TensorPrologVdrAdd(TensorProlog_Q16, mu, noise_term, x_t_minus_1, n_pixels, stream);
}

// Full denoising loop
void denoise(void* model_weights, void* x_T, void* x_0,
              i32 n_steps, i32 n_pixels, TensorPrologSession_t session, TensorPrologStream_t stream) {

    void* x_current = x_T;
    for (i32 t = n_steps - 1; t >= 0; t--) {
        reverse_step(model_weights, x_current, x_next,
            alphas[t], alpha_bars[t], alpha_bars[t > 0 ? t-1 : 0],
            n_pixels, session, stream);
        x_current = x_next;
    }
    // x_current is x_0 — the denoised output.
    // Every intermediate step used exact arithmetic.
    // No decoherence from cumulative float drift across 1000 steps.
    // The reverse process is a more faithful inverse of the forward process
    // because both directions use the same exact arithmetic.
}
```

**The key insight from the paper (VDR-26):** Conventional diffusion has a decoherence problem — each denoising step operates on the degraded output of the previous step, and cumulative float drift means the reverse process doesn't exactly invert the forward. After 1000 steps of forward + 1000 steps of reverse, you don't get back to the original image. VDR gives exact roundtrip at sufficient remainder depth.

---

### 2.3 Diffusion Training — Exact Loss Gradients

```
void diffusion_training_step(void* unet_weights, void* x_0, i32 n_pixels,
                              TensorPrologComputeGraph_t graph, TensorPrologStream_t stream) {

    // Sample random timestep
    i32 t = random_int(0, n_timesteps);

    // Sample noise — integer noise from a discrete distribution
    // mapped to VDR values
    void* noise;
    sample_integer_noise(noise, n_pixels);

    // Forward diffuse to timestep t — exact
    forward_diffusion(x_0, noise, x_t, alpha_bars[t], n_pixels, stream);

    // Record compute graph
    TensorPrologTrainComputeGraphRecord(graph, true);

    // Predict noise — U-Net forward
    unet_forward(unet_weights, x_t, t, predicted_noise, session, stream);

    TensorPrologTrainComputeGraphRecord(graph, false);

    // MSE loss between predicted and actual noise — exact
    TensorPrologVdrSub(TensorProlog_Q16, predicted_noise, noise, diff, n_pixels, stream);
    TensorPrologVdrMul(TensorProlog_Q16, diff, diff, squared, n_pixels, stream);  // element-wise square
    TensorPrologStatsMean(TensorProlog_Q16, squared, &mse_loss, n_pixels, stream);
    // mse_loss is an exact VDR fraction.

    // Backward — exact gradients through the U-Net
    TensorPrologTrainBackwardPass(graph, &mse_loss, stream);

    // Update
    TensorPrologTrainAdamUpdate(TensorProlog_Q16, unet_weights, gradients, ...);
}
```

---

### 2.4 Classifier-Free Guidance — Exact Interpolation

```
void cfg_guided_denoise(void* model_weights, void* x_t, void* x_t_minus_1,
                          i32* conditioning, vdr_q16 guidance_scale,
                          i32 n_pixels, TensorPrologStream_t stream) {

    // Predict noise conditioned and unconditioned
    unet_forward(model_weights, x_t, t, conditioning, cond_noise, stream);
    unet_forward(model_weights, x_t, t, NULL, uncond_noise, stream);

    // Guided noise = uncond + scale * (cond - uncond)
    TensorPrologVdrSub(TensorProlog_Q16, cond_noise, uncond_noise, diff, n_pixels, stream);
    TensorPrologVdrScale(TensorProlog_Q16, diff, &guidance_scale, scaled_diff, n_pixels, stream);
    TensorPrologVdrAdd(TensorProlog_Q16, uncond_noise, scaled_diff, guided_noise, n_pixels, stream);

    // guided_noise is the exact interpolation at the declared guidance scale.
    // guidance_scale = 7.5 in conventional float.
    // In Q16: guidance_scale.v = 491520 → need Q32 for values > 1.0 at high precision.
    // Or at Q16: approximate 7.5 as 7*65536 + 32768 = 491520... but V is i32 so fine.
    // The point: the interpolation is exact at whatever Q-basis you choose.

    // Continue with standard reverse step using guided_noise
    reverse_step_with_noise(model_weights, x_t, x_t_minus_1,
        guided_noise, alphas, alpha_bars, n_pixels, stream);
}
```

---

### 2.5 Latent Diffusion — VAE Encode/Decode

```
// VAE encoder and decoder are convolutional networks in exact integer arithmetic

void vae_encode(void* encoder_weights, void* image, void* latent,
                 i32 height, i32 width, i32 channels, i32 latent_dim,
                 TensorPrologStream_t stream) {

    // Convolution layers — exact integer MAC
    for (i32 layer = 0; layer < n_encoder_layers; layer++) {
        TensorPrologTransformConv2D(TensorProlog_Q16,
            current_feature_map, conv_kernel[layer], next_feature_map,
            h, w, kernel_h, kernel_w, stream);

        // Activation — exact via integer surrogate or FRU
        // Batch norm — exact mean and variance
        TensorPrologVdrLayerNorm(TensorProlog_Q16, next_feature_map, normed,
            bn_gamma[layer], bn_beta[layer], feature_size, 0, stream);
    }

    // Latent = exact encoded representation
    // No float quantization of the latent space.
    // Decoding from this latent will recover the exact encoded features.
}

void vae_decode(void* decoder_weights, void* latent, void* reconstructed,
                 i32 latent_dim, i32 height, i32 width, i32 channels,
                 TensorPrologStream_t stream) {

    // Transpose convolutions — exact integer MAC
    // Mirror of encoder. Exact reconstruction at sufficient model capacity.
}

// Full latent diffusion pipeline:
// 1. Encode image → latent (exact)
// 2. Forward diffuse latent (exact noise addition)
// 3. Denoise in latent space (exact reverse steps)
// 4. Decode latent → image (exact)
// Roundtrip without diffusion (encode → decode) is exact.
// Roundtrip WITH diffusion depends on model quality, but arithmetic contributes zero error.
```

---

### 2.6 Deterministic Diffusion Sampling (DDIM)

```
// DDIM uses deterministic sampling — no random noise added at each step.
// In float, "deterministic" DDIM still produces slightly different results
// on different hardware. In TensorProlog, truly deterministic.

void ddim_step(void* model_weights, void* x_t, void* x_t_minus_1,
                vdr_q16 alpha_bar_t, vdr_q16 alpha_bar_t_minus_1,
                i32 n_pixels, TensorPrologStream_t stream) {

    // Predict noise
    unet_forward(model_weights, x_t, t, predicted_noise, stream);

    // Predict x_0 from x_t and predicted noise — exact
    // x_0_pred = (x_t - sqrt(1-alpha_bar_t) * noise) / sqrt(alpha_bar_t)
    // ... (exact VDR operations as in reverse_step)

    // DDIM deterministic step:
    // x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_pred + sqrt(1-alpha_bar_{t-1}) * direction
    // direction = (x_t - sqrt(alpha_bar_t) * x_0_pred) / sqrt(1-alpha_bar_t)

    // All exact. Given the same x_T and model weights:
    // DDIM produces bit-identical output on any hardware, any time, any run.
    // Conventional DDIM can't guarantee this.
}

// Use case: reproducible image generation.
// Same seed → same latent x_T → same denoising trajectory → same image.
// Not "visually identical." Pixel-identical. Bit-for-bit.
// Ship the seed as a compact representation of the output.
```

---

## Part 3: Other ML Workflows (10%)

---

### 3.1 Vision Transformer (ViT) — Image Classification

```
// Patch embedding: split image into patches, linear project
// Same GEMM + attention as LLM, different input shape

void vit_forward(void* vit_weights, void* image_patches, void* class_logits,
                  i32 n_patches, i32 patch_dim, i32 n_classes,
                  TensorPrologStream_t stream) {

    // Patch embedding — linear projection
    TensorPrologVdrGemm(TensorProlog_Q16, NO_TRANS, NO_TRANS,
        n_patches, d_model, patch_dim,
        &one_q16, image_patches, n_patches,
        patch_embed_weights, patch_dim,
        &zero_q16, embedded, n_patches, stream);

    // Add positional embedding — exact
    TensorPrologVdrAdd(TensorProlog_Q16, embedded, pos_embeddings, embedded,
        n_patches * d_model, stream);

    // Transformer layers — identical to LLM
    for (i32 layer = 0; layer < n_layers; layer++) {
        // LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
        // Same code as 1.2. Not "similar code." Same functions, same API.
    }

    // CLS token → classification head
    TensorPrologVdrGemm(TensorProlog_Q16, NO_TRANS, NO_TRANS,
        1, n_classes, d_model,
        &one_q16, cls_hidden, 1,
        classifier_weights, d_model,
        &zero_q16, class_logits, 1, stream);

    TensorPrologVdrSoftmax(TensorProlog_Q16, class_logits, class_probs, n_classes, stream);
    // Exact class probabilities summing to D.
}

// This is I-ViT's result validated: integer-only ViT achieves comparable
// accuracy to float. TensorProlog makes it the default, not a special mode.
```

---

### 3.2 Reinforcement Learning — Exact Value Estimation

```
// PPO with exact advantage estimation

void ppo_step(void* policy_weights, void* value_weights,
               void* states, void* actions, void* rewards,
               i32 n_steps, TensorPrologStream_t stream) {

    // Value function — exact
    void* values;
    TensorPrologMallocTyped(&values, TensorProlog_Q16, n_steps);
    for (i32 t = 0; t < n_steps; t++) {
        value_network_forward(value_weights, states + t * state_dim, &values[t], stream);
    }

    // GAE (Generalized Advantage Estimation) — exact accumulation
    // advantages[t] = sum_{l=0}^{T-t} (gamma*lambda)^l * delta[t+l]
    // delta[t] = reward[t] + gamma * V[t+1] - V[t]
    // Each delta is exact. Gamma and lambda are exact VDR fractions.
    // The sum accumulates exactly — no float drift over long episodes.

    void* advantages;
    TensorPrologMallocTyped(&advantages, TensorProlog_Q16, n_steps);
    vdr_q16 gae_accumulator = { .v = 0, .r0 = 0 };

    for (i32 t = n_steps - 1; t >= 0; t--) {
        // delta = reward + gamma * V[t+1] - V[t]
        TensorPrologVdrMul(TensorProlog_Q16, &gamma, &values[t+1], &gamma_v_next, 1, stream);
        TensorPrologVdrAdd(TensorProlog_Q16, &rewards[t], &gamma_v_next, &r_plus_gv, 1, stream);
        TensorPrologVdrSub(TensorProlog_Q16, &r_plus_gv, &values[t], &delta, 1, stream);

        // accumulator = delta + gamma * lambda * accumulator
        TensorPrologVdrMul(TensorProlog_Q16, &gamma_lambda, &gae_accumulator, &discounted, 1, stream);
        TensorPrologVdrAdd(TensorProlog_Q16, &delta, &discounted, &gae_accumulator, 1, stream);

        advantages[t] = gae_accumulator;
        // After 10,000 steps this accumulator has zero float drift.
        // In conventional RL, GAE over long episodes accumulates substantial error.
    }

    // PPO clipped objective — exact ratio comparison
    // ratio = pi_new(a|s) / pi_old(a|s)
    // Both are exact softmax probabilities. Division is exact VDR.
    // Clip comparison: is ratio > 1+epsilon? Exact integer comparison.
    // No "almost clipped" ambiguity at the boundary.
}
```

**Why this matters for RL:** Value estimation errors are the primary source of instability in policy gradient methods. Exact value computation means the advantage signal is clean. No noise from arithmetic — only noise from the environment, which is the signal you're trying to learn from.

---

### 3.3 Graph Neural Networks — Exact Message Passing

```
void gnn_message_pass(void* node_features, void* edge_index,
                        void* edge_weights, void* updated_features,
                        i32 n_nodes, i32 n_edges, i32 feature_dim,
                        TensorPrologStream_t stream) {

    // For each node: aggregate neighbor features weighted by edge weight
    // Conventional GNNs accumulate float error proportional to node degree.
    // High-degree nodes (hubs) get noisier representations than low-degree nodes.
    // TensorProlog: all aggregations exact. Hub nodes and leaf nodes computed with
    // identical precision.

    for (i32 node = 0; node < n_nodes; node++) {
        vdr_q16 aggregated[MAX_FEATURE_DIM] = {0};
        for (i32 e = 0; e < node_degree[node]; e++) {
            i32 neighbor = neighbors[node][e];
            // Scale neighbor features by edge weight — exact
            TensorPrologVdrScale(TensorProlog_Q16,
                node_features + neighbor * feature_dim,
                &edge_weights[edge_id],
                scaled_neighbor, feature_dim, stream);
            // Accumulate — exact
            TensorPrologVdrAdd(TensorProlog_Q16, aggregated, scaled_neighbor,
                aggregated, feature_dim, stream);
        }
        // Normalize by degree — exact division
        vdr_q16 degree_vdr = { .v = node_degree[node] * 65536, .r0 = 0 };
        TensorPrologVdrDiv(TensorProlog_Q16, aggregated, &degree_vdr,
            updated_features + node * feature_dim, feature_dim, stream);
    }
    // Use graph builtins for structure operations:
    TensorPrologBuiltinGraphPageRankExact(TensorProlog_Q16, adjacency, ranks, n_iters, stream);
    // PageRank as exact VDR fractions. Converges to exact steady state.
}
```

---

### 3.4 Time Series Forecasting — Exact Recurrence

```
void lstm_forward(void* weights, void* input_seq, void* output_seq,
                   i32 seq_len, i32 input_dim, i32 hidden_dim,
                   TensorPrologStream_t stream) {

    vdr_q16 h[MAX_HIDDEN], c[MAX_HIDDEN];
    memset(h, 0, hidden_dim * sizeof(vdr_q16));
    memset(c, 0, hidden_dim * sizeof(vdr_q16));

    for (i32 t = 0; t < seq_len; t++) {
        // Gates: i, f, g, o = sigmoid/tanh of linear combinations
        // All computed via FRU exact transcendentals or integer surrogates

        // input gate
        TensorPrologVdrGemm(TensorProlog_Q16, ..., input[t], W_i, ..., gate_i_pre, stream);
        // sigmoid via FRU or rational approximation
        TensorPrologFRUExp(TensorProlog_Q16, negated_gate_i_pre, exp_neg, 6, stream);
        // sigmoid = 1 / (1 + exp(-x)) — exact

        // Cell update: c = f*c + i*g — exact
        TensorPrologVdrMul(TensorProlog_Q16, f_gate, c, fc, hidden_dim, stream);
        TensorPrologVdrMul(TensorProlog_Q16, i_gate, g_gate, ig, hidden_dim, stream);
        TensorPrologVdrAdd(TensorProlog_Q16, fc, ig, c, hidden_dim, stream);

        // Hidden: h = o * tanh(c) — exact
        // After 10,000 timesteps: zero accumulated drift in hidden state.
        // Conventional LSTMs: hidden state drifts from compounding float error,
        // contributing to the "forgetting" problem beyond what the architecture intends.
    }
}
```

---

### 3.5 Embedding Search — Exact Similarity

```
void exact_cosine_similarity(void* embeddings_a, void* embeddings_b,
                              void* similarities, i32 n_pairs, i32 dim,
                              TensorPrologStream_t stream) {

    for (i32 i = 0; i < n_pairs; i++) {
        vdr_q16 dot, norm_a, norm_b;

        TensorPrologVdrDot(TensorProlog_Q16,
            embeddings_a + i * dim, embeddings_b + i * dim,
            &dot, dim, stream);

        TensorPrologVdrDot(TensorProlog_Q16,
            embeddings_a + i * dim, embeddings_a + i * dim,
            &norm_a_sq, dim, stream);

        TensorPrologVdrDot(TensorProlog_Q16,
            embeddings_b + i * dim, embeddings_b + i * dim,
            &norm_b_sq, dim, stream);

        TensorPrologFRUSqrt(TensorProlog_Q16, &norm_a_sq, &norm_a, 6, stream);
        TensorPrologFRUSqrt(TensorProlog_Q16, &norm_b_sq, &norm_b, 6, stream);

        TensorPrologVdrMul(TensorProlog_Q16, &norm_a, &norm_b, &norm_product, 1, stream);
        TensorPrologVdrDiv(TensorProlog_Q16, &dot, &norm_product, &similarities[i], 1, stream);
    }

    // Similarity is exact VDR fraction.
    // Ranking by similarity: sort by exact integer comparison.
    // No "these two documents have the same similarity score due to float rounding"
    // producing unstable rankings. If they're different, the ranking is stable.
    TensorPrologBuiltinSort(TensorProlog_Q16, similarities, n_pairs);
}
```

---

### 3.6 Quantization-Aware Training — Unnecessary

```
// This section is deliberately empty.
//
// In conventional ML:
// - Train in FP32
// - Quantize to INT8 (lossy)
// - Calibrate with representative dataset
// - Fine-tune to recover accuracy lost in quantization
// - Deploy quantized model
// - Hope the calibration held for production data
//
// In TensorProlog:
// - Train in Q16 integers
// - Deploy in Q16 integers
//
// Quantization-aware training doesn't exist because there's no quantization.
// The training representation IS the deployment representation.
// No calibration step. No accuracy recovery. No QAT, GPTQ, AWQ, SmoothQuant,
// GGUF, GGML, or any other quantization format.
//
// The entire quantization subfield of ML engineering is eliminated.
// Not simplified. Eliminated.
```

---

### 3.7 A/B Testing Models — Zero Noise Floor

```
void ab_test(void* model_a, void* model_b, eval_dataset_t* eval) {

    i32 correct_a = 0, correct_b = 0;

    for (i32 i = 0; i < eval->n_examples; i++) {
        // Both models, same input, same code path
        i32 pred_a = predict(model_a, eval->examples[i]);
        i32 pred_b = predict(model_b, eval->examples[i]);

        if (pred_a == eval->examples[i].target) correct_a++;
        if (pred_b == eval->examples[i].target) correct_b++;
    }

    // Accuracy A: correct_a / n_examples — exact fraction
    // Accuracy B: correct_b / n_examples — exact fraction
    // Comparison: exact integer comparison
    i32 cmp;
    TensorPrologVdrCompare(TensorProlog_Q16, &accuracy_a, &accuracy_b, &cmp, 1, stream);

    // If cmp > 0: model A is better. Period.
    // Not "model A is better with p < 0.05 after accounting for evaluation noise."
    // The evaluation has zero arithmetic noise. The only variance is from
    // the test set itself. Statistical testing is still needed for generalization
    // claims, but the evaluation numbers are exact — you're testing the model,
    // not the arithmetic.

    // Run this test 100 times: same numbers, 100 times.
    // In conventional A/B testing, float nondeterminism adds variance
    // to the evaluation itself, requiring larger test sets to achieve
    // statistical significance. TensorProlog evaluation variance is zero.
    // Every sample counts at full weight.
}
```

---

### 3.8 Distillation — Exact Teacher Signals

```
void distillation_step(void* teacher_weights, void* student_weights,
                        i32* input_ids, i32 seq_len, vdr_q16 temperature,
                        vdr_q16 alpha, TensorPrologStream_t stream) {

    // Teacher forward — exact soft targets
    forward_pass(teacher_weights, input_ids, seq_len, teacher_logits, stream);
    // Divide logits by temperature — exact
    TensorPrologVdrDiv(TensorProlog_Q16, teacher_logits, &temperature,
        teacher_scaled, seq_len * vocab_size, stream);
    TensorPrologVdrSoftmax(TensorProlog_Q16, teacher_scaled, teacher_soft_targets,
        vocab_size, stream);
    // Teacher soft targets: exact probability distribution.
    // The dark knowledge (non-peak probabilities) is represented exactly.
    // In float distillation, small probabilities in the teacher's output
    // get rounded to zero or near-zero, losing the inter-class relationships
    // that make distillation work. In TensorProlog, if the teacher assigns
    // probability 3/65536 to class 47, the student sees exactly 3/65536.

    // Student forward
    forward_pass(student_weights, input_ids, seq_len, student_logits, stream);
    TensorPrologVdrDiv(TensorProlog_Q16, student_logits, &temperature,
        student_scaled, seq_len * vocab_size, stream);
    TensorPrologVdrSoftmax(TensorProlog_Q16, student_scaled, student_probs, vocab_size, stream);

    // KL divergence loss — exact
    // KL(teacher || student) = sum(teacher * log(teacher/student))
    // All exact VDR operations. The distillation signal is clean.

    // Combined loss: alpha * KL_loss + (1-alpha) * CE_loss
    // alpha is exact fraction. Weighting is exact.
    TensorPrologTrainBackwardPass(graph, &combined_loss, stream);
    TensorPrologTrainAdamUpdate(TensorProlog_Q16, student_weights, gradients, ...);
}
```

---

### 3.9 Federated Learning — Deterministic Aggregation

```
// N clients train locally, send updates to server, server aggregates

// Client side:
void federated_client_train(void* local_weights, train_data_t* local_data,
                             void* weight_delta, TensorPrologStream_t stream) {

    // Copy global weights
    TensorPrologMemcpy(local_weights, global_weights, n_params * 8, DeviceToDevice);

    // Train locally for K steps
    for (i32 step = 0; step < K; step++) {
        local_training_step(local_weights, local_data, stream);
    }

    // Compute delta: local - global — exact
    TensorPrologVdrSub(TensorProlog_Q16, local_weights, global_weights, weight_delta,
        n_params, stream);

    // Send weight_delta to server. Integers. Bit-identical transmission.
    // No "close enough" — the server receives exactly what the client computed.
}

// Server side:
void federated_aggregate(void** client_deltas, i32 n_clients,
                           void* global_weights, TensorPrologStream_t stream) {

    // Average deltas — exact
    void* sum_delta;
    TensorPrologMallocTyped(&sum_delta, TensorProlog_Q16, n_params);
    TensorPrologMemset(sum_delta, 0, n_params * 8);

    for (i32 c = 0; c < n_clients; c++) {
        TensorPrologVdrAdd(TensorProlog_Q16, sum_delta, client_deltas[c], sum_delta,
            n_params, stream);
    }

    // Divide by n_clients — exact
    vdr_q16 n_clients_vdr = { .v = n_clients * 65536, .r0 = 0 };
    TensorPrologVdrDiv(TensorProlog_Q16, sum_delta, &n_clients_vdr, avg_delta, n_params, stream);

    // Apply to global model
    TensorPrologVdrAdd(TensorProlog_Q16, global_weights, avg_delta, global_weights, n_params, stream);

    // Deterministic aggregation means:
    // - Same client updates in any order → same global model
    // - Verifiable: any observer can recompute the aggregation and get bit-identical result
    // - Auditable: the exact contribution of each client is traceable
    // This is confidential computing on exact arithmetic — the compliance story writes itself.
}
```

---

### 3.10 Workflow Summary Table

| Workflow | Key TensorProlog Modules | What's Eliminated |
|---|---|---|
| LLM inference | vdr_math, attention, session | Quantization, mixed-precision config |
| LLM training | training, vdr_math, distributed | Loss scaling, gradient clipping, warmup, NaN recovery |
| KV-cache | session, kb | KV-cache quantization, cache eviction heuristics |
| Distributed training | distributed, training | Non-deterministic allreduce, gradient sync bugs |
| LoRA fine-tuning | training, vdr_math | Merge precision loss, QLoRA |
| Serving | session, safety, grammar | Guardrail frameworks, output validation |
| Full VDR-LLM-Prolog | All modules | 85-97% of LLM tokens |
| Speculative decoding | vdr_math, attention | Float boundary errors in accept/reject |
| MoE routing | vdr_math, attention | Non-deterministic expert assignment |
| RLHF/DPO | training, functional_remainder | Noisy preference gradients |
| Model merging | vdr_math | Lossy merge artifacts |
| Diffusion inference | vdr_math, functional_remainder | Cumulative denoising drift |
| Diffusion training | training, transform | Noise schedule float error |
| CFG guidance | vdr_math | Interpolation imprecision |
| Latent diffusion | transform, vdr_math | VAE roundtrip loss |
| DDIM | vdr_math | "Deterministic" sampling that isn't |
| ViT | vdr_math, attention | Same as LLM — shared architecture |
| RL / PPO | vdr_math, functional_remainder | GAE accumulation drift |
| GNN | vdr_math, builtins (graph) | Degree-dependent precision loss |
| Time series | vdr_math, functional_remainder | Hidden state drift |
| Embedding search | vdr_math, builtins (sort) | Unstable similarity rankings |
| QAT | — | The entire subfield |
| A/B testing | vdr_math, profiling | Evaluation arithmetic noise |
| Distillation | training, vdr_math | Dark knowledge truncation |
| Federated learning | distributed, training, safety | Non-deterministic aggregation, audit gaps |
