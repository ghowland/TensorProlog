```
================================================================================
FIND AND REPLACE — all 20 host modules
================================================================================

vlp_gpu_params.zig:
────────────────────
DELETE entire PipelineId enum.

ADD to vlp_gpu_shared.zig (or vlp_gpu_params.zig if keeping separate):

  pub const OpCode = enum(i32) {
      embedding_lookup = 0,
      layer_norm = 1,
      qkv_project = 2,
      attention_scores = 3,
      softmax_exact = 4,
      attention_weighted_sum = 5,
      output_project = 6,
      mlp = 7,
      lm_head = 8,
      kv_cache_append = 9,
      residual_add = 10,
      fact_write_batch = 11,
      fact_read_batch = 12,
      fact_scan_by_tag = 13,
      scoped_search = 14,
      unify_candidates = 15,
      rule_match_scan = 16,
      rule_body_eval = 17,
      rule_check_satisfied = 18,
      builtin_unary = 19,
      builtin_binary = 20,
      builtin_reduction = 21,
      builtin_sort = 22,
      builtin_matmul = 23,
      confidence_combine = 24,
      confidence_chain = 25,
      buffer_copy = 26,
      buffer_fill = 27,
  };

vlp_bridge.zig:
────────────────

FIND:  pipelines: [gpu.PipelineId.count]VkPipeline,
REPL:  pipeline: VkPipeline,

FIND:  pipeline_layouts: [gpu.PipelineId.count]VkPipelineLayout,
REPL:  pipeline_layout: VkPipelineLayout,

FIND:  shader_modules: [gpu.PipelineId.count]VkShaderModule,
REPL:  shader_module: VkShaderModule,

DELETE: entire PipelineId.count references

FIND (in DispatchConfig):
    pipeline: gpu.PipelineId,
REPL:
    op_code: gpu.OpCode,

FIND (in dispatch fn body, conceptual):
    // 4. Bind pipeline[config.pipeline]
REPL:
    // 4. Bind pipeline (always the same one)

vlp_llm.zig:
─────────────

FIND:  .pipeline = .embedding_lookup,
REPL:  .op_code = .embedding_lookup,

FIND:  .pipeline = .layer_norm,
REPL:  .op_code = .layer_norm,

FIND:  .pipeline = .qkv_project,
REPL:  .op_code = .qkv_project,

FIND:  .pipeline = .attention_scores,
REPL:  .op_code = .attention_scores,

FIND:  .pipeline = .softmax_exact,
REPL:  .op_code = .softmax_exact,

FIND:  .pipeline = .attention_weighted_sum,
REPL:  .op_code = .attention_weighted_sum,

FIND:  .pipeline = .output_project,
REPL:  .op_code = .output_project,

FIND:  .pipeline = .mlp,
REPL:  .op_code = .mlp,

FIND:  .pipeline = .lm_head,
REPL:  .op_code = .lm_head,

FIND:  .pipeline = .kv_cache_append,
REPL:  .op_code = .kv_cache_append,

FIND:  .pipeline = .residual_add,
REPL:  .op_code = .residual_add,

vlp_kb_store.zig:
──────────────────

FIND:  .pipeline = .fact_write_batch,
REPL:  .op_code = .fact_write_batch,

FIND:  .pipeline = .fact_read_batch,
REPL:  .op_code = .fact_read_batch,

FIND:  .pipeline = .fact_scan_by_tag,
REPL:  .op_code = .fact_scan_by_tag,

vlp_prolog.zig:
────────────────

FIND:  .pipeline = .unify_candidates,
REPL:  .op_code = .unify_candidates,

FIND:  .pipeline = .rule_match_scan,
REPL:  .op_code = .rule_match_scan,

vlp_builtin.zig:
─────────────────

FIND:  .pipeline = .builtin_unary,
REPL:  .op_code = .builtin_unary,

FIND:  .pipeline = .builtin_binary,
REPL:  .op_code = .builtin_binary,

FIND:  .pipeline = .builtin_reduction,
REPL:  .op_code = .builtin_reduction,

FIND:  .pipeline = .builtin_sort,
REPL:  .op_code = .builtin_sort,

FIND:  .pipeline = .builtin_matmul,
REPL:  .op_code = .builtin_matmul,

FIND:  .pipeline = .builtin_confidence_combine,
REPL:  .op_code = .confidence_combine,

FIND:  .pipeline = .builtin_confidence_chain,
REPL:  .op_code = .confidence_chain,

================================================================================
GLOBAL across all files:
================================================================================

FIND:  gpu.PipelineId
REPL:  gpu.OpCode

FIND:  PipelineId.count
REPL:  (delete — no longer needed, single pipeline)

FIND:  PipelineId
REPL:  OpCode
```
