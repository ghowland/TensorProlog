```
================================================================================
VLP Single-Kernel GPU Architecture — Technical Specification
================================================================================

WHAT THIS IS
────────────
One Zig SPIR-V kernel that runs the entire GPU compute layer. The host loads
one .spv binary, creates one Vulkan compute pipeline, and dispatches it
repeatedly with different op_codes and parameters. The kernel is a router
that switches on op_code and calls the appropriate subsystem function.

WHAT IT INCLUDES
────────────────
1. vlp_gpu_shared.zig — SPIR-V-safe shared definitions
   - All pub const values (FACTS_PER_KB, FACT_INTS, KB_STRUCT_SIZE, etc)
   - All extern struct types that appear in GPU buffers
     (Q16, Fact, Kb, Term, Rule, Binding)
   - All enum types used by GPU (FactTag, TermType, SourceType, OpCode)
   - Confidence table as comptime array
   - Descriptor set binding index constants
   - NO functions. NO std imports. NO allocators. Pure declarations.
   - Compiles for both spirv32-vulkan and native x86_64.
   - This file is the single source of truth for the host-device contract.

2. vlp_kernel.zig — The kernel
   - One export fn main() callconv(.spirv_kernel)
   - Reads op_code from uniform params buffer
   - Switches to one of 28 operations
   - Each operation is a fn within this file (or inlined)
   - All buffer access via fixed-size arrays in extern structs
   - All loops bounded by params or compile-time constants
   - No recursion. No dynamic allocation. No std.
   - Imports vlp_gpu_shared.zig for types and constants.
   - Compiles to one .spv file.

3. vlp_types.zig — Host types (MODIFIED)
   - Imports vlp_gpu_shared.zig, re-exports all shared types
   - Adds host-only types (Session, Runner, Grant, Command, AuditEntry, etc)
   - Adds host-only functions (Q16 arithmetic methods, Status helpers, etc)
   - Adds LevelStats, SessionHandle, SnapshotHandle, RunnerHandle
   - GPU-shared types have NO methods in vlp_gpu_shared.zig.
     Methods are added in vlp_types.zig for host use only.

4. vlp_bridge.zig — Vulkan bridge (SIMPLIFIED)
   - One VkShaderModule (loaded from vlp_kernel.spv)
   - One VkPipeline
   - One VkPipelineLayout
   - One set of VkDescriptorSetLayouts (4 sets, same as before)
   - dispatch() writes op_code + params to uniform buffer, dispatches
   - No pipeline selection. No pipeline array. PipelineId enum removed.
   - OpCode enum (from vlp_gpu_shared.zig) replaces PipelineId.

5. vlp_gpu_params.zig — Dispatch parameters (RESTRUCTURED)
   - Common header: op_code (i32) + reserved (3 × i32)
   - Followed by per-op params as a flat region
   - Host writes header + op-specific params to uniform buffer
   - Kernel reads header, switches, then reads op-specific fields
   - All params in one extern struct with op_code as first field
   - Per-op params are a union in host code, flat bytes on GPU

6. 20 host modules — UNCHANGED in interface
   - vlp_llm.zig, vlp_kb_store.zig, vlp_prolog.zig, vlp_grammar.zig,
     vlp_builtin.zig, vlp_confidence.zig, vlp_session.zig,
     vlp_snapshot.zig, vlp_runner.zig, vlp_grant.zig, vlp_access.zig,
     vlp_audit.zig, vlp_command.zig, vlp_inference.zig, vlp_seed.zig,
     vlp_system.zig, vlp_multi_device.zig, vlp_test.zig,
     vlp_device_memory.zig
   - Each calls bridge.dispatch(op_code, params) instead of
     bridge.dispatch(pipeline_id, params)
   - Internal logic unchanged. Same functions, same signatures.

WHAT IT DOES NOT INCLUDE
────────────────────────
1. No GLSL. All GPU code is Zig compiled to SPIR-V.

2. No multiple pipelines. One pipeline, one shader module. The op_code
   switch is the dispatch mechanism. Vulkan pipeline switching overhead
   is eliminated.

3. No runtime kernel loading. The .spv is embedded in the host binary
   at compile time via @embedFile. No filesystem access needed at runtime
   to load shaders.

4. No specialization constants. FACTS_PER_KB and similar are pub const
   in vlp_gpu_shared.zig, resolved at Zig compile time. Changing them
   requires recompiling vlp_kernel.zig (one file, fast).

5. No tiled GEMM with shared memory (initially). The first implementation
   uses the naive per-element GEMM pattern (one invocation per output
   element, full dot product in a loop). Tiled GEMM with shared memory
   is an optimization that can be added to the existing functions without
   changing the dispatch interface.

6. No multi-kernel dispatch sequences. The host dispatches one kernel
   at a time with pipeline barriers between. dispatchSequence() still
   records multiple dispatches into one command buffer, but they all
   invoke the same pipeline with different op_codes and params.

7. No async I/O or evented GPU work. All dispatches are synchronous
   (submit + fence + wait). Async dispatch is a future optimization.

8. No Zig SPIR-V features beyond the working subset:
   - No [*]addrspace(.storage_buffer) pointer arithmetic
   - No std library functions
   - No error unions
   - No allocators
   - No recursion
   - Only: extern struct, fixed-size arrays, integer arithmetic,
     std.gpu builtins, callconv(.spirv_kernel), addrspace(),
     @setExecProperty, basic control flow (if/switch/while/for)

FILE MANIFEST
─────────────
GPU side (compiled to SPIR-V):
  src/vlp_gpu_shared.zig    — shared constants + extern structs
  src/vlp_kernel.zig        — the kernel (one entry point, 28 ops)

Host side (compiled to native):
  src/vlp_gpu_shared.zig    — same file, imported by vlp_types.zig
  src/vlp_types.zig         — host types + re-exported shared types
  src/vlp_gpu_params.zig    — dispatch param structs with op_code header
  src/vlp_bridge.zig        — single-pipeline Vulkan bridge
  src/vlp_device_memory.zig — memory layout + sizing
  src/vlp_llm.zig           — LLM forward pass orchestration
  src/vlp_kb_store.zig      — KB store engine
  src/vlp_prolog.zig        — Prolog engine
  src/vlp_grammar.zig       — grammar engine (host-only)
  src/vlp_builtin.zig       — builtin executor
  src/vlp_confidence.zig    — confidence propagation
  src/vlp_session.zig       — session manager
  src/vlp_snapshot.zig      — snapshot manager
  src/vlp_runner.zig        — runner scheduler
  src/vlp_grant.zig         — grant enforcer
  src/vlp_access.zig        — access control
  src/vlp_audit.zig         — audit log
  src/vlp_command.zig       — command processor
  src/vlp_inference.zig     — inference loop
  src/vlp_seed.zig          — seed layer
  src/vlp_system.zig        — top-level wiring
  src/vlp_multi_device.zig  — multi-device support
  src/vlp_test.zig          — test infrastructure

Build:
  build.zig                 — compiles vlp_kernel.zig to .spv,
                              embeds in host binary, compiles host

Total: 24 source files. 1 build file.

BUILD CHAIN
───────────
  vlp_gpu_shared.zig ──┬── vlp_kernel.zig ── zig build-obj spirv32-vulkan ── kernel.spv
                       │                                                        │
                       │                                                   @embedFile
                       │                                                        │
                       └── vlp_types.zig ── all host modules ── zig build native ── executable

The build system compiles vlp_kernel.zig as a separate build step targeting
spirv32-vulkan, producing kernel.spv. The host executable embeds kernel.spv
via addAnonymousImport. One zig build command produces the final binary
with the kernel baked in.

BUFFER LAYOUT (UNCHANGED)
─────────────────────────
Set 0 — Model (read-only, bound once)
  binding 0: embedding_table   [vocab_size × d_model] i32
  binding 1: layer_weights     [n_layers × layer_stride] i32
  binding 2: lm_head           [d_model × vocab_size] i32
  binding 3: layer_norm_params [n_layers × 2 × d_model] i32

Set 1 — KB Data (per session)
  binding 0: kb_store          [max_kbs × KB_STRUCT_INTS] i32
  binding 1: fact_store        [max_kbs × FACTS_PER_KB × FACT_INTS] i32
  binding 2: rule_store        [max_rules × RULE_INTS] i32
  binding 3: term_store        [max_terms × TERM_INTS] i32
  binding 4: live_state        [live_state_size] i32

Set 2 — Scratch (per dispatch)
  binding 0: scratch_a         [scratch_size] i32
  binding 1: scratch_b         [scratch_size] i32
  binding 2: kv_cache          [kv_cache_size] i32

Set 3 — Control (per dispatch)
  binding 0: params            uniform buffer [256 bytes]
  binding 1: status_buffer     [max_invocations] i32
  binding 2: result_counts     [16] i32

All buffers declared in vlp_kernel.zig as extern struct with fixed-size
arrays. Array sizes are pub const from vlp_gpu_shared.zig.

DISPATCH PROTOCOL
─────────────────
Host:
  1. Write op_code + per-op params to uniform buffer (Set 3, binding 0)
  2. Reset status_buffer and result_counts if needed
  3. Begin command buffer
  4. Bind pipeline (always the same one)
  5. Bind descriptor sets
  6. vkCmdDispatch(group_x, group_y, group_z)
  7. Pipeline barrier if another dispatch follows
  8. End command buffer
  9. Submit + fence

Kernel:
  1. Read op_code from params
  2. Switch to operation
  3. Read op-specific params
  4. Compute using global_invocation_id as work index
  5. Read from input buffers (Set 0/1/2 depending on op)
  6. Write to output buffers (Set 1/2 depending on op)
  7. Write errors to status_buffer if any
  8. Atomic increment result_counts if applicable

OP_CODE TABLE
─────────────
  0  embedding_lookup
  1  layer_norm
  2  qkv_project
  3  attention_scores
  4  softmax_exact
  5  attention_weighted_sum
  6  output_project
  7  mlp
  8  lm_head
  9  kv_cache_append
  10 residual_add
  11 fact_write_batch
  12 fact_read_batch
  13 fact_scan_by_tag
  14 scoped_search
  15 unify_candidates
  16 rule_match_scan
  17 rule_body_eval
  18 rule_check_satisfied
  19 builtin_unary
  20 builtin_binary
  21 builtin_reduction
  22 builtin_sort
  23 builtin_matmul
  24 confidence_combine
  25 confidence_chain
  26 buffer_copy
  27 buffer_fill

INVARIANTS (ALL PRESERVED)
──────────────────────────
All 15 invariants from the CPU-GPU integration spec hold. The single-kernel
architecture does not weaken any guarantee. The kernel is still stateless
(invariant 11), non-recursive (12), non-allocating (13). Integer data
crosses the boundary identically (14). Every op has bounded execution (15).
Softmax still sums to D exactly (1). All VDR arithmetic is identical (8,9).

PERFORMANCE CONSIDERATIONS
──────────────────────────
Single pipeline means zero pipeline switching overhead. The GPU stays in
the same shader between dispatches — only the uniform buffer changes.

The switch statement in the kernel entry point adds one branch per dispatch.
On GPU this is effectively free — all invocations take the same branch
(uniform control flow from the uniform buffer param).

The downside: the kernel binary is larger (all 28 ops compiled in). This
may increase instruction cache pressure on GPUs with small I-cache.
For H100/L4/T4 with large I-cache, this is not a concern. For mobile
GPUs it could matter — but those are not the target.

Workgroup size is set per-dispatch via @setExecProperty or via
specialization constant. All ops currently use 256. If an op needs
different workgroup size, it checks op_code and adjusts. This is a
compile-time decision within the kernel, not a host decision.

KNOWN RISKS
───────────
1. Zig SPIR-V backend may reject the switch statement with 28 cases
   if it generates control flow the backend can't handle. Mitigation:
   test incrementally, adding ops one at a time.

2. Fixed-size array declarations may cause spirv-val warnings about
   array bounds not matching buffer size. These are warnings, not errors.
   Vulkan runtime ignores them.

3. Shared memory (addrspace(.shared)) usage for softmax and reduction
   may not work in the Zig backend. If it doesn't, those ops fall back
   to scratch buffer round-trips (slower but functional).

4. @setExecProperty may not accept runtime-variable local_size.
   If it requires comptime, workgroup size is fixed at 256 for all ops.
   This is fine for most ops. Reduction ops that want a single workgroup
   still work — they just have idle invocations.

5. int64_t (i64) support requires the Int64 SPIR-V capability.
   All datacenter GPUs (T4, L4, H100) support it. Mobile GPUs may not.
   The kernel uses i64 extensively for Q16 multiply accumulators.
   No workaround exists — i64 is mandatory for exact VDR arithmetic.

WHAT TO BUILD FIRST
───────────────────
1. vlp_gpu_shared.zig — constants and extern structs
2. vlp_kernel.zig with ONLY op_code 27 (buffer_fill) — simplest op
3. build.zig that compiles kernel to .spv and embeds in host
4. Minimal vlp_bridge.zig that loads .spv, creates pipeline, dispatches
5. Test: fill a buffer with a known value, read back, verify
6. Add ops one at a time: residual_add (10), then fact_read_batch (12),
   then softmax_exact (4), etc. Test each before adding the next.
7. The host modules (vlp_llm.zig etc) don't change — they just call
   bridge.dispatch with the op_code instead of a pipeline ID.
```
