# VLP CPU-GPU Integration Technical Specification

## How the Host Adapts to the Device Contract

### Version 0.2 — Post SPIR-V Conformance Revision

---

## 1. Governing Principle

The GPU is the constraint. The CPU adapts.

Every design decision in this spec follows from one fact: SPIR-V compute shaders on Vulkan have hard architectural limits that cannot be worked around. The CPU has no such limits. Therefore when a conflict exists between what the original spec assumed and what SPIR-V allows, the CPU side changes. The GPU side does not.

The 7-turn implementation defines 22 Zig modules totaling ~2,400 lines. Those modules ARE the system API. This spec documents how the host orchestration from the original spec (Sections 5, 7-17) maps onto that API, where the mapping is direct, and where the original spec's assumptions required restructuring.

---

## 2. What Moved and Why

The original spec (Section 6) defined five "device engines" that ran autonomously on GPU. The SPIR-V conformance review identified that three of those engines contained operations incompatible with GPU compute:

| Original Device Engine | What Moved to Host | Why |
|---|---|---|
| LLM Engine | Sampling, token classification | Serial scan, branchy — wrong workload for GPU |
| KB Store Engine | KB creation, path index, COW management, children/mounts, text store | Hash maps, string keys, page table manipulation, tree walks |
| Prolog Engine | Search tree management, backtracking, rule CRUD, term parsing | Recursive backtracking, variable-depth control flow |
| Grammar Engine | **Entire engine** | Serial text manipulation, string concatenation, formatting |
| Builtin Executor | Dispatch decision, IOSE validation, operational builtins | Function pointer dispatch, grant checks, filesystem/network I/O |

What stayed on GPU:

| GPU Kernel | What It Does | Why It Belongs on GPU |
|---|---|---|
| LLM forward pass (12 kernels) | GEMM, attention, softmax, layernorm, MLP, residual | Embarrassingly parallel matrix arithmetic |
| Fact write/read batch | Parallel bulk writes and reads to fact store | Coalesced memory access, many independent writes |
| Fact scan by tag | Parallel scan of fact array for matching tags | Each invocation checks one fact independently |
| Scoped search | Parallel scan across pre-computed chain | Host pre-computes chain, GPU scans in parallel |
| Unify candidates | Parallel unification of query against N facts | Each unification is independent — one per invocation |
| Rule match scan | Parallel check of rule heads against query | Each rule tested independently |
| Rule body eval | Parallel evaluation of body conditions | Each condition checked independently |
| Rule check satisfied | Reduction: AND across body results per rule | Parallel reduction, one per matched rule |
| Builtin unary/binary/reduction/sort/matmul | Element-wise and reduction math on Q16 arrays | Embarrassingly parallel integer arithmetic |
| Confidence combine/chain | Product/complement operations on Q16 arrays | Parallel multiply-reduce |
| Buffer copy/fill | Utility memory operations | GPU memory bandwidth |

---

## 3. The Host-Device Boundary

### 3.1 Data Crosses the Boundary as Typed Integers

Every byte that crosses between CPU and GPU is part of an `extern struct` defined in `vlp_types.zig`. These structs have explicit padding, fixed sizes, and no pointers. They are the contract.

| Type | Size (bytes) | Crosses Boundary | Direction |
|---|---|---|---|
| `Q16` | 8 | Yes | Both |
| `Fact` | 40 | Yes | Both |
| `Kb` | 256 | Yes | Both |
| `Term` | 24 | Yes | Both |
| `Rule` | 48 | Yes | Both |
| `Binding` | 8 | Yes | GPU→Host |
| `Grammar` | 28 | Yes | Host→GPU (store), GPU→Host (read) |
| `Session` | 128 | No | Host only (device memory for snapshot) |
| `Runner` | 72 | No | Host only |
| `Grant` | 48 | No | Host only |
| `AuditEntry` | 32 | No | Host only |
| `Command` | 24 | No | Host only |
| All GPU dispatch params | 16-32 | Yes | Host→GPU (uniform buffer) |

The rule: if a struct touches a Vulkan storage buffer, it is `extern struct` with explicit layout. If it lives only in host memory, it can be a regular Zig `struct`.

### 3.2 The Bridge is the Only Crossing Point

No module other than `vlp_bridge.zig` touches Vulkan. Every GPU interaction goes through the bridge:

```
Host module → bridge.dispatch() → GPU kernel
Host module → bridge.uploadToBuffer() → GPU storage buffer
Host module ← bridge.downloadFromBuffer() ← GPU storage buffer
Host module → bridge.writeParams() → GPU uniform buffer
Host module ← bridge.readStatus() ← GPU status buffer
Host module ← bridge.readResultCount() ← GPU atomic counter
```

The bridge owns all `VkBuffer`, `VkPipeline`, `VkDescriptorSet`, `VkCommandBuffer`, and `VkFence` handles. No other module allocates or frees Vulkan resources.

### 3.3 The Bridge Decides GPU vs Host

`bridge.shouldUseGpu(op, data_size)` returns a boolean. The thresholds:

| Operation | GPU threshold | Rationale |
|---|---|---|
| `llm_forward` | Always GPU | No question — this is the entire point |
| `fact_scan` | > 256 facts | GPU dispatch overhead ~10μs, CPU scans ~1 fact/10ns |
| `unification` | > 32 candidates | One warp = 32 invocations = 32 candidates |
| `rule_match` | > 64 rules | Two warps minimum for amortized dispatch |
| `builtin_array` | > 512 elements | Amortize dispatch + transfer overhead |
| `text_grammar` | Never GPU | Serial text manipulation |
| `access_check` | Never GPU | 3-10 integer comparisons, nanoseconds |
| `sampling` | > 256K vocab AND batch > 1 | Single-token sampling is a scan, CPU is fine |

These thresholds are configurable via the bridge but default to values that make the dispatch overhead worthwhile. Below threshold, the host module runs the same algorithm on CPU using the same integer arithmetic from `vlp_types.zig`. The results are identical — same types, same operations, same determinism.

---

## 4. Descriptor Set Architecture

Original spec did not address descriptor sets. The implementation uses 4 sets organized by update frequency:

```
Set 0 — Model weights (bound once at model load)
  binding 0: embedding_table
  binding 1: layer_weights
  binding 2: lm_head
  binding 3: layer_norm_params

Set 1 — KB data (bound per session switch)
  binding 0: kb_store
  binding 1: fact_store
  binding 2: rule_store
  binding 3: term_store
  binding 4: live_state

Set 2 — Scratch (bound per dispatch or reused)
  binding 0: scratch_a
  binding 1: scratch_b
  binding 2: kv_cache

Set 3 — Control (written per dispatch)
  binding 0: dispatch_params (uniform buffer)
  binding 1: status_buffer
  binding 2: result_counts
```

The original spec's `vlp_device_memory_layout` maps 1:1 to these buffers. Each region in the layout becomes a `VkBuffer` in the bridge. The bridge's `updateKbDescriptors()` is called when the active session changes, rebinding Set 1 to point at the new session's region. Sets 0 and 3 rarely change. Set 2 is recycled between dispatches.

---

## 5. How Each Original Spec Section Maps

### 5.1 Session Manager (Original Section 5.1)

**Original:** `vlp_session_create` allocates device memory, creates TensorProlog stream.

**Implementation:** `vlp_session.create()` allocates a `Session` struct in host memory array. No device memory allocation per session — the device memory is pre-allocated in the layout and sessions index into it. The "stream" concept becomes the bridge's command buffer, which is shared and serialized via fence.

**Difference:** Original assumed per-session GPU streams for parallelism. Implementation uses a single command buffer with fencing. Multi-session parallelism would require multiple command buffers and a queue per session. This is a future optimization, not a SPIR-V constraint.

**Original:** `vlp_session_snapshot` pauses runners, fences GPU operations, copies device state.

**Implementation:** `vlp_snapshot.captureFromDevice()` calls `bridge.downloadFromBuffer()` for each region. The bridge internally submits a fence and waits, ensuring all prior GPU work completes before the download. Runner pausing is handled by `vlp_runner.stop()` which joins the runner thread.

**Difference:** The original assumed atomic capture via GPU-side fencing. The implementation does the same thing but through the bridge's fence mechanism rather than a custom TensorProlog fence. The result is identical: all GPU work completes before state is read.

**Original:** `vlp_session_clone` sets up COW with device-side page fault handling.

**Implementation:** `vlp_kb_store.CowPageTable` tracks dirty bits in host memory. COW faults are detected at write time in `kb_store.factWrite()` — when writing to a page that isn't dirty, the host copies the page via `bridge.copyBufferToBuffer()` before writing. This is software COW, not hardware page faults.

**Difference:** Original assumed CUDA unified memory page faults or similar hardware mechanism. SPIR-V/Vulkan has no equivalent. Software COW is explicit but functionally identical. The COW granularity (4 KB pages, ~100 facts per page) is the same.

### 5.2 Runner Scheduler (Original Section 5.2)

**Original:** Four runner types with thread pool, recycling, backoff.

**Implementation:** `vlp_runner.RunnerScheduler` implements all four types. Each runner gets its own `std.Thread`. The thread runs a type-specific loop:

- **Poller:** `inference.executeL3()` at `interval_ms` intervals. Zero LLM tokens.
- **Processor:** `inference.cycle()` continuously. Recycles at `max_turns_before_recycle`.
- **Internal:** Same as poller (fires rules on a specific KB).
- **Batch:** Pops task from queue KB, clones session, processes in clone, merges back, kills clone.

**Difference:** Original specified a thread pool. Implementation uses one thread per runner for simplicity. A pool can be added later without changing the module interface. The runner-to-session binding, recycle lifecycle, and backoff logic match the original spec exactly.

### 5.3 Grant Enforcer (Original Section 5.3)

**Original:** All integer checks, grant store on device, index on (user_id, grant_class).

**Implementation:** `vlp_grant.GrantEnforcer` keeps grants in host memory. The index is a flat array of `(user_id, grant_class, grant_store_index)` tuples scanned linearly.

**Difference:** Original placed the grant store in device memory. Implementation keeps it host-side because grant checks are sequential (scan matching grants, check state, check expiry, check pattern, decrement uses). This is 5-20 integer comparisons per check — nanoseconds on CPU, not worth a GPU dispatch. The grant store is included in snapshots by the snapshot manager reading it from host memory rather than from a device buffer.

### 5.4 Command Processor (Original Section 5.4)

**Original:** `vlp_command_parse` and `vlp_command_execute` with detailed dispatch per command type.

**Implementation:** `vlp_command.CommandProcessor` implements `parse()` and `execute()` with the same flow:
1. Access check via `vlp_access.check()`
2. Grant check via `vlp_grant.check()` (if operational)
3. Dispatch by command type

Each command type calls the appropriate module:
- `KB_ASSERT` → `kb_store.factWrite()`
- `KB_QUERY` → `kb_store.factRead()` or `kb_store.factScanByTag()`
- `PROLOG_QUERY` → `prolog.query()` — which may dispatch to GPU internally
- `BUILTIN_CALL` → `builtins.dispatch()` — which may dispatch to GPU internally
- `GRAMMAR_RENDER` → `grammar.render()` — always host
- `DIRECT_OUTPUT` → `kb_store.factRead()` + `grammar.render()`

**Difference:** None. The command processor was always host-side in the original. The only change is that `GRAMMAR_RENDER` and `DIRECT_OUTPUT` cannot delegate to GPU, which the original didn't explicitly require.

### 5.5 Access Control (Original Section 5.5)

**Original:** `vlp_access_check` walks KB parent chain checking visibility.

**Implementation:** `vlp_access.check()` does exactly this. Loads KB via `kb_store.getKb()` (which may hit host cache or read from device), checks `visibility` field, walks `parent_id` chain.

**Difference:** None. This was always a host operation. The implementation adds `resolveVisible()` which enumerates all accessible KBs from a scope by walking the tree and pruning at invisible ancestors.

---

## 6. How Each Device Engine Maps

### 6.1 LLM Engine (Original Section 6.1)

**Original:** Single `vlp_llm_forward` function dispatches the entire forward pass.

**Implementation:** `vlp_llm.LlmEngine.forward()` orchestrates 12 GPU kernel dispatches per layer:

```
embedding_lookup
For each layer:
    layer_norm (pre-attention)
    qkv_project
    kv_cache_append
    attention_scores
    softmax_exact
    attention_weighted_sum
    output_project
    residual_add
    layer_norm (pre-MLP)
    mlp
    residual_add
layer_norm (final)
lm_head
```

Each dispatch is a call to `bridge.dispatch()` with a specific `PipelineId` and typed params struct from `vlp_gpu_params.zig`.

**Difference from original:** The original assumed a single monolithic dispatch. The implementation decomposes into individual kernels with pipeline barriers between them. This is necessary because SPIR-V compute shaders are single entry-point programs — you cannot have one shader that calls another. Each kernel is a separate `.spv` module.

The host records all dispatches into a single command buffer using `bridge.dispatchSequence()` for the full forward pass, so there is only one submit and one fence wait per forward pass, not one per kernel.

**KV Cache:** Original stored KV cache as KB facts (8M fact slots). Implementation stores KV cache in a dedicated `kv_cache` buffer (Set 2, binding 2) as a flat Q16 array indexed by `(layer, position, head, kv_select, dim)`. The offset computation in `KvCacheConfig.offsetFor()` replaces the original's fact-based addressing. This is more efficient — no fact tag overhead, pure contiguous Q16 data.

**Sampling:** Original had `vlp_llm_generate_token` on device. Implementation downloads logits to host and samples there. `sampleGreedy()`, `sampleTopK()`, `sampleTopP()` are host-side scans over the vocabulary. For vocab_size=32K this is ~128 KB of data — microseconds on CPU.

**Constrained generation:** `generateCommandTokens()` applies a vocabulary mask before sampling. The mask zeroes out all non-command-token logits, leaving ~300 candidates. This is a host-side operation on the downloaded logit array.

### 6.2 KB Store Engine (Original Section 6.2)

**Original:** Five functions: init, create_kb, fact_write, fact_read, scoped_search, plus COW.

**Implementation split:**

| Function | Runs On | Module |
|---|---|---|
| `createKb` | Host | `vlp_kb_store.zig` |
| `getKb` | Host (cached) or device read | `vlp_kb_store.zig` |
| `freezeKb`, `setVisibility` | Host + device write | `vlp_kb_store.zig` |
| `pathResolve`, `pathRegister` | Host only | `vlp_kb_store.PathIndex` |
| `factWrite` (single) | Host → device upload | `vlp_kb_store.zig` |
| `factRead` (single) | Device download → host | `vlp_kb_store.zig` |
| `factWriteBatch` | GPU kernel if > 256 | `vlp_kb_store.zig` → `bridge.dispatch(.fact_write_batch)` |
| `factReadBatch` | GPU kernel if > 256 | `vlp_kb_store.zig` → `bridge.dispatch(.fact_read_batch)` |
| `factScanByTag` | GPU kernel if > 256 | `vlp_kb_store.zig` → `bridge.dispatch(.fact_scan_by_tag)` |
| `scopedSearch` | Host builds chain, delegates to `factScanByTag` per KB | `vlp_kb_store.zig` |
| `cowInit`, `cowResolve`, `cowDestroy` | Host | `vlp_kb_store.CowPageTable` |
| `textAppend`, `textRead` | Host → device buffer | `vlp_kb_store.zig` |
| `addChild`, `removeChild`, `addMount` | Host + device write | `vlp_kb_store.zig` |

**Key difference:** The original `vlp_kb_store_scoped_search` walked the parent chain on device. The implementation walks the chain on host (integer `parent_id` reads from cached KB structs), builds a `ChainEntry` array, then dispatches `factScanByTag` per KB in the chain. This avoids pointer chasing on GPU — the GPU only sees flat arrays with pre-computed offsets.

**Host KB cache:** The implementation adds a host-side cache of KB structs (`kb_cache` array, up to 1024 entries). Structural operations (checking `parent_id`, `visibility`, `frozen`, `facts_offset`) hit the cache instead of reading from device. Cache invalidation occurs on `writeKbToDevice()`. This optimization is invisible to callers — `getKb()` returns the same data regardless of cache hit/miss.

### 6.3 Prolog Engine (Original Section 6.3)

**Original:** Recursive unification, depth-first search with GPU parallelism across candidates.

**Implementation:** All recursion eliminated. Search is iterative on host. GPU parallelism is per-candidate-set:

```
Host: build chain → collect candidate offsets
Host: if candidates > 32 → dispatch vlp_kernel_unify_candidates to GPU
      else → run unifySingle() on host for each candidate
Host: collect results
Host: if matched rules have body conditions → dispatch vlp_kernel_rule_body_eval
Host: if sub-goals remain → build new candidate set, repeat (iterative backtracking)
Host: enforce depth limit (100 iterations max)
```

**Difference:** The original described recursive `vlp_prolog_unify` with depth bound. The implementation provides `unifySingle()` which handles flat (non-nested) unification as a host function with no recursion. For COMPOUND terms with args, comparison is iterative over the args array with a fixed bound from `args_count`.

Nested compounds (compound inside compound) are handled by the host driving multiple rounds of dispatch. Each round handles one nesting level. The host flattens the query tree into a sequence of flat unification rounds. This matches the SPIR-V constraint (no recursion) while preserving the original's semantic behavior.

**Rule firing:** The original `vlp_prolog_fire_rules` evaluated all rules and returned proposed actions. The implementation's `fireRules()` follows the same logic:
1. `ruleMatchScan()` — parallel head matching (GPU if > 64 rules)
2. `ruleBodyEval()` — parallel body condition checking
3. Collect fully satisfied rules
4. Return firing rule IDs (actions not applied yet)

`fireAndCommit()` applies actions immediately — same as original's convenience function.

### 6.4 Grammar Engine (Original Section 6.4)

**Original:** Device-side grammar rendering with template walking and slot filling.

**Implementation:** Entirely host-side in `vlp_grammar.zig`. The engine:
1. `compile()` — parses template, extracts `{slot_name}` markers, builds literal range + slot position tables
2. `render()` — walks template copying literal ranges and rendering fills
3. `renderFromKb()` — fills come from KB facts (read via `kb_store.factRead()`)
4. `inherit()` — walks KB parent chain looking for grammar at given slot

**Difference:** This is the largest deviation from the original spec. The original assumed grammar rendering on GPU. The implementation moves it entirely to host because:
- Template walking is serial and branchy (copy N literal bytes, render fill, copy M literal bytes, ...)
- Integer-to-text conversion (`q16ToString`, `i32ToString`) is sequential digit extraction
- Text concatenation has variable-length outputs
- Nested grammar rendering would require recursion

None of these map to GPU parallelism. A grammar render takes microseconds on CPU. The LLM forward pass takes milliseconds. Grammar rendering is never on the critical path.

The grammar types (`Grammar`, `GrammarSlot`, `GrammarFill`, `GrammarKbMapping`) remain `extern struct` in `vlp_types.zig` because they are stored in the device grammar store buffer for snapshot portability. But the rendering code never runs on GPU.

### 6.5 Builtin Executor (Original Section 6.5)

**Original:** 448 builtins dispatched via function pointer table on device.

**Implementation:** `vlp_builtin.BuiltinExecutor` dispatches via host-side `switch`:

```
Host: validateIoSe() — check arg types/counts (host, integer checks)
Host: if operational (id >= 404) → dispatchOperational() (host only, grant-gated)
Host: if pure and builtinToPipeline() finds a mapping:
        if shouldUseGpu(data_size) → dispatchGpu() with appropriate pipeline
        else → dispatchHost() with same arithmetic
Host: if no pipeline mapping → dispatchHost()
```

The GPU pipelines group builtins by operation type:

| Pipeline | Builtin IDs | Op code selects |
|---|---|---|
| `builtin_unary` | 0-34 | abs, negate, sign, square, double, halve, ... |
| `builtin_binary` | 35-74 | add, sub, mul, div, mod, min, max, gcd, ... |
| `builtin_reduction` | 75-99 | sum, product, min, max, mean, variance, ... |
| `builtin_sort` | 100-114 | sort, reverse, unique, ... |
| `builtin_matmul` | 115-144 | matmul, dot, transpose, ... |
| `builtin_confidence_combine` | 195-204 | agreeing, conflicting combination |
| `builtin_confidence_chain` | 205-209 | repeated multiplication |

**Difference:** The original had a single dispatch table with function pointers. SPIR-V does not support function pointers. The implementation uses 7 GPU pipelines with `op_code` params inside the kernel. Each kernel has a `switch` on `op_code` — this is static branching resolved at compile time per invocation, not dynamic dispatch.

The 44 operational builtins (filesystem, compile, execute, lint, network, process) never touch GPU. They execute on host via OS APIs. The original spec already implied this (they interact with Docker, HTTP, filesystem) but listed them as device-side. The implementation makes the boundary explicit.

---

## 7. The Inference Loop — Spec Section 7 Revisited

The original inference loop is preserved almost verbatim in `vlp_inference.InferenceEngine.cycle()`. The mapping:

```
Original Phase 1 (Input Processing):
  → tokenize() — host-side, compiled tokenizer

Original Phase 2 (Context Assembly):
  → buildContext() — host-side, reads from KB via kb_store
  Context includes: system prompt, input tokens, scratchpad, scope ref
  Context does NOT include: previous turns, raw data, prior reasoning

Original Phase 3 (LLM Generation):
  → llm.forward() dispatches 12×n_layers GPU kernels
  → llm.generateToken() downloads logits, samples on host
  → classifyToken() determines prose/command/direct_output/end_of_turn
  → For commands: llm.generateCommandTokens() with constrained vocab
    → commands.parse() + commands.execute()
    → Result to scratchpad
  → For direct output: commands.parseKbUrl() + commands.execute(.direct_output)
    → Grammar rendering on host, output to buffer
  → For prose: tokenToText(), append to output

Original Phase 4 (Command Execution):
  → commands.execute() — access check, grant check, dispatch
  → This may trigger GPU work (fact scan, unification, builtin)
  → All GPU work is synchronous — execute() blocks until GPU fence signals

Original Phase 5 (Post-Processing):
  → session_mgr.incrementTurn()
  → session_mgr.updateLevelStats()

Original Phase 6 (Auto-Persist):
  → session_mgr.shouldAutoSnapshot() checks turn counter mod interval
  → If true: snapshot_mgr.captureFromDevice()
```

**Differences from original:**

1. Original had `vlp_llm_generate_command` as a device function. Implementation generates command tokens on host using constrained sampling over downloaded logits.

2. Original had `vlp_scratchpad_write` as implicit. Implementation has `Scratchpad` struct with explicit `write()` and `clear()`.

3. Original described interleaved command/prose generation in a single loop. Implementation uses `classifyToken()` to branch on token type, then switches to the appropriate generation mode (constrained for commands, unconstrained for prose).

4. Original's `DIRECT_OUTPUT` resolved `kb://` URLs on device. Implementation resolves them on host via `commands.parseKbUrl()` which calls `kb_store.pathResolve()`.

---

## 8. Execution Levels — Spec Section 8 Revisited

The three execution levels map directly:

**L1 — Full LLM Judgment:**
`inference.executeL1()` → `inference.cycle()` → full generation loop.
50-500 tokens. LLM reads, reasons, generates commands + prose.
At the end, LLM may emit `CMD_PROLOG_ASSERT_RULE` to formalize judgment.
Cost: ~25-40 tokens for the rule assertion command.
Value: transitions pattern from L1 to L2/L3.

**L2 — LLM Invokes Stored Rule:**
`inference.executeL2()` → `prolog.query()` against stored rules.
LLM generates `CMD_PROLOG_QUERY` (~8 tokens) + wraps result in prose (~10 tokens).
The Prolog query may dispatch to GPU (parallel unification) or run on host.

**L3 — Automatic Rule Firing:**
`inference.executeL3()` → `prolog.fireAndCommit()`.
Zero LLM tokens. Called by poller/internal runners.
Rule matching dispatches to GPU if rule count > 64.
Body evaluation dispatches to GPU if candidates > 32.
Otherwise runs on host. Same results either way.

**Level tracking:**
`session_mgr.updateLevelStats()` increments the appropriate counter.
`session_mgr.getLevelStats()` returns `LevelStats` with exact integer counts.
`autoTriageNum() / autoTriageDen()` gives the exact L3 fraction.
At investigation 100: expected ~93% L3 (from paper Appendix D).

---

## 9. Confidence Propagation — Spec Section 9 Revisited

**Original:** Three operations — assign, combine, chain.

**Implementation:** `vlp_confidence.zig` provides all three with identical semantics:

- `assign()` — sets `fact.provenance.confidence` from `confidence_table[source_type]`. Pure host, one array index.

- `combineAgreeing()` — formula: `1 - ∏(1-C_i)`. Host for N ≤ 64 (i64 accumulator, iterative multiply). GPU for N > 64 via `builtin_confidence_combine` pipeline.

- `combineConflicting()` — same as agreeing + penalty per conflict pair.

- `chain()` — `C^N` via repeated `Q16.mul()`. Always host — the iteration count is N (link count), typically < 10.

- `propagate()` — walks provenance chain via `fact.provenance.derivation_rule_id`. Always host — it's a linked-list traversal through KB facts.

**Difference:** Original had confidence propagation potentially on device. Implementation keeps it host-side because the provenance chain walk is sequential and the combine operation is only worth GPU dispatch for very large source counts (> 64 agreeing sources is rare in practice).

The arithmetic is identical. `Q16.mul()` in `vlp_types.zig` is the same integer multiply whether it runs on host or in a GPU kernel. The results are bit-identical.

---

## 10. Snapshot Format — Spec Section 10 Revisited

**Original:** Binary format with header + contiguous regions.

**Implementation:** `vlp_snapshot.SnapshotHeader` is an `extern struct` matching the original layout exactly. The regions are downloaded from device in order:

```
[header]
[kb_store region]
[fact_store region]
[rule_store region]
[term_store region]
[text_store region]
[grammar_store region]
[live_state region]
[grant_store region]
```

**Difference:** The original included `path_index_region`. The implementation's path index is host-side (a hash map in `KbStore.PathIndex`). For snapshot, the path index is reconstructed on restore by scanning the KB store and re-registering each KB's path. The header includes `path_index_region_size = 0` for compatibility.

**Checksum:** `computeChecksum()` implements CRC32 with the standard polynomial (0xEDB88320). This is integer-only, deterministic, and computed on the raw snapshot bytes. If the checksum doesn't match on restore, `restoreToDevice()` returns `ERR_SNAPSHOT_CORRUPT` — hard fail, never silent.

**Diff:** `diff()` compares two snapshots region by region. If regions differ in size or content, a `DiffEntry` is recorded. Because all data is integers, every difference is a real change. No float noise. If the diff is non-empty and you expected identical snapshots, something changed. Find it.

---

## 11. Seed Layer — Spec Section 11 Revisited

**Original:** ~23,400 entries across ~1.5 MB, frozen after init.

**Implementation:** `vlp_seed.zig` creates 10 well-known KBs with fixed IDs:

```
0: root
1: root.system
2: root.system.oso              (15 engineering principles)
3: root.system.confidence       (11-entry confidence table)
4: root.system.builtins         (IOSE declarations)
5: root.system.command_vocab    (15 command type names)
6: root.system.hygiene          (3 hygiene rule descriptions)
7: root.templates
8: root.templates.sentences
9: root.templates.formats
```

All are frozen after population via `kb_store.freezeKb()`.

**Difference:** The original described ~23,400 entries. The implementation's `createFresh()` populates the structural content (principles, confidence table, command vocab, hygiene rules) but leaves sentence templates and format grammars as empty KBs to be populated from a snapshot. The `init()` function tries to load from a snapshot file first, falling back to `createFresh()`.

The hygiene rules are stored as fact descriptions (text references) rather than as compiled Prolog rules. This breaks a circular dependency: Prolog rule assertion requires the KB store, which requires seed initialization. The Prolog engine can load these descriptions and compile them into rules on first use.

---

## 12. Multi-Device — Spec Section 12 Revisited

**Original:** Pipeline parallelism with NVLink transfers.

**Implementation:** `vlp_multi_device.MultiDeviceManager` creates one bridge per device, assigns a contiguous layer range to each:

```
Device 0: layers [0, k)
Device 1: layers [k, 2k)
...
Device N-1: layers [(N-1)k, n_layers)
```

Forward pass: device 0 runs embedding + its layers, transfers hidden state to device 1 (host-staged copy for now, NVLink peer copy when available), device 1 runs its layers, and so on. Last device runs final norm + lm_head.

**Difference:** The original assumed NVLink peer-to-peer transfer via `transferHiddenState()`. The implementation provides a host-staged fallback (download from source device, upload to target device). This adds latency but works on any Vulkan-capable hardware. Peer transfer via `VK_KHR_external_memory` is a future optimization.

**KB replication:** `replicateKb()` downloads KB struct + facts from source device, uploads to target device. `syncKb()` broadcasts from device 0. All data is integer bytes — bit-identical after transfer.

---

## 13. Error Handling — Spec Section 15 Revisited

**Original:** Error categories with deterministic recovery.

**Implementation:** `vlp_types.Status` has `ErrorCategory` + `ErrorCode` + detail i32. `recoverFromError()` maps codes to `RecoveryAction` via a deterministic `switch` — same error in same state always produces same recovery.

**GPU errors:** Kernels report errors via `status_buffer[global_invocation_id]`. After dispatch, the host reads the status buffer. Any non-zero entry is an error. The bridge's `dispatch()` function checks this automatically and returns the first error found.

**Difference:** The original had errors as return values from device functions. SPIR-V kernels cannot return errors — they write to a buffer and return void. The status buffer pattern is the SPIR-V-compatible equivalent.

---

## 14. Invariants — What Changed, What Didn't

All 10 original invariants hold. The 5 new invariants (11-15) were added for SPIR-V conformance:

| Invariant | Original | Implementation | Status |
|---|---|---|---|
| 1: Softmax sums to D | Device-side check | GPU kernel guarantees by construction (FRU remainder distribution) | **Unchanged** |
| 2: Facts at integer addresses exact | Device memory | Same device memory, accessed through bridge | **Unchanged** |
| 3: Bounded primitives can't exceed bounds | Device-side enforcement | Host-side enforcement in `kb_store.factWrite()` bounds checks | **Moved to host, same guarantee** |
| 4: Snapshot restore is bit-identical | Device-to-device copy | Device→host→device copy via bridge, integers transfer exactly | **Unchanged** |
| 5: Clone COW invisible to parent | Device-side page faults | Host-side software COW in `CowPageTable` | **Same guarantee, different mechanism** |
| 6: Access-denied data is absent | Device-side path skip | Host-side `access.check()` before any data read | **Moved to host, same guarantee — data never touched** |
| 7: Grant denial before execution | Device-side gate | Host-side `grants.check()` before dispatch | **Moved to host, same guarantee — no side effects on denial** |
| 8: Integer arithmetic deterministic across devices | Device guarantee | Same integer types, same operations, host or device | **Unchanged** |
| 9: Prolog uses exact comparison | Cross-multiply on device | `Q16.eql()` or `Q16.crossMultiplyCompare()` on host or device | **Unchanged** |
| 10: Audit log append-only and complete | Device-side ring buffer | Host-side `AuditLog` ring buffer | **Moved to host, same guarantee** |
| 11: GPU kernels stateless | NEW | All state in storage buffers, kernels are pure functions | **SPIR-V conformance** |
| 12: No recursion on GPU | NEW | All algorithms iterative, max iteration from dispatch_params | **SPIR-V conformance** |
| 13: No dynamic allocation on GPU | NEW | All buffers pre-allocated by host | **SPIR-V conformance** |
| 14: Host-device data integer-identical | NEW | No format conversion, integers transfer exactly | **SPIR-V conformance** |
| 15: Dispatch overhead bounded | NEW | Every kernel has bounded iteration, errors reported via status buffer | **SPIR-V conformance** |

---

## 15. Performance Expectations

The original spec's benchmarks (Section 14.2) map to the implementation:

**Forward pass:** Same GPU kernels, same arithmetic. The decomposition into 12 kernels per layer adds pipeline barrier overhead but all dispatches are in one command buffer, so submit overhead is amortized. Expected throughput: within 5-10% of a monolithic kernel.

**Prolog query:** For > 32 candidates, GPU dispatch. For ≤ 32 candidates, host executes same algorithm. Host path may be faster for small queries due to no dispatch overhead. The paper's FPGA reference (~1.1μs for 200 facts) is a lower bound; GPU dispatch overhead adds ~10μs, so GPU path wins at ~300+ candidates.

**Grammar render:** Host-only. Microseconds. Never on critical path.

**Builtin execution:** Pure builtins on GPU for arrays > 512. The paper's "50,000× faster than LLM token" claim holds — a reduction over 1000 elements is ~1μs on GPU vs ~50ms for a single LLM forward pass.

**Snapshot:** Host memory copy. Size is typically 10-500 KB (non-model state). Microseconds for small sessions, milliseconds for large ones.

---

## 16. Implementation Stages — Spec Section 17 Revisited

The 5-stage build plan maps to the 22 modules:

**Stage 1 (Foundation):**
- `vlp_types.zig` — all type declarations
- `vlp_device_memory.zig` — layout + sizing
- `vlp_gpu_params.zig` — dispatch param structs
- `vlp_bridge.zig` — Vulkan init + dispatch
- `vlp_kb_store.zig` — KB CRUD, fact ops, path index
- `vlp_access.zig` — visibility checks
- `vlp_audit.zig` — ring buffer

**Stage 2 (Intelligence):**
- `vlp_prolog.zig` — unification, query, rule fire
- `vlp_grammar.zig` — compile, render, inherit
- `vlp_confidence.zig` — assign, combine, chain, propagate
- `vlp_session.zig` — create, snapshot, clone, merge, kill
- `vlp_snapshot.zig` — save, load, diff, merge
- `vlp_grant.zig` — check, create, revoke

**Stage 3 (Precision):**
- Q32 and Q335 operations added to `vlp_types.zig`
- Additional GPU kernels for wide-integer builtins
- FRU (Fixed Remainder Unit) kernel for exact softmax

**Stage 4 (Operations):**
- `vlp_builtin.zig` — operational builtins (filesystem, compile, execute, network)
- `vlp_command.zig` — command parser + executor
- `vlp_inference.zig` — full inference loop
- `vlp_runner.zig` — poller, processor, internal, batch

**Stage 5 (Scale):**
- `vlp_llm.zig` — full forward pass kernel orchestration
- `vlp_multi_device.zig` — pipeline parallelism
- `vlp_seed.zig` — seed layer
- `vlp_system.zig` — top-level wiring
- `vlp_test.zig` — test infrastructure
- GLSL compute shader implementations for all 28 kernels
- Production deployment tooling

---

## 17. File Manifest

22 files. No additional files. All helpers are internal to these modules.

```
vlp_types.zig           — shared types, Q16 arithmetic, confidence table, error codes
vlp_device_memory.zig   — memory layout, capacity planning, default configs
vlp_gpu_params.zig      — 28 dispatch param structs, pipeline IDs, descriptor bindings
vlp_bridge.zig          — Vulkan lifecycle, dispatch, buffer transfer, GPU/host decision
vlp_llm.zig             — forward pass orchestration, KV cache, sampling
vlp_kb_store.zig        — KB CRUD, fact ops, path index, COW, text store, scoped search
vlp_prolog.zig          — iterative unification, query, rule match/fire, term store
vlp_grammar.zig         — template compile, render, inherit, VDR formatting
vlp_builtin.zig         — 448 builtin dispatch, IOSE validation, host/GPU routing
vlp_confidence.zig      — assign, combine agreeing/conflicting, chain, propagate
vlp_session.zig         — session lifecycle, clone, merge, level stats
vlp_snapshot.zig        — capture, restore, save, load, diff, three-way merge, CRC32
vlp_runner.zig          — poller, processor, internal, batch, threading, recycle
vlp_grant.zig           — check, create, revoke, cleanup, pattern matching
vlp_access.zig          — visibility walk, visible KB enumeration
vlp_audit.zig           — ring buffer, filtered query, convenience writers
vlp_command.zig         — parse, execute, batch, all 15 command types
vlp_inference.zig       — full cycle, L1/L2/L3, context build, scratchpad
vlp_seed.zig            — seed KB tree, OSO principles, confidence table, hygiene rules
vlp_system.zig          — top-level init/deinit, status, recovery
vlp_multi_device.zig    — pipeline parallelism, KB replication
vlp_test.zig            — determinism, roundtrip, isolation, confidence, softmax tests
```

---

## 18. Summary of Deviations from Original Spec

| Original Assumption | Implementation Reality | Impact |
|---|---|---|
| Grammar rendering on GPU | Host-only | None — microseconds, never on critical path |
| Path index hash map on GPU | Host-only | None — single lookup per command |
| COW via hardware page faults | Software COW with dirty bits | Same semantics, explicit page copy on first write |
| Grant store on device | Host memory | None — 5-20 integer checks per grant, nanoseconds |
| Recursive Prolog unification | Iterative with explicit stack on host | Same semantics, host drives backtracking, GPU does parallel candidate evaluation |
| Function pointer dispatch for builtins | Host-side switch + 7 GPU pipelines with op_code | Same results, no dynamic dispatch on GPU |
| Per-session GPU streams | Shared command buffer with fencing | Serialized — future optimization for multi-session parallelism |
| KV cache as KB facts | Dedicated flat Q16 buffer | More efficient — no fact tag overhead |
| Monolithic forward pass | 12 kernels per layer | Pipeline barriers add ~5% overhead, offset by flexibility |
| NVLink peer transfer | Host-staged copy (NVLink future) | Adds latency for multi-device, functional on any hardware |
| Device-side audit writes | Host-side ring buffer | Same guarantee — append-only, complete |
| ~23,400 seed entries from code | Structural seed + snapshot load | Fresh seed has ~50 entries, full seed loaded from snapshot |

Every deviation preserves the original spec's semantics. The integers are the same. The arithmetic is the same. The access control is the same. The safety model is the same. The determinism guarantee is the same. The execution just happens in a different place — wherever SPIR-V allows it.
