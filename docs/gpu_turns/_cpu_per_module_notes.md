```
================================================================================
IMPLEMENTATION NOTES — CPU modules against GPU contract
Only non-obvious constraints. If it's straightforward, it's not here.
================================================================================

vlp_types.zig
─────────────
- Q16.mul accumulator MUST be i64. i32*i32 overflows at v=256 (256*256=65536 which
  is fine, but 65535*65535=4294836225 which exceeds i32 max). Every multiply widens.
- Q16.div: caller MUST check b.v != 0 before calling. No runtime panic on GPU.
  Host mirrors this — no division without guard.
- extern struct padding: Kb._reserved must stay [58]u8 to hit 256 bytes exactly.
  If you add a field to Kb, shrink _reserved by the same byte count. GPU buffer
  indexing is `kb_id * 256` — if struct size drifts, every KB read is wrong.
- Fact is 40 bytes. This is NOT a power of two. GPU fact_store indexing is
  `(facts_offset + slot_id) * 40`. The bridge does byte-offset arithmetic, not
  element indexing. If you change Fact size, update FACT_SIZE in device_memory
  AND every GPU kernel that indexes the fact buffer.
- Term union packing: all union fields are always present (not a tagged union at
  the Zig level). GPU reads whichever field matches term.type. Unused fields
  contain garbage. Never read a field that doesn't match the type tag.
- ErrorCode values must stay stable. GPU status_buffer writes raw i32 error codes.
  If you renumber ErrorCode, old GPU shader binaries report wrong errors until
  recompiled. Pin the enum values.

vlp_device_memory.zig
─────────────────────
- computeLayout assigns offsets SEQUENTIALLY. No gaps. If you insert a new region
  between existing ones, every subsequent offset shifts and all descriptor set
  bindings break. Add new regions at the END only.
- KV cache sizing: `2 * n_layers * max_seq * n_heads * d_head * 8` bytes.
  For 32 layers, 4096 seq, 32 heads, 128 d_head = 2.1 GB. This is the largest
  non-model buffer. If it doesn't fit, reduce max_seq_len first.
- status_buffer is sized for max_dispatch_invocations. If you dispatch more
  invocations than this, status writes go out of bounds on GPU. The kernel has
  no way to check — the host must guarantee dispatch count <= buffer size.

vlp_gpu_params.zig
──────────────────
- Every params struct MUST be extern struct. Zig regular structs have undefined
  layout — the GPU will read garbage from the uniform buffer.
- Params structs should be padded to 16-byte alignment. Vulkan spec requires
  uniform buffer offsets to be multiples of minUniformBufferOffsetAlignment
  (typically 16 or 256 bytes). The current structs are 16-32 bytes. If you add
  fields that push past 256 bytes, you need to check the device limit.
- op_code fields in builtin params must match the GPU kernel's switch. If you
  add a new unary op on host, you must add the matching case in the GLSL kernel.
  There is no runtime check — wrong op_code means the kernel does nothing or
  does the wrong thing silently.

vlp_bridge.zig
──────────────
- dispatch() is SYNCHRONOUS. It submits, waits on fence, returns. Do not call
  dispatch() from two threads simultaneously — the command buffer and fence are
  shared. If you need concurrent dispatches, you need multiple command buffers
  and fences (not implemented yet).
- dispatchSequence() records multiple dispatches into ONE command buffer with
  pipeline barriers. Use this for the LLM forward pass. Do NOT use N separate
  dispatch() calls per layer — the per-dispatch fence overhead kills performance.
- uploadToBuffer with mapped memory does a memcpy to the mapped pointer.
  This is a COHERENT write — no flush needed IF the memory is HOST_COHERENT.
  If the device doesn't support host-coherent memory, you need vkFlushMappedMemoryRanges.
  The bridge should check this at init and set a flag.
- downloadFromBuffer with non-mapped memory allocates a staging copy command.
  This means it submits a command buffer. If you're in the middle of recording
  a dispatch sequence, you CANNOT call downloadFromBuffer until that sequence
  is submitted. The bridge has one command buffer — interleaving record and
  submit corrupts it.
- resetStatusBuffer and resetResultCounts use fillBuffer which submits a command.
  Call these BEFORE starting a dispatch sequence, not during.
- shouldUseGpu returns false if bridge.initialized is false. This means during
  startup (before bridge.init completes), all modules fall back to host path.
  This is intentional — it lets you test KB/Prolog logic without a GPU.

vlp_llm.zig
───────────
- forwardLayer dispatches 12 kernels. The scratch_a and scratch_b buffers are
  used ALTERNATELY — embedding output goes to scratch_b, layer_norm reads from
  scratch_b and writes to scratch_a, qkv_project reads scratch_a writes scratch_b,
  etc. If you reorder kernels, you must verify the ping-pong is correct or you
  read stale data.
- The pipeline barrier between kernels is compute-write → compute-read. If you
  skip a barrier (e.g., fusing two kernels), you get a race condition. Vulkan
  validation layers will catch this in debug mode.
- attention_scores dispatch uses (n_heads, n_tokens, 1) workgroups. Each workgroup
  computes one row of scores. If n_heads * n_tokens > maxComputeWorkGroupCount[0],
  you need to split into multiple dispatches. H100 limit is 2^31-1, so this
  only matters for extremely long sequences on weak hardware.
- softmax_exact MUST redistribute remainder to make sum == D exactly. The kernel
  does: normalize, compute sum, find element with largest remainder, add/subtract
  1. If the kernel doesn't do this, INVARIANT 1 is violated.
- KV cache offset computation in KvCacheConfig.offsetFor is 6 multiplies.
  These must match the GPU kernel's offset computation EXACTLY. If they disagree,
  the GPU reads wrong K/V values and attention is garbage. Test with known
  values at specific positions.
- sampleGreedy downloads the ENTIRE logit array (vocab_size * 4 bytes).
  For vocab=32K this is 128 KB. For vocab=256K this is 1 MB. Consider
  downloading only top-K from GPU if vocab is very large (requires a GPU
  top-K kernel, not implemented).
- generateCommandTokens masks logits AFTER download. The mask loop is O(vocab_size)
  per token. For 300 command tokens out of 32K vocab, this scans 32K entries.
  Fine for single tokens. If batch generating, consider precomputing the mask
  as a bitmap.

vlp_kb_store.zig
────────────────
- factWrite checks COW BEFORE uploading. If a COW page copy is needed, it does
  a bridge.copyBufferToBuffer (which submits a command). Then it does the
  uploadToBuffer for the actual write. These are two separate submits. If the
  COW copy fails, the write must not proceed — check the status.
- getKb reads from host cache first. The cache is invalidated by writeKbToDevice.
  If anything modifies the KB struct on device WITHOUT going through writeKbToDevice
  (e.g., a GPU kernel writes to kb_store buffer directly), the cache is stale.
  Currently no GPU kernel modifies KB structs — all KB struct writes are host-side.
  Keep it that way.
- factScanByTag GPU path: the kernel writes matching SLOT INDICES (not facts)
  to scratch_a via atomic counter. The host then reads these indices and fetches
  the actual facts one by one. This is two round-trips: dispatch + readback indices,
  then N individual fact reads. For better performance, a second kernel could
  gather the matching facts into scratch_b — not implemented yet.
- scopedSearch calls factScanByTag per KB in the chain. Each call may dispatch
  to GPU independently. For a chain of 5 KBs each with 1000 facts, that's 5
  GPU dispatches. Consider building a single flat offset array across the chain
  and dispatching scoped_search kernel once.
- textAppend is append-only. text_used only grows. There is no compaction.
  Over a long session, the text store fills up. Snapshot + restore resets it
  (snapshot captures only used bytes). Without periodic snapshots, text_store
  will eventually exhaust its allocated region.
- PathIndex uses FNV-1a hash. Collisions are resolved by linear probe.
  If two different paths produce the same hash, the second path gets the next
  slot. If you delete a path and don't rehash, subsequent lookups for paths
  that probed past the deleted slot will miss. rehashFrom handles this but
  it's O(cluster_length) per delete.
- COW page size is 4096 bytes (COW_PAGE_SIZE constant). At 40 bytes per fact,
  that's 102 facts per page. A single factWrite to slot 0 of a cloned KB
  copies 102 facts. This is the granularity tradeoff — smaller pages mean
  less copy but more dirty-bit overhead.

vlp_prolog.zig
──────────────
- unifySingle is NOT recursive. It handles flat terms only. COMPOUND terms
  match on functor_id + args_count but do NOT recursively compare args.
  For nested compound unification, the host must drive multiple rounds:
  round 1 matches outer functor, round 2 matches args. This is the SPIR-V
  recursion workaround. If your query has nesting depth N, you need N rounds.
- unifyCandidatesGpu uploads candidate offsets as ABSOLUTE fact_store indices,
  not slot-relative. The GPU kernel reads fact_store[candidate_offsets[gid]].
  If you pass slot-relative indices, every read is wrong.
- query collects ALL candidate fact offsets across the chain into candidate_buf
  (capacity 4096). If the chain has more than 4096 total facts, candidates
  are silently truncated. For large KBs, increase candidate_buf capacity or
  implement batched querying.
- ruleBodyEval calls self.query() recursively for each body condition. This is
  HOST-SIDE recursion (not GPU), bounded by body_count (typically 1-5).
  But each query may dispatch to GPU. So a rule with 3 body conditions may
  cause 3 GPU dispatches. For rules with many conditions, this is slow.
  Consider batching body condition evaluation into a single scoped search.
- fireAndCommit updates rule.fire_count and rule.last_fired by downloading
  the rule struct, modifying, and re-uploading. This is 2 transfers per fired
  rule. For 100 fired rules, that's 200 transfers. Consider batching: download
  all fired rules, modify on host, upload all at once.
- termStore advances kb_store.next_term_offset. This is a shared cursor.
  If two callers store terms concurrently, offsets collide. Currently impossible
  (single-threaded), but be aware if adding concurrency.

vlp_grammar.zig
───────────────
- compile reads the template from raw bytes, NOT from the text store. The caller
  passes the template as []const u8. But render reads it BACK from the text
  store (because compile stored it there via textAppend). If textAppend fails
  silently (text store full), render reads garbage.
- render walks slot_positions in order. If compile produced positions out of
  template order (it shouldn't, but verify), the output will have slots in
  wrong positions. The literal ranges between slots depend on slot_positions
  being sorted by template_offset.
- renderFromKb reads facts via kb_store.factRead. Each read may be a device
  download. For a grammar with 10 slots, that's 10 device reads. If any read
  fails (null), the fill is emptyFill which renders as nothing — silent gap
  in output.
- q16ToString produces "integer.fraction" format. The fraction is 4 decimal
  places. If you need more precision, change the `* 10000` constant. The
  fractional part is always non-negative — negative values show as "-N.FFFF".
- inherit walks parent chain with depth limit 100. If the KB tree is deeper
  than 100, grammars in ancestors beyond 100 are invisible. This matches
  the Prolog depth limit. Both limits should be the same value.

vlp_builtin.zig
───────────────
- builtinToPipeline maps builtin_id ranges to GPU pipelines. If you add a
  new builtin, you must add it to the correct range AND add the op_code case
  to the GPU kernel. If the mapping returns null, the builtin runs host-only
  regardless of array size.
- dispatchGpu extracts Q16.v values from facts and uploads as flat i32 array.
  The GPU kernel operates on i32 values, not Fact structs. Remainders (.r0)
  are discarded. If you need remainder-preserving GPU builtins, upload the
  full Q16 (8 bytes per element) and adjust the kernel.
- dispatchHost only handles the first 1-2 input slots. If a builtin needs more
  inputs (e.g., ternary operation), add the read. Currently falls through to
  returning the first input unchanged for unknown ops.
- Operational builtins (404-447) return ERR_INIT_FAILED as a stub. Each
  operational builtin needs its own implementation: filesystem ops need
  std.fs, network ops need std.net or HTTP client, execute ops need
  std.process.Child. These are pure host code, no GPU involvement.
- validateIoSe checks output KB is not frozen. It does NOT check that the
  output slot is within bounds. Add bounds check: slot_id < kb.facts_capacity.

vlp_confidence.zig
──────────────────
- combineAgreeing i64 accumulator: the product ∏(D - C_i) can overflow i64
  if N is large. D=65536, max complement = 65536. 65536^N overflows i64 at
  N=4 (65536^4 = 2^64). For N > 3, you must reduce the product modulo D
  at each step: `product = (product * complement) / D`. The implementation
  does this. Do not "optimize" by accumulating and dividing once at the end.
- propagate has a depth limit (max_depth=100) to prevent infinite loops from
  circular derivations. If fact A was derived from rule R which reads fact B
  which was derived from rule S which reads fact A, the depth limit kicks in
  and returns Q16.zero(). This is correct — circular derivation has undefined
  confidence.
- chain uses repeated Q16.mul. For large N (>20), precision degrades because
  each mul truncates. The .r0 remainder tells you how much was lost. If chain
  confidence matters for your use case, use Q32 for the intermediate and
  convert back to Q16 at the end.

vlp_session.zig
───────────────
- create scans session_active array linearly for a free slot. At 10K sessions,
  this is 10K bool checks worst case. If session creation is frequent, add a
  free-list. Currently fast enough — 10K checks at 1ns each = 10μs.
- clone calls kb_store.cowInit which allocates dirty_bits on host. If
  the fact region is large (e.g., 1M facts * 40 bytes = 40 MB), the dirty
  bit array is 40MB / 4096 = ~10K pages = ~1.25 KB. Tiny.
- merge iterates dirty pages. For each dirty page, it either copies or
  records a conflict. The number of dirty pages is bounded by writes the
  clone made. If a clone wrote to 100 facts across 10 pages, merge touches
  10 pages. Not 10K.
- incrementTurn updates session.current_turn. This is the ONLY place the
  turn counter advances. If you call forward/generate outside of
  inference.cycle, the turn counter doesn't increment and auto-snapshot
  doesn't trigger.

vlp_snapshot.zig
────────────────
- captureFromDevice downloads ALL regions, not just the session's owned
  subset. This means a snapshot includes every KB, every fact, every rule
  in the entire device. For multi-session systems, this is wasteful.
  Future: filter to session-owned regions using kb.owner or COW page table.
- restoreToDevice OVERWRITES the entire device state. If other sessions
  are active, their data is clobbered. Only restore when the device is
  dedicated to this session, or implement region-scoped restore.
- CRC32 is computed over all payload bytes AFTER the header. The header
  itself is NOT checksummed (because the checksum field is IN the header).
  If the header is corrupted, the checksum may be wrong, causing a false
  corrupt error. This is conservative — false negatives (accepting corrupt
  data) are worse than false positives (rejecting valid data).
- mergeThreeWay operates at REGION granularity, not fact granularity.
  If branch A modifies fact 5 in the fact region and branch B modifies
  fact 5000, the entire fact region is marked as conflicting because the
  diff compares regions as blobs. For fact-level merging, implement
  per-fact comparison within the region.

vlp_grant.zig
─────────────
- check scans the index linearly. Index is flat array, not hash map.
  For 100K grants, this scans 100K entries per check. At ~2ns per
  comparison, that's 200μs — acceptable but not great. If this becomes
  a bottleneck, sort the index by (user_id, grant_class) and binary search.
- consumeUse modifies the grant in-place. If the host crashes between
  consumeUse and the audit write, the use is consumed but not audited.
  For critical accounting, write audit BEFORE consuming, or make both
  atomic (not currently implemented).
- matchTarget reads pattern text from text_store via kb_store.textRead.
  This may be a device download if text_store is not host-mapped. For
  frequent grant checks, cache the pattern text on host at grant creation.
- isAdmin checks for ANY active grant with target "*". This is a convention,
  not a dedicated field. If an admin grant expires, the user loses admin
  silently. Audit will show the grant expiry but there's no notification.

vlp_access.zig
──────────────
- check calls kb_store.getKb per ancestor. The host cache makes this fast
  for recently accessed KBs. For a tree depth of 10, that's at most 10
  cache lookups (~10ns each). If the cache misses, each getKb is a device
  download (~10μs each). Pre-warm the cache for the session's KB subtree
  at session creation.
- resolveVisible does a depth-first scan of ALL KBs looking for children.
  This is O(N * depth) where N is total KBs. At 100K KBs and depth 10,
  that's 1M getKb calls. This is too slow for large systems. Replace with
  a host-side children index: parent_id → [child_ids]. Build at init,
  update on createKb/addChild.

vlp_audit.zig
─────────────
- Ring buffer head advances without locking. If two runner threads write
  simultaneously, entries are interleaved or corrupted. Either: one audit
  log per runner (merge at query time), or mutex around write. Current
  code assumes single writer.
- query scans the entire ring. At 1M entries and complex filter, this is
  slow. Add a secondary index on (session_id, timestamp) if audit queries
  are performance-sensitive.

vlp_command.zig
───────────────
- parse treats token[1] as direct kb_id. Real implementation needs a
  token→path→kb_id resolution: token[1] is a vocab index, look up the
  corresponding path string in command_vocab KB, then pathResolve. This
  is the missing piece between LLM output and system addressing.
- execute calls access.check which reads KB structs. Then it calls the
  engine function which also reads KB structs. These are potentially
  redundant reads. The KB cache prevents double device downloads, but
  verify the cache is warm.
- executeGrammarRender reads the template, compiles it, then renders.
  The compile step happens EVERY TIME. Cache compiled grammars keyed
  by (grammar_id, template_length) — if both match, skip compile.
- executeBatch aborts on first failure for non-query commands. If command
  3 of 5 fails, commands 4-5 never execute. Results[0..2] are populated,
  results[3..4] are undefined. Caller must check the returned Status.

vlp_inference.zig
─────────────────
- cycle calls llm.forward with the FULL context. This includes system
  prompt + input + scratchpad. If the context exceeds max_seq_len, the
  forward pass reads past the KV cache. Truncate context to max_seq_len
  BEFORE calling forward.
- The generation loop has no timeout. If the LLM never generates end_of_turn,
  the loop runs until max_tokens (2048). Add a wall-clock timeout for
  production — std.time.Timer, check after each token.
- buildContext uses session.kb_root_id for scope. If the session was
  cloned and the clone's root differs from parent's, the context reflects
  the clone's scope. This is correct — clones can have different views.
- classifyToken uses hardcoded token IDs (32000, 32001, 2). These must
  match the tokenizer's special tokens. Load from seed KB (command_vocab)
  instead of hardcoding.

vlp_runner.zig
──────────────
- Runner threads call inference.executeL3 or inference.cycle. Both of these
  use the bridge, which is NOT thread safe. If two runners dispatch GPU
  work simultaneously, command buffer corruption occurs. Solutions:
  (a) one bridge per runner (expensive — separate Vulkan devices), or
  (b) mutex around bridge.dispatch (serializes GPU work), or
  (c) multiple command buffers in the bridge with per-runner allocation.
  Option (b) is simplest and correct. Add a mutex to Bridge.dispatch.
- runProcessor has no actual external connection. The source_url config
  is stored but not used. Implement connection for each SourceConnectionType:
  http_poll → std.http.Client, websocket → custom, webhook → listen on port.
- runBatch reads task from factRead. If task is a complex structure (not
  just a single fact), the batch runner needs to know the task schema.
  Convention: task fact at slot N is TAG_COMPOUND with reference to a
  task description KB. The process function interprets this.
- recycle does stop→clone→kill→reassign→start. Between kill and start,
  the runner's session handle changes. Any state held by the runner
  thread that references the old session is now invalid. The implementation
  handles this by rebuilding the context, but verify no stale references
  leak through closures.

vlp_seed.zig
────────────
- createFresh hardcodes KB IDs 0-9. kb_store.createKb returns sequential
  IDs starting from next_kb_id=0. If anything creates a KB before seed
  init, the IDs are wrong and every well-known constant (ROOT_KB_ID,
  SYSTEM_KB_ID, etc.) points to the wrong KB. Seed MUST run first.
- Hygiene rules are stored as text descriptions, not compiled Prolog rules.
  The Prolog engine must parse these descriptions into Rule structs on
  first use. This parsing logic is not yet implemented — add it to
  prolog.ruleAssert or to a hygiene runner initialization step.
- populateCommandVocab stores 15 command type names. The full command
  vocabulary (~300 tokens per the original spec) includes argument
  patterns, KB path tokens, and end markers. Expand this to cover the
  full vocabulary needed by generateCommandTokens.

vlp_system.zig
──────────────
- init creates modules in dependency order. If module N fails, modules
  0..N-1 are already initialized. deinit must handle partial init —
  currently it destroys everything unconditionally, which may crash if
  a module's deinit touches an uninitialized predecessor. Add initialized
  flags per module, skip deinit for uninitialized modules.
- Checkpoint load failure is non-fatal (system can run L3 without LLM).
  Seed load failure IS fatal. If seed fails, init returns null. The caller
  gets no system and no error detail. Consider returning Status instead
  of ?*System for better error reporting.
- getSystemStatus reads counters from multiple modules. These reads are
  not atomic — session_count might change between reading it and reading
  runner_count. For monitoring this is fine. For billing, snapshot the
  counters under a lock.

vlp_multi_device.zig
────────────────────
- transferHiddenState is a STUB. It does nothing. Multi-device forward
  pass will produce wrong results until this is implemented. For host-
  staged transfer: downloadFromBuffer(src, scratch_b, hidden_state),
  uploadToBuffer(dst, scratch_a, hidden_state). For NVLink: use
  VK_KHR_external_memory to share buffers between devices.
- Layer assignment assumes even split. If model has 33 layers and 4
  devices, device 3 gets 9 layers, others get 8. The last device also
  runs lm_head. Uneven layer counts cause load imbalance. For production,
  profile per-layer latency and assign layers to minimize max-device time.
- replicateKb allocates a staging buffer on host for the fact data.
  Size = facts_count * 40 bytes. For a KB with 100K facts, that's 4 MB
  host allocation per replication. If many KBs are replicated, host memory
  spikes. Reuse a fixed staging buffer sized to the largest expected KB.
```
