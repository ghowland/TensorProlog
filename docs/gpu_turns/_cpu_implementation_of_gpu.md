```
================================================================================
VLP CPU-GPU Integration — Module Usage Specification
Per-function caller contract for all 22 modules.
================================================================================

================================================================================
MODULE: vlp_types.zig
Used by: every module. Import first.
Thread safety: all types are value types, no shared state.
GPU: extern structs cross the host-device boundary. Regular structs do not.
================================================================================

Q16.zero() -> Q16                                          — returns 0/65536
Q16.one() -> Q16                                           — returns 65536/65536
Q16.fromParts(v: i32, r0: i16) -> Q16                     — direct construction
Q16.add(a: Q16, b: Q16) -> Q16                            — exact add with remainder carry
Q16.sub(a: Q16, b: Q16) -> Q16                            — exact sub with borrow
Q16.mul(a: Q16, b: Q16) -> Q16                            — widening multiply, shift, remainder
Q16.div(a: Q16, b: Q16) -> Q16                            — integer division, caller checks b!=0
Q16.crossMultiplyCompare(a: Q16, b: Q16) -> i32           — returns -1/0/1
Q16.eql(a: Q16, b: Q16) -> bool                           — exact equality including remainder
Q32.zero() -> Q32                                          — returns 0/2^32
Q32.one() -> Q32                                           — returns 2^32/2^32
Q32.fromQ16(q: Q16) -> Q32                                — scale up, no precision loss
Q32.toQ16(self: Q32) -> Q16                               — scale down, remainder captured
Q335.zero() -> Q335                                        — returns 240 bytes of zeroes
Fact.empty() -> Fact                                       — tag=empty, zero provenance
Fact.isEmpty(self: Fact) -> bool                           — true if tag==empty
Provenance.direct(source, kb_id, slot_id, time) -> Provenance  — non-derived fact provenance
Provenance.derived(rule_id, kb_id, slot_id, conf, time) -> Provenance  — rule-derived provenance
Kb.isPublic/isInternal/isFrozen/isRoot(self) -> bool      — integer field checks
Kb.factSlotOffset(self, slot_id) -> i32                    — facts_offset + slot_id
Term.atom(id) -> Term                                      — construct atom term
Term.variable(id) -> Term                                  — construct variable term
Term.integer(val) -> Term                                  — construct integer term
Term.vdr(val: Q16) -> Term                                 — construct VDR term
Term.compound(functor, args_offset, args_count) -> Term    — construct compound term
Term.list(head_offset, tail_offset) -> Term                — construct list term
Term.textRef(offset, length) -> Term                       — construct text reference term
Term.isAtom/isVariable/isCompound(self) -> bool            — type checks
Rule.successRate(self) -> Q16                              — success_count / total as Q16
UnificationResult.success(offset, count) -> UnificationResult  — unified=1
UnificationResult.failure() -> UnificationResult           — unified=0
Grammar.isValid(self) -> bool                              — validated != 0
Grant.isActive/isUnlimited/isExpired/isExhausted(self) -> bool  — state checks
Grant.consumeUse(self: *Grant) -> bool                     — decrements remaining, returns false if exhausted
Command.requiresGrant(self) -> bool                        — grant_required >= 0
Command.grantClass(self) -> ?GrantClass                    — enum cast or null
Command.isOperational(self) -> bool                        — op_filesystem..op_process
Session.isActive/isClone/hasSnapshot(self) -> bool         — state checks
Session.turnsRemaining(self) -> i32                        — max_turns - current_turn, -1 if unlimited
Runner.shouldRecycle(self) -> bool                         — processor type + iterations >= max
Runner.shouldStop(self) -> bool                            — errors_consecutive >= max
Status.ok() -> Status                                      — category=none, code=ok
Status.err(cat, code, detail) -> Status                    — construct error status
Status.isOk/isErr(self) -> bool                            — category check
recoverFromError(status: Status) -> RecoveryAction         — deterministic switch on error code
LevelStats.totalCount(self) -> i64                         — l1+l2+l3
LevelStats.autoTriageNum/autoTriageDen(self) -> i32        — exact fraction numerator/denominator
LevelStats.avgTokensPerInteraction(self) -> Q16            — total_tokens / total_ops as Q16
confidence_table: [11]Q16                                  — immutable, indexed by SourceType

================================================================================
MODULE: vlp_device_memory.zig
Used by: vlp_bridge (at init), vlp_system (capacity planning).
GPU: defines buffer sizes. No GPU dispatch.
================================================================================

computeCapacity(config: *const SizingConfig) -> CapacityResult      — calculates bytes per region, total, n_devices
computeLayout(config: *const SizingConfig) -> DeviceMemoryLayout    — assigns base offsets sequentially
defaultSizingConfig() -> SizingConfig                                — 7B model, 10K sessions, reference config

================================================================================
MODULE: vlp_gpu_params.zig
Used by: vlp_llm, vlp_kb_store, vlp_prolog, vlp_builtin, vlp_confidence.
GPU: these structs are written to the uniform buffer before each dispatch.
All are extern struct, 16-byte aligned.
================================================================================

No functions. Pure type declarations.
Caller constructs param struct, passes to bridge.writeParams() or bridge.dispatch().
PipelineId.count = 28 — total compute pipelines.
DescriptorSet enum: model=0, kb_data=1, scratch=2, control=3.
ModelBindings/KbDataBindings/ScratchBindings/ControlBindings — binding indices within each set.
MAX_WORKGROUP_SIZE = 256, WARP_SIZE = 32.
SHARED_MEM_BASELINE/EXTENDED/H100 — tier thresholds in bytes.

================================================================================
MODULE: vlp_bridge.zig
Used by: every module that touches GPU. Central dependency.
GPU: owns all Vulkan resources. Only module that calls Vulkan.
Thread safety: NOT thread safe. Caller must serialize dispatch calls.
================================================================================

init(allocator, config: *const BridgeConfig) -> Bridge                    — creates Vulkan instance/device/pipelines/buffers/descriptors
deinit(self: *Bridge) -> void                                             — destroys all Vulkan resources in reverse order
dispatch(self, config: *const DispatchConfig) -> Status                   — submit+fence+wait, checks status buffer after
dispatchAsync(self, config: *const DispatchConfig) -> VkFence             — submit, returns fence, caller must waitFence before reading results
dispatchSequence(self, configs: []const DispatchConfig) -> Status         — multiple dispatches in one command buffer, one submit, one fence
waitFence(self, fence: VkFence, timeout_ns: u64) -> Status               — blocks until fence signals or timeout
uploadToBuffer(self, target: BufferTarget, offset: i64, data: []const u8) -> Status    — host→device, mapped or staged
downloadFromBuffer(self, source: BufferTarget, offset: i64, dest: []u8) -> Status      — device→host, mapped or staged
copyBufferToBuffer(self, src, src_off, dst, dst_off, size) -> Status      — device-to-device copy via command buffer
fillBuffer(self, target, offset, size, value: u32) -> Status              — vkCmdFillBuffer
getMappedPtr(self, target: BufferTarget) -> ?[*]u8                        — returns mapped pointer or null
isMapped(self, target: BufferTarget) -> bool                              — true if host-visible
readStatus(self, invocation_index: i32) -> i32                            — reads status_buffer[i], mapped or staged
readResultCount(self, slot: i32) -> i32                                   — reads result_counts[slot]
resetStatusBuffer(self) -> Status                                         — fills status buffer with 0
resetResultCounts(self) -> Status                                         — fills result counts with 0
updateModelDescriptors(self) -> Status                                    — bind Set 0, called once at model load
updateKbDescriptors(self, session_kb_offset, session_fact_offset) -> Status  — rebind Set 1 for session switch
updateScratchDescriptors(self) -> Status                                  — rebind Set 2 if scratch region changed
updateControlDescriptors(self) -> Status                                  — rebind Set 3 if control buffers changed
shouldUseGpu(self, op: OperationType, data_size: i32) -> bool            — threshold check, returns false if not initialized
sharedMemoryTier(self) -> SharedMemoryTier                                — baseline/extended/h100 from device properties
writeParams(self, comptime T, params: *const T) -> Status                — uploads typed struct to params uniform buffer
readScratchSlice(self, comptime T, target, offset, count, out: []T) -> Status  — downloads typed array from scratch buffer
deviceName(self) -> []const u8                                            — cached device name string
totalDeviceMemory(self) -> i64                                            — cached total device memory in bytes
maxWorkgroupSize(self) -> i32                                             — cached max invocations per workgroup

================================================================================
MODULE: vlp_llm.zig
Used by: vlp_inference (forward pass + generation), vlp_system (init + checkpoint load).
GPU: dispatches 12 kernels per layer via bridge. Sampling is host-only.
Thread safety: NOT thread safe. One LlmEngine per system, serialized by inference loop.
Caller must: init bridge first, load checkpoint before forward pass.
================================================================================

init(bridge, config: *const ModelConfig, allocator) -> LlmEngine          — computes attention scale, sets up KV config
deinit(self: *LlmEngine) -> void                                          — resets state, does not free bridge resources
loadCheckpoint(self, path: []const u8) -> Status                           — reads file, uploads weights to model_weights buffer
validateChecksum(self) -> Status                                           — downloads weights, computes CRC32, compares to header
kvCacheClear(self) -> Status                                               — zeros kv_cache buffer, resets seq_len to 0
kvCacheTruncate(self, position: i32) -> Status                             — zeros entries beyond position, updates seq_len
kvCacheSeqLen(self) -> i32                                                 — current sequence length
forward(self, input_ids: []const i32) -> ForwardResult                     — full forward pass, dispatches all layer kernels, returns logit location
forwardSingleToken(self, token_id: i32) -> ForwardResult                   — forward with single token, wraps forward()
generateToken(self, sampling: *const SamplingConfig) -> i32                — forward + download logits + sample on host
sampleFromLogits(logits: []const i32, config: *const SamplingConfig) -> i32  — host-side dispatch to greedy/topk/topp/temperature
sampleGreedy(logits: []const i32) -> i32                                   — argmax scan
sampleTopK(logits, k, temperature_v) -> i32                                — partial sort + normalized sample
sampleTopP(logits, p_v, temperature_v) -> i32                              — sorted accumulation to threshold
sampleTemperature(logits, temperature_v) -> i32                            — scale logits then greedy
generateCommandTokens(self, command_vocab, max_tokens, output) -> i32      — constrained generation, masks non-vocab tokens
generateProse(self, sampling, max_tokens, output) -> i32                   — unconstrained generation until EOS

================================================================================
MODULE: vlp_kb_store.zig
Used by: vlp_prolog, vlp_grammar, vlp_builtin, vlp_command, vlp_session,
         vlp_snapshot, vlp_confidence, vlp_seed, vlp_access.
GPU: dispatches fact_write_batch, fact_read_batch, fact_scan_by_tag, scoped_search
     when data size exceeds threshold. Host path for small operations.
Thread safety: NOT thread safe. Serialized by caller (inference loop or runner thread).
Caller must: init bridge first.
================================================================================

init(bridge, allocator, max_kbs: i32) -> KbStore                          — allocates path index, host cache, search buffers, COW list
deinit(self: *KbStore) -> void                                             — frees all host allocations, destroys COW tables
createKb(self, config: *const KbCreateConfig) -> i32                       — allocates KB struct + fact region, registers path, returns kb_id
getKb(self, kb_id: i32) -> ?Kb                                            — host cache hit or device download, returns null if invalid
freezeKb(self, kb_id: i32) -> Status                                      — sets frozen=1, writes to device
setVisibility(self, kb_id: i32, visibility: i8) -> Status                  — updates visibility field, writes to device
pathResolve(self, path: []const u8) -> ?i32                                — host-side hash lookup, returns kb_id or null
pathRegister(self, path: []const u8, kb_id: i32) -> Status                 — host-side hash insert
pathRemove(self, path: []const u8) -> Status                               — host-side hash delete with rehash
factWrite(self, kb_id, slot_id, fact: *const Fact) -> Status               — bounds check, COW check, upload to device, update KB metadata
factRead(self, kb_id, slot_id) -> ?Fact                                    — download from device, returns null if empty or out of range
factRetract(self, kb_id, slot_id) -> Status                                — writes Fact.empty() to slot
factWriteBatch(self, kb_id, slot_ids, facts) -> Status                     — GPU dispatch if > 256, host loop otherwise
factReadBatch(self, kb_id, slot_ids, out) -> Status                        — GPU dispatch if > 256, host loop otherwise
factScanByTag(self, kb_id, tag: FactTag, max_results) -> SearchResult      — GPU dispatch if facts_count > 256, host scan otherwise
scopedSearch(self, config: *const ScopedSearchConfig) -> SearchResult       — host builds chain, calls factScanByTag per KB
buildChain(self, start_kb_id, max_depth) -> []ChainEntry                   — walks parent_id chain, returns (kb_id, facts_offset, facts_count) array
textAppend(self, data: []const u8) -> i32                                  — uploads to text_store buffer, returns offset
textRead(self, offset, length, buf) -> Status                              — downloads from text_store buffer
addChild(self, parent_id, child_id) -> Status                              — increments parent children_count
removeChild(self, parent_id, child_id) -> Status                           — decrements parent children_count
addMount(self, kb_id, source_kb_id, mount_name) -> Status                  — increments mounts_count
removeMount(self, kb_id, mount_name) -> Status                             — decrements mounts_count
cowInit(self, parent_session, clone_session, region_size, parent_off, private_off) -> Status  — allocates CowPageTable with dirty bits
cowDestroy(self, clone_session_id) -> void                                 — frees COW table for clone
cowResolve(self, clone_session_id) -> Status                               — copies all non-dirty pages, making clone independent
currentTimestamp() -> i32                                                   — host wall clock truncated to i32

================================================================================
MODULE: vlp_prolog.zig
Used by: vlp_command (PROLOG_QUERY, PROLOG_ASSERT_RULE), vlp_inference (L2, L3),
         vlp_runner (poller fires rules).
GPU: dispatches unify_candidates, rule_match_scan, rule_body_eval, rule_check_satisfied
     when candidate/rule count exceeds threshold.
Thread safety: NOT thread safe. Owns reusable buffers.
Caller must: init bridge and kb_store first.
================================================================================

init(bridge, kb_store, allocator, config: *const QueryConfig) -> PrologEngine  — allocates binding/stack/candidate/result buffers
deinit(self: *PrologEngine) -> void                                            — frees all buffers
unifySingle(a, b: *const Term, bindings, binding_count: *i32) -> bool          — flat unification, no recursion, handles atom/var/int/vdr/compound
unifyCandidatesGpu(self, query_term, candidate_offsets) -> i32                 — uploads offsets, dispatches GPU kernel, returns match count
query(self, start_kb_id, query_term: *const Term) -> QueryResult               — builds chain, collects candidates, GPU or host unification, returns bindings
ruleMatchScan(self, kb_id, query_term) -> []i32                                — parallel head matching, GPU if > 64 rules, returns matched rule IDs
ruleBodyEval(self, matched_rule_ids, kb_id) -> []bool                          — evaluates body conditions per matched rule, sequential sub-queries
fireRules(self, kb_id) -> FireResult                                           — match + body eval + collect fully satisfied rules
applyActions(self, actions: []const PrologAction) -> Status                    — asserts/retracts facts per action
fireAndCommit(self, kb_id) -> i32                                              — fireRules + update rule stats, returns fire count
ruleAssert(self, kb_id, head, body, actions) -> i32                            — stores terms, builds Rule struct, uploads, returns rule_id
ruleRetract(self, kb_id, rule_id) -> Status                                    — zeroes rule in rule_store
ruleGet(self, rule_id) -> ?Rule                                                — downloads rule from device
termStore(self, term: *const Term) -> i32                                      — uploads term, returns offset
termStoreBatch(self, terms: []const Term) -> i32                               — uploads N terms contiguously, returns start offset
termLoad(self, offset: i32) -> ?Term                                           — downloads term from device

================================================================================
MODULE: vlp_grammar.zig
Used by: vlp_command (GRAMMAR_RENDER, DIRECT_OUTPUT), vlp_seed (format templates).
GPU: NONE. Entirely host-side.
Thread safety: NOT thread safe. Owns reusable render buffer.
Caller must: init kb_store first.
================================================================================

init(allocator, kb_store) -> GrammarEngine                                     — allocates render, slot, literal, position buffers
deinit(self: *GrammarEngine) -> void                                           — frees all buffers
compile(self, template, grammar_id, session_id) -> CompileResult               — parses {slot} markers, builds literal ranges + slot positions
validate(self, grammar: *const Grammar) -> Status                              — re-parses template, checks brace matching
render(self, grammar, compiled, fills, config, output) -> i32                  — walks template, copies literals, renders fills, returns byte count
renderFromKb(self, grammar, compiled, mappings, config, output) -> i32         — reads facts from KB, builds fills, delegates to render()
inherit(self, kb_id, grammar_slot) -> ?Grammar                                 — walks parent chain looking for grammar at slot
q16ToString(value: Q16, buf) -> i32                                            — integer_part.fractional_part as decimal text
i32ToString(value: i32, buf) -> i32                                            — decimal text conversion

================================================================================
MODULE: vlp_builtin.zig
Used by: vlp_command (BUILTIN_CALL).
GPU: dispatches builtin_unary/binary/reduction/sort/matmul/confidence pipelines
     when array length exceeds threshold. Host path for small arrays and operational builtins.
Thread safety: NOT thread safe.
Caller must: init bridge and kb_store first.
================================================================================

init(bridge, kb_store, allocator) -> BuiltinExecutor                           — builds IOSE table for 448 builtins
deinit(self: *BuiltinExecutor) -> void                                         — resets initialized flag
getIoSe(self, builtin_id) -> ?IoSe                                            — returns IOSE declaration for builtin
isOperational(builtin_id: i32) -> bool                                         — true if id >= 404
requiredGrant(self, builtin_id) -> ?GrantClass                                — returns grant class or null for pure builtins
validateIoSe(self, builtin_id, args: *const BuiltinArgs) -> Status             — checks arg count, types, output KB writable
dispatch(self, builtin_id, args: *const BuiltinArgs) -> BuiltinResult          — validates, routes to GPU/host/operational, writes result to KB

================================================================================
MODULE: vlp_confidence.zig
Used by: vlp_command (on KB_ASSERT sets provenance), vlp_prolog (derived fact confidence).
GPU: dispatches builtin_confidence_combine for N > 64 sources.
Thread safety: stateless functions except propagate which reads KB.
================================================================================

assign(fact: *Fact, source_type: SourceType) -> void                           — sets fact.provenance.confidence from confidence_table
combineAgreeing(bridge, confidences: []const Q16) -> Q16                       — 1 - ∏(1-C_i), GPU for N>64
combineConflicting(bridge, confidences, penalty: Q16) -> Q16                   — agreeing + penalty per conflict pair
chain(per_link: Q16, n_links: i32) -> Q16                                      — repeated Q16.mul, always host
propagate(kb_store, kb_id, slot_id) -> Q16                                     — walks derivation_rule_id chain, returns accumulated confidence
isHighConfidence(c: Q16) -> bool                                               — c.v >= 52428 (80%)
isMediumConfidence(c: Q16) -> bool                                             — c.v >= 32768 and < 52428
isLowConfidence(c: Q16) -> bool                                                — c.v < 32768
confidenceToPercent(c: Q16) -> i32                                             — integer 0-100

================================================================================
MODULE: vlp_session.zig
Used by: vlp_command, vlp_inference, vlp_runner, vlp_system.
GPU: none directly. Delegates to kb_store (COW) and bridge (buffer copy) for clone/merge.
Thread safety: NOT thread safe. Session array access serialized by caller.
Caller must: init bridge and kb_store first.
================================================================================

init(bridge, kb_store, allocator, max_sessions) -> SessionManager              — allocates session array, stats, conflict buffer
deinit(self: *SessionManager) -> void                                          — frees all arrays
create(self, config: *const SessionConfig) -> ?SessionHandle                   — finds free slot, initializes Session struct
destroy(self, handle) -> Status                                                — destroys COW, marks slot free
get(self, handle) -> ?*Session                                                 — returns pointer to live session or null
kill(self, handle) -> Status                                                   — immediate destroy, no snapshot
clone(self, parent_handle, config: *const CloneConfig) -> ?SessionHandle       — creates child session, sets up COW, copies live state if requested
merge(self, parent, child, policy: MergePolicy) -> MergeResult                 — copies dirty COW pages per policy, detects conflicts
updateLevelStats(self, handle, level: i8, tokens: i32) -> Status               — increments L1/L2/L3 counter
getLevelStats(self, handle) -> LevelStats                                      — returns current level distribution
shouldAutoSnapshot(self, handle) -> bool                                       — current_turn mod interval == 0
incrementTurn(self, handle, llm_tokens, command_tokens) -> Status              — updates session counters
findByUserId(self, user_id) -> ?SessionHandle                                 — scans for active session with matching user
activeCount(self) -> i32                                                       — current session_count

================================================================================
MODULE: vlp_snapshot.zig
Used by: vlp_session (snapshot/restore), vlp_system (snapshotSession), vlp_seed (load from snapshot).
GPU: downloads/uploads all device regions via bridge.
Thread safety: NOT thread safe.
Caller must: fence all GPU work before capture. Resume after restore.
================================================================================

init(allocator, bridge) -> SnapshotManager                                     — stores allocator + bridge reference
deinit(self: *SnapshotManager) -> void                                         — no-op (no persistent state)
captureFromDevice(self, session: *const Session) -> ?SnapshotHandle            — downloads all regions, builds header, computes CRC32
restoreToDevice(self, data: []const u8, session: *Session) -> Status           — validates magic+version+checksum, uploads all regions, restores session metadata
save(self, data: []const u8, path: []const u8) -> Status                       — writes snapshot blob to file
load(self, path: []const u8) -> ?[]u8                                          — reads file into allocated buffer
freeData(self, data: []u8) -> void                                             — frees snapshot buffer
diff(self, a: []const u8, b: []const u8) -> DiffResult                        — compares regions, returns entries where content differs
mergeThreeWay(self, base, branch_a, branch_b, policy) -> ?[]u8                — applies non-conflicting changes, resolves conflicts per policy
validateChecksum(data: []const u8) -> bool                                     — verifies magic + CRC32
computeChecksum(data: []const u8) -> i32                                       — CRC32 over raw bytes

================================================================================
MODULE: vlp_grant.zig
Used by: vlp_command (grant check before operational commands).
GPU: NONE. Entirely host-side.
Thread safety: NOT thread safe.
Caller must: init kb_store first (for pattern text reads).
================================================================================

init(allocator, kb_store, max_grants) -> GrantEnforcer                         — allocates grant array + index
deinit(self: *GrantEnforcer) -> void                                           — frees arrays
check(self, session, grant_class, target: []const u8) -> GrantResult           — scans index, checks state/expiry/uses/pattern, consumes use if granted
create(self, admin_session, grant: *const Grant) -> Status                     — validates admin privilege, appends to grant array + index
revoke(self, admin_session, grant_id) -> Status                                — sets state=revoked, records revoked_at/by
list(self, user_id, out: []Grant) -> i32                                       — returns all grants for user
cleanup(self) -> i32                                                            — marks expired/exhausted grants, returns count cleaned

================================================================================
MODULE: vlp_access.zig
Used by: vlp_command (before every execute), vlp_inference (context building).
GPU: NONE. Integer comparisons on cached KB structs.
Thread safety: stateless — reads from kb_store which must be serialized.
================================================================================

check(checker, session, kb_id) -> bool                                         — walks parent chain, returns false if any ancestor fails visibility
resolveVisible(checker, session, scope_kb_id, visible: []i32) -> i32           — enumerates accessible KBs, prunes at invisible ancestors

================================================================================
MODULE: vlp_audit.zig
Used by: vlp_command (after every execute), vlp_grant (after check), vlp_access (on denial).
GPU: NONE. Host-side ring buffer.
Thread safety: NOT thread safe. Single writer assumed (inference loop or runner).
================================================================================

init(allocator, capacity: i32) -> AuditLog                                     — allocates entry array, zeroes
deinit(log: *AuditLog, allocator) -> void                                      — frees entry array
write(log, entry: *const AuditEntry) -> void                                   — appends to ring, wraps oldest
writeAllowed(log, time, session, user, action, kb, slot) -> void               — convenience: constructs allowed entry + writes
writeDenied(log, time, session, user, action, kb, slot) -> void                — convenience: constructs denied entry + writes
writeGrantCheck(log, time, session, user, kb, grant_id, granted) -> void       — convenience: grant check audit entry
query(log, filter: *const AuditFilter, out: []AuditEntry) -> i32              — scans ring with filter, returns match count
count(log, filter) -> i32                                                       — counts matches without copying
latest(log, n, out: []AuditEntry) -> i32                                       — returns N newest entries
totalWritten(log) -> i64                                                        — lifetime write count (may exceed capacity)
currentSize(log) -> i32                                                         — current entries in ring
isFull(log) -> bool                                                             — count >= capacity

================================================================================
MODULE: vlp_command.zig
Used by: vlp_inference (command dispatch during generation loop).
GPU: indirectly via kb_store, prolog, builtins which may dispatch to GPU.
Thread safety: NOT thread safe. Owns arg_buf and render_buf.
Caller must: init all engine modules first.
================================================================================

init(kb_store, prolog, grammar, builtins, grants, access, audit, session_mgr, allocator) -> CommandProcessor
deinit(self: *CommandProcessor) -> void                                        — frees arg_buf and render_buf
parse(self, tokens: []const i32) -> ?Command                                   — first token→type, second→kb_id, third→slot, rest→args
parseKbUrl(self, text: []const u8) -> KbUrl                                    — strips kb://, splits on last dot, resolves path→kb_id
execute(self, handle: SessionHandle, command: *const Command) -> CommandResult  — access check → grant check → dispatch by type → audit
executeBatch(self, handle, commands, results) -> Status                         — sequential execute, aborts on first non-query failure

================================================================================
MODULE: vlp_inference.zig
Used by: vlp_runner (all runner types call cycle or executeL3), vlp_system (handleUserInput).
GPU: via llm (forward pass), via command processor (which may trigger GPU ops).
Thread safety: NOT thread safe. Owns scratchpad, context, token buffers.
Caller must: init all modules. Load model checkpoint. Populate seed.
================================================================================

init(session_mgr, llm, commands, kb_store, allocator, context_config) -> InferenceEngine
deinit(self: *InferenceEngine) -> void                                         — frees scratchpad, context, token buffers
cycle(self, handle, input: []const u8, output: *OutputBuffer) -> Status        — full inference: tokenize→context→forward→generate loop→post-process→auto-snapshot
executeL1(self, handle, input, output) -> Status                               — delegates to cycle(), full LLM judgment
executeL2(self, handle, pattern: *const Term) -> Status                        — prolog.query() + level stats update
executeL3(self, handle, kb_id: i32) -> Status                                  — prolog.fireAndCommit() + level stats update, zero LLM tokens
OutputBuffer.init(allocator, capacity) -> OutputBuffer                         — allocates byte buffer
OutputBuffer.deinit(self, allocator) -> void                                   — frees buffer
OutputBuffer.append(self, bytes) -> void                                       — copies bytes, respects capacity
OutputBuffer.appendByte(self, b) -> void                                       — single byte append
OutputBuffer.reset(self) -> void                                               — sets length to 0
OutputBuffer.contents(self) -> []const u8                                      — returns data[0..length]
Scratchpad.init(allocator, capacity) -> Scratchpad                             — allocates entry array
Scratchpad.deinit(self, allocator) -> void                                     — frees entries
Scratchpad.write(self, cmd_idx, result) -> void                                — appends entry
Scratchpad.clear(self) -> void                                                 — sets count to 0

================================================================================
MODULE: vlp_runner.zig
Used by: vlp_system (deployment setup).
GPU: indirectly via inference engine.
Thread safety: each runner runs on its own std.Thread.
              RunnerScheduler itself is NOT thread safe — create/start/stop from main thread.
Caller must: init session_mgr and inference engine first.
================================================================================

init(allocator, session_mgr, inference, max_runners) -> RunnerScheduler        — allocates runner/context/thread arrays
deinit(self: *RunnerScheduler) -> void                                         — stops all runners, joins threads, frees arrays
createPoller(self, config: *const PollerConfig) -> ?RunnerHandle               — allocates slot, builds runner + context
createProcessor(self, config: *const ProcessorConfig) -> ?RunnerHandle         — allocates slot with recycle/backoff config
createInternal(self, config: *const InternalConfig) -> ?RunnerHandle           — allocates slot, same loop as poller
createBatch(self, config: *const BatchConfig) -> ?RunnerHandle                 — allocates slot with task queue config
start(self, handle) -> Status                                                   — spawns thread, sets state=running
stop(self, handle) -> Status                                                    — sets stop flag, joins thread, state=stopped
kill(self, handle) -> Status                                                    — sets kill flag, joins thread, state=stopped
recycle(self, handle) -> Status                                                 — stop→clone session→kill old session→point runner at clone→start
getStatus(self, handle) -> RunnerStatus                                        — returns state/iterations/errors/timing
listAll(self, out: []RunnerStatus) -> i32                                      — returns status of all active runners

================================================================================
MODULE: vlp_seed.zig
Used by: vlp_system (at init, before any session creation).
GPU: indirectly via kb_store (creates KBs, writes facts to device buffers).
Caller must: init kb_store first. Seed runs before any sessions exist.
================================================================================

init(kb_store, config: *const SeedConfig) -> Status                            — tries snapshot load, falls back to createFresh
createFresh(kb_store) -> Status                                                — creates 10 KBs, populates content, freezes all
populateOso(kb_store) -> Status                                                — 15 engineering principles as text facts
populateConfidenceTable(kb_store) -> Status                                    — 11 confidence_table entries as value facts
populateCommandVocab(kb_store) -> Status                                       — 15 command type names as text facts
populateHygieneRules(kb_store) -> Status                                       — 3 hygiene rule descriptions as rule_ref facts

================================================================================
MODULE: vlp_system.zig
Used by: application main(). Single entry point.
GPU: indirectly via all modules.
Thread safety: handleUserInput is NOT thread safe per session.
              Different sessions on different threads require separate inference engines.
================================================================================

init(allocator, config: *const SystemConfig) -> ?*System                       — creates all 14 modules in dependency order, loads checkpoint, seeds KBs
deinit(system: *System) -> void                                                — destroys all modules in reverse order, frees System
handleUserInput(system, handle, input, output) -> Status                       — delegates to inference.cycle()
createSession(system, user_id) -> ?SessionHandle                               — creates session with system defaults
destroySession(system, handle) -> Status                                       — delegates to session_mgr.destroy()
snapshotSession(system, handle) -> ?SnapshotHandle                             — delegates to snapshot_mgr.captureFromDevice()
getSystemStatus(system) -> SystemStatus                                        — aggregates counts from all modules
recoverFromError(system, handle, err: Status) -> Status                        — maps error to RecoveryAction, executes recovery

================================================================================
MODULE: vlp_multi_device.zig
Used by: vlp_system (when n_devices > 1).
GPU: creates one bridge per device, dispatches layer shards across devices.
Thread safety: NOT thread safe.
Caller must: have multiple Vulkan-capable devices.
================================================================================

init(allocator, config: *const MultiDeviceConfig) -> ?*MultiDeviceManager      — assigns layer ranges, creates bridge per device
deinit(mgr: *MultiDeviceManager) -> void                                       — destroys all bridges
forward(mgr, input_ids, logits: []i32) -> Status                              — pipeline parallel forward: device0→transfer→device1→...→logits
replicateKb(mgr, source_device, target_device, kb_id) -> Status               — downloads KB+facts from source, uploads to target
syncKb(mgr, kb_id) -> Status                                                   — broadcasts from device 0 to all others

================================================================================
MODULE: vlp_test.zig
Used by: test harness, CI.
GPU: indirectly via modules under test.
================================================================================

testDeterminism(test_fn, n_runs) -> TestResult                                 — runs N times, compares status bit-by-bit
testSnapshotRoundtrip(session_mgr, snap_mgr, kb_store, handle) -> TestResult   — snapshot→modify→restore→verify identical
testCloneIndependence(session_mgr, kb_store, parent) -> TestResult             — write to clone, verify parent unchanged
testAccessIsolation(session_mgr, kb_store, session_a, session_b) -> TestResult — OWNER_ONLY KB invisible to non-owner
testConfidencePropagation(kb_store, bridge) -> TestResult                      — verifies table values, combine, chain, mul arithmetic
testSoftmaxSumInvariant(probs, denominator) -> TestResult                      — verifies sum == D exactly
runFullSuite(allocator, session_mgr, snap_mgr, kb_store, bridge) -> TestSuiteResult  — runs all tests, returns total/passed/failed
```
