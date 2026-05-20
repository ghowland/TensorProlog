# TensorProlog

Integer-only GPU compute stack for exact LLM inference, knowledge operations, and autonomous session management.

Zig 0.15.1. No float anywhere. All arithmetic uses VDR (Value/Denominator/Remainder) triples with fixed power-of-2 denominators.

## What This Is

TensorProlog replaces the CUDA/cuBLAS/cuDNN stack with integer arithmetic that produces deterministic, bit-identical results across runs, devices, and reduction topologies. The system combines an LLM inference engine with a Prolog deduction engine, grammar-directed structural output, bounded knowledge bases, session lifecycle with copy-on-write snapshots, four autonomous runner types, and a multi-protocol server — all operating on exact integer values with zero floating-point operations.

The LLM is one component in a larger system. It handles the 7% of operations that require judgment. The other 93% resolve through integer Prolog unification, grammar rendering, and KB queries at nanosecond cost.

## Core Idea

Every value in the system is a Q16 triple:

```
Q16 = { v: i32, r0: i16, _pad: i16 }
D = 65536 (implicit, never stored)
```

Multiply: `i64 product = a.v * b.v; result.v = product / D; result.r0 = product % D`

Softmax: shift inputs so min=0, square each, divide by sum. Last element absorbs rounding remainder. Sum equals D exactly, verified by integer equality, not tolerance.

This single property — exact integer arithmetic — eliminates NaN propagation, loss scaling, mixed-precision management, non-deterministic allreduce, tolerance-based testing, and the entire complexity layer that exists to manage floating-point failure modes.

## How It Works

### The Universal Cycle

Everything in the system runs one function:

```
vlp_cycle(session, input, output, kb_store, llm_engine, stream) → cycle_result
```

**Phase 0 — Pre-LLM Rule Evaluation:**
Fire all matching Prolog rules before the LLM sees anything. If rules produce a finding with a grammar reference and confidence above threshold, render the response from KB facts and return. Zero tokens consumed. This is the L3 path.

**Phase 1 — Context Assembly:**
Build what the LLM actually sees: system prompt (~200 tokens), scope reference (~5 tokens), scratchpad (~0-50 tokens), user input. Total: ~300-600 tokens, bounded, constant regardless of turn number. Prior turns are in KBs. History is in rules. Formatting is in grammars.

**Phase 2 — LLM Generation + Command Dispatch:**
Generate tokens one at a time. Classify each as COMMAND_START, DIRECT_OUTPUT, END_OF_TURN, or PROSE. Commands go through access check (integer comparison) then grant check (four integer comparisons) then dispatch. Direct output references resolve from KB through inherited grammar. Prose passes to output.

**Phase 3 — Post-Cycle:**
Update session counters. Auto-snapshot if configured. Check turn budget for recyclable runners.

### Execution Levels

- **L1 — Full LLM:** 50-500 tokens. Novel situations requiring judgment.
- **L2 — LLM Invokes Rule:** ~8 command tokens + ~10 prose tokens. LLM recognizes stored pattern.
- **L3 — Automatic Rule Fire:** 0 tokens. Prolog handles everything. 93% of mature operations.

### Data Flow

```
User Input
    │
    ▼
┌─────────────────────────┐
│ Phase 0: Fire Prolog    │──→ L3 resolution? ──→ Grammar render ──→ Output
│ rules against KB facts  │                                         (0 tokens)
└─────────┬───────────────┘
          │ (not resolved)
          ▼
┌─────────────────────────┐
│ Phase 1: Build context  │  ~350 tokens (bounded, constant)
│ prompt + scope + scratch│
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│ Phase 2: LLM generates  │
│ tokens, classifies each │
│                         │
│ COMMAND ──→ access check ──→ grant check ──→ execute
│            (int compare)    (4 int compares)   (KB/Prolog/builtin)
│                                                    │
│ DIRECT_OUTPUT ──→ load from KB ──→ grammar render  │
│                                                    │
│ PROSE ──→ pass to output                           │
│                                                    │
│ END_OF_TURN ──→ break                              │
└─────────┬───────────────┘                          │
          │                                          │
          ▼                                          ▼
┌─────────────────────────┐              ┌──────────────────┐
│ Phase 3: Update counters│              │ Results to       │
│ auto-snapshot, recycle   │              │ scratchpad (LLM  │
│ check                   │              │ inspects next     │
└─────────────────────────┘              │ generation step) │
                                         └──────────────────┘
```

### Security Model

Data the session cannot access never enters the LLM context. Not filtered, not redacted — absent.

```
vlp_access_check(session, kb_id):
  walk from kb_id up parent chain
  at each ancestor: compare visibility integer against session level
  any ancestor fails → entire subtree invisible
  all integer comparisons — no prompt modifies any value in this check
```

Operational primitives (filesystem, network, execute) require positive grants:

```
vlp_grant_check(session, grant_class, target):
  state == ACTIVE?        (integer comparison)
  not expired?            (integer comparison)
  uses remaining?         (integer comparison)
  target pattern match?   (prefix comparison)
  → four checks, all integer, all before any side effect
```

### Session Lifecycle

```
create ──→ active session with KB root, user_id, visibility
clone  ──→ COW page table sharing parent's persistent KBs
           writes to shared pages trigger copy-on-write
           parent never sees clone's modifications
merge  ──→ apply clone's dirty pages back to parent (ours/theirs/fail policy)
snapshot → atomic capture: KBs + facts + rules + live state + CRC32 checksum
restore → validate checksum (mismatch = hard fail), overwrite all state
           bit-identical because integers
kill   ──→ immediate free, no snapshot, drift dies, parent's facts survive
```

The disposable clone pattern: snapshot → run workers → kill on drift → launch fresh from same frozen baseline. Knowledge persists through the snapshot. Accumulated drift dies with the killed session.

### Runner Types

All four runner types call the same `vlp_cycle` function:

**Poller:** Timer fires → synthetic input → Phase 0 fires rules → if fully resolved, zero LLM tokens → output to notification KB. Repeats every interval.

**Processor:** External data arrives → try rule-based compaction (L3) → fall through to LLM (L1) → LLM writes rules so novel becomes known → recycle at turn threshold (snapshot → kill → clone → restore connection). Continuous.

**Internal:** Timer fires → compute derived facts from existing facts → all exact integer arithmetic → zero LLM tokens. Rolling averages, trend detection, coverage gaps.

**Batch:** Pop task from KB queue → clone session → run cycle in clone → merge results back → kill clone. Concurrent up to configured max.

### Server Architecture

```
TCP accept
    │
    ▼
Capacity check (integer comparison against max_connections)
    │
    ▼
Find free connection slot
    │
    ▼
Protocol handshake (compiled parser, not LLM)
    │
    ▼
Authenticate (hash token → scan auth KB → load grants → issue credential with integer TTL)
    │
    ▼
Clone session from template snapshot (bit-identical starting state + unique user_id/grants)
    │
    ▼
Request loop:
  credential check (two integer comparisons: valid flag + timestamp)
  rate limit check (per-user counter in KB, exact threshold)
  read request → parse → vlp_cycle → grammar render response → send
    │
    ▼
Close: kill session (or snapshot if persistent), free resources
```

Every structural byte in the response comes from a grammar template. The LLM generates content. The grammar generates structure. 100% correct JSON, HTTP headers, protocol framing — by construction, not by hope.

## Directory Structure

```
src/
├── vdr/                    Q16/Q32/Q335 arithmetic, type definitions
│   ├── types.zig           VlpStatus (24 error codes), all enums (VlpFactTag,
│   │                       VlpSourceType, VlpVisibility, VlpRunnerType, etc)
│   ├── q16.zig             Q16 struct + all arithmetic (add/sub/mul/div/compare/
│   │                       softmax/dotProduct/fromFraction/zero/one/negate/abs)
│   ├── q32.zig             Q32 struct + arithmetic (i64 value, two i32 remainders)
│   ├── q335.zig            Q335 struct + limb arithmetic (6×i64, four remainder levels)
│   └── reproject.zig       Q-basis conversion (Q16↔Q32↔Q335) with exact remainders
│
├── kb/                     Knowledge base storage layer
│   ├── types.zig           VlpFact (40 bytes), VlpKB (256 bytes), VlpProvenance (28 bytes),
│   │                       KBCreateConfig, KBStoreConfig
│   ├── store.zig           KBStore: contiguous KB/fact/text arrays, createKB, getKB, count
│   ├── fact.zig            factAssert/Query/Retract/Search/ScopedSearch — O(1) by two
│   │                       integer indices, scoped search walks parent chain
│   ├── tree.zig            addChild/removeChild/getParent/getChildren/ancestorWalk
│   ├── path_index.zig      Open-addressing hash map: dotted path string → kb_id
│   ├── text_store.zig      Append-only byte array, referenced by offset+length
│   └── visibility.zig      checkAccess (ancestor walk with integer visibility comparison),
│                            resolveVisibleKBs
│
├── safety/                 Structural access control
│   ├── grant.zig           GrantStore: create/check/revoke/list/cleanup, four integer
│   │                       comparisons per check, monotonic state transitions
│   └── audit.zig           AuditRing: append-only ring buffer, 28-byte entries,
│                            query with filter
│
├── confidence/             Exact confidence propagation
│   └── propagate.zig       CONFIDENCE_TABLE (11 Q16 values), assignFromSource,
│                            combineAgreeing (1-∏(1-Ci)), combineConflicting, chain (C^N),
│                            propagate (walk derivation tree)
│
├── prolog/                 Prolog deduction engine
│   ├── types.zig           VlpTerm (24-byte tagged union), VlpRule (44 bytes),
│   │                       VlpBinding, BindingSet, QueryConfig
│   ├── term.zig            Term constructors: makeAtom/Var/Int/Vdr/Text/List/Compound
│   ├── unify.zig           Recursive unification with depth limit 100, cross-multiply
│   │                       comparison for VDR values, structural match for compounds/lists
│   ├── query.zig           Depth-first search with backtracking over KB facts
│   ├── rule.zig            RuleStore: assertRule/retractRule/fireAll/fireAndCommit/
│   │                       getRuleStats — fireAll returns candidates without committing
│   └── hygiene.zig         hygieneScan: identify stale (>90d), failing (<20% success),
│                            orphaned (revoked grant) rules
│
├── grammar/                Grammar-directed structural output
│   ├── compile.zig         Parse template with {name:type} slots, validate, extract slot table
│   ├── render.zig          Walk template copying literals, render fills at slots —
│   │                       every structural byte from template, zero LLM forward passes
│   ├── validate.zig        Template structural validation
│   └── inherit.zig         Walk KB tree upward for grammar at requested slot
│
├── primitives/             Bounded data structures (fixed capacity, never grow)
│   ├── lru.zig             LRU cache: doubly-linked list + hash map, evict oldest at capacity
│   ├── counter.zig         Saturating i32 counter with min/max, clamp not wrap
│   ├── lock.zig            Non-blocking boolean coordination signal
│   ├── queue.zig           Bounded FIFO, push returns false when full
│   ├── stack.zig           Bounded LIFO, push returns false when full
│   ├── ring.zig            Fixed-size sliding window, write always succeeds (overwrites oldest)
│   └── bitset.zig          Packed bit array, set/clear/get/popcount
│
├── session/                Session lifecycle with COW
│   ├── lifecycle.zig       create/destroy/clone/merge/kill
│   ├── cow.zig             COWPageTable: read checks dirty bit, write triggers copy-on-write
│   └── snapshot.zig        Binary blob: header(magic+version+CRC32) + contiguous regions,
│                            save/restore/diff, checksum mismatch = hard fail
│
├── engine/                 Inference engine core
│   ├── context.zig         Five-segment context builder (~300-600 tokens, constant size)
│   ├── scratchpad.zig      Ring buffer for command results and Phase 0 output
│   ├── token_classify.zig  COMMAND_START / DIRECT_OUTPUT / END_OF_TURN / PROSE
│   ├── level_stats.zig     L1/L2/L3 counters, getAutoTriageRate as exact fraction
│   ├── command_parse.zig   Token → VlpCommand: match type enum, resolve path, parse args
│   ├── command_exec.zig    Access check → grant check → dispatch to 15 targets
│   └── cycle.zig           vlp_cycle: Phase 0 (fire rules) → Phase 1 (context) →
│                            Phase 2 (generate + dispatch) → Phase 3 (counters + snapshot)
│
├── llm/                    LLM inference (all Q16, no float)
│   ├── model.zig           ModelConfig, Model struct, weight loading, parameter counting
│   ├── forward.zig         Full transformer forward pass: embed → layernorm → QKV →
│   │                       attention → residual → MLP → final norm → logits
│   ├── softmax.zig         Quadratic surrogate: shift-square-divide, sum=D exactly
│   ├── attention.zig       Multi-head attention with causal mask, verifySoftmaxSum
│   ├── kv_cache.zig        KV cache as KB facts at computed slot IDs — survives
│   │                       snapshot/restore, shares via COW
│   ├── generate.zig        prefill, generateToken, generateCommand (constrained vocab),
│   │                       generateProse (unconstrained)
│   └── sampling.zig        Greedy/Top-K/Top-P/Temperature, deterministic integer RNG
│
├── builtins/               448-target builtin library (180+ registered)
│   ├── dispatch.zig        BuiltinTable: 512 slots, fn ptr dispatch, register/lookup
│   ├── text.zig            17 functions: reverse, split, contains, replace, join, trim,
│   │                       upper, lower, startsWith, endsWith, indexOf, substring,
│   │                       repeat, padLeft, padRight, charAt, length
│   ├── arithmetic.zig      25 functions: add through distance, all exact Q16
│   ├── collections.zig     36 functions: sort, filter, map, reduce, groupBy, scan,
│   │                       findFirst, binarySearch, merge, deduplicate, window, etc
│   ├── sets.zig            14 functions: union, intersection, difference, subset checks,
│   │                       all operating on sorted Q16 arrays with exact comparison
│   ├── mappings.zig        15 functions: VlpMap (open-addressing hash table),
│   │                       get/set/delete/merge/filter/invert
│   ├── conversion.zig      14 functions: parseJson/Csv/Xml/Yaml (write to KB, zero tokens),
│   │                       toJson/toCsv (export from KB), number/string conversion,
│   │                       baseConvert, timestampToFields
│   ├── linalg.zig          8 functions: matVecMul, transpose, gaussianElim, inverse,
│   │                       determinant, gramSchmidt, eigenvalues, svd
│   ├── stats.zig           8 functions: mean, variance, median, bayes (posterior sums to
│   │                       D exactly), normalize (sum=D exactly), histogram, correlation,
│   │                       covariance — all exact fractions
│   ├── graph.zig           13 functions: Graph struct with Edge{from,to,weight:Q16},
│   │                       bfs, dfs, shortestPath (Dijkstra with exact weights),
│   │                       topoSort, components, cycleDetect, pageRankExact,
│   │                       markovSteady — all converge to exact steady state
│   ├── integer_ops.zig     21 functions: wrapping arithmetic, factorial, choose,
│   │                       bitwise and/or/xor/not/shift/popcount/reverse
│   ├── time_ops.zig        10 functions: timestamps, durations, field extraction
│   └── register_*.zig      Registration functions mapping IDs to function pointers
│
├── seed/                   Initial KB tree (~300 entries loaded at startup)
│   ├── seed_init.zig       Create root KB tree, call all loaders, return SeedIds
│   ├── oso_rules.zig       15 engineering principles as text facts
│   ├── confidence_table.zig 11 Q16 confidence values (knowability spectrum)
│   ├── command_vocab.zig   ~300 command names for constrained generation
│   ├── hygiene_rules.zig   3 self-maintenance rule definitions (stale/failing/orphan)
│   ├── sentence_templates.zig  12 SRE domain grammar templates
│   ├── format_grammars.zig 18 structural format templates (JSON, HTTP, SMTP, etc)
│   └── builtin_declarations.zig  36 representative IOSE declarations
│
├── test_scenarios/         Integration validation
│   ├── sre_scenario.zig    End-to-end SRE scenario: create KB tree, assert Prometheus
│   │                       facts, test confidence propagation, compile and render
│   │                       grammar, verify L1/L2/L3 level stats progression
│   └── determinism_tests.zig  100× memcmp verification across 9 categories:
│                              arithmetic, softmax, collections, sets, linalg, stats,
│                              graph, KB facts, confidence
│
├── runner/                 Autonomous execution loops
│   ├── types.zig           VlpRunner (72 bytes), configs, enums, RunnerStatus
│   ├── pool.zig            ThreadPool (32 threads max), TaskQueue (256 capacity),
│   │                       RunnerTable (64 slots), worker loop
│   ├── poller.zig          Timer loop: fire Prolog rules → L3 path → output to KB
│   ├── processor.zig       Persistent connection, rule-based compaction, recycle at
│   │                       turn threshold (snapshot→kill→clone→restore), exponential
│   │                       backoff reconnect (1s→60s cap)
│   ├── internal.zig        Timer loop calling compute function, zero LLM tokens
│   ├── batch.zig           Clone-per-task from KB queue, merge results, kill clone,
│   │                       max 16 concurrent
│   ├── runner_manager.zig  Facade: create/start/stop/kill/recycle/destroy/getStatus
│   ├── runner_ops.zig      Convenience wrappers for processor/internal/batch creation
│   └── sre_deployment.zig  4-runner SRE deployment (prometheus processor, deploy
│                            processor, triage poller, hygiene internal)
│
├── server/                 Multi-protocol server
│   ├── types.zig           Server struct (256 connections), ServerConfig, ServerCredential,
│   │                       ServerConnection (~16KB each), protocol/state/close enums
│   ├── listener.zig        TCP socket via std.posix, accept loop with capacity check,
│   │                       connection slot allocation, reject with protocol response
│   ├── auth.zig            FNV-1a token hash, auth KB layout (4 slots per user),
│   │                       authenticate/credentialCheck/credentialRevoke,
│   │                       registerUser/suspendUser/reactivateUser
│   ├── handler.zig         HTTP request parser (method/path/headers/body), request
│   │                       routing (/health /metrics /kb/* /query), response builder,
│   │                       keepalive support
│   ├── rate_limit.zig      Per-user counter in KB, window-based reset, exact integer
│   │                       threshold — zero drift, zero false crossings
│   ├── health.zig          HealthReport from integer counters, renderHealthJson
│   │                       (grammar-rendered, zero LLM tokens)
│   ├── reaper.zig          Periodic scan: idle timeout, credential expiry, turn budget
│   │                       — three integer comparisons per connection
│   ├── shutdown.zig        Graceful: shutdown flag → close listener → drain connections
│   │                       → timeout → force-close with optional snapshot
│   └── server_main.zig     ServerRuntime: start (listen + accept thread + reaper thread),
│                            stop (graceful shutdown + join), health/metrics queries
│
├── protocol/               Wire protocol handlers
│   ├── http.zig            Full HTTP/1.1 request parser (32 headers, WebSocket upgrade
│   │                       detection), response builder, error responses
│   ├── websocket.zig       Upgrade handshake, frame read (unmask) / write (7/16/64-bit
│   │                       length), message loop with credential expiry (close 4001),
│   │                       ping/pong
│   ├── grammars.zig        10 protocol grammar templates stored in KB, HTTP/JSON/SMTP/
│   │                       MQTT rendering helpers
│   ├── protocol_router.zig Route connections by protocol type
│   ├── smtp.zig            SMTP state machine stub (EHLO→MAIL→RCPT→DATA→response)
│   └── mqtt.zig            MQTT stub (CONNECT/CONNACK/PUBLISH/PINGREQ/DISCONNECT)
│
├── ops/                    Grant-gated operational primitives
│   ├── filesystem.zig      fsRead/Write/Append/Delete/Stat via std.fs, fsReadToKB
│   ├── network.zig         netFetch (stub), netFetchToKB
│   ├── execute.zig         execRun (stub), execRunToKB
│   ├── compile_check.zig   Balanced delimiter verification
│   ├── process.zig         procStart/Kill/Status (stubs)
│   └── ops_dispatch.zig    Builtin wrappers + registerOpsBuiltins (IDs 500-510, pure=false)
│
├── config/                 System configuration
│   ├── system_config.zig   SystemConfig struct with production defaults
│   ├── cli.zig             Command-line argument parser, --help, --version, --test
│   ├── config_file.zig     Key=value config file parser
│   └── integration_test.zig 13-check integration test exercising complete stack
│
├── gpu/                    GPU compute layer (CPU fallback)
│   ├── device.zig          Device enumeration, properties (all integer), global state
│   ├── memory.zig          DeviceMemoryLayout (13 regions, 256-byte aligned),
│   │                       computeLayout, DeviceAllocation backed by host memory
│   ├── transfer.zig        hostToDevice/deviceToHost/deviceToDevice, Q16 typed transfer,
│   │                       mirrorKBStore, deviceFactWrite/Read by computed offset
│   ├── profiling.zig       KernelStats (integer metrics only), SessionStats, Profiler
│   ├── benchmarks.zig      8 benchmarks (forward, softmax, attention, sort, layernorm,
│   │                       prolog, elementwise, gemm), nanoTimestamp measurement
│   ├── determinism.zig     Run kernel N times, byte-compare all outputs,
│   │                       all_identical must be true (false = bug, not drift)
│   └── kernels/
│       ├── gemm.zig        q16Gemm/Batched/StridedBatched/MatVecMul — widening i64 MAC
│       ├── softmax.zig     q16Softmax (quadratic surrogate, sum=D), batched, verify
│       ├── elementwise.zig q16Add/Sub/Mul/Div/Scale/Dot/Compare/Negate/Abs/Min/Max/
│       │                   Clamp/Fill/Copy/Sum/RemainderMagnitude
│       ├── normalize.zig   q16LayerNorm (exact mean+variance+intSqrt), q16RMSNorm
│       ├── activation.zig  q16ReLU, q16GELU (linear approx), q16SiLU (linear approx)
│       ├── attention.zig   Fused QK^T→softmax→AV, verifySoftmaxSumAllHeads,
│       │                   fusedAttentionWithKVCache (single query against cache)
│       ├── sort.zig        Merge sort, argSort (descending), topK
│       ├── prolog_kernel.zig  batchUnify, batchCrossMultiplyCompare, scopeFilter
│       │                      (visibility for all KBs), parallelRuleEval (N×M fire matrix)
│       └── reduction.zig   q16ReduceSum/Max/Min/ArgMax/ArgMin, allReduce stubs
│
├── deploy/                 Deployment and verification
│   ├── distributed.zig     Comm struct, allReduceSum/Max/Min (single-rank passthrough),
│   │                       broadcast, allGather, reduceScatter, kbSync, snapshotBroadcast
│   ├── model_parallel.zig  Layer sharding across devices, pipelineForward (stub)
│   ├── load_balancer.zig   Round-robin / least-connections, 16 backends,
│   │                       addBackend/route/release/markUnhealthy/removeUnhealthy
│   ├── prometheus_export.zig  Render metrics as "metric_name value\n" format
│   ├── chaos.zig           4 chaos tests: snapshot recovery (corrupt→restore→verify),
│   │                       kill-restart (10 facts survive), concurrent write (100 facts
│   │                       exact), determinism after restart (100× suite)
│   └── deploy_main.zig     deployAndVerify: integration → chaos → benchmarks →
│                            determinism → report pass/fail
│
└── main.zig                Entry point: parse CLI → load config → run tests or start server
```

## Build and Run

```bash
# Build
zig build

# Run integration tests
zig build run -- --test

# Start server
zig build run -- --port 8080 --max-connections 64

# Run with config file
zig build run -- --config tensorprolog.conf

# Show help
zig build run -- --help
```

Configuration file format (key=value):
```
port = 8080
max_connections = 64
max_kbs = 100000
max_facts = 10000000
credential_ttl = 3600
idle_timeout = 300
rate_limit_window = 60
rate_limit_max = 100
layers = 1
d_model = 64
n_heads = 1
vocab_size = 256
```

## Memory Model

All memory is pre-allocated at startup. No heap allocation in hot paths. Fixed arena, sized once, crashes on exceed.

```
Model weights:  n_params × 8 bytes (Q16)
KB store:       max_kbs × 256 bytes
Fact store:     max_facts × 40 bytes
Text store:     configured bytes (default 100MB)
Live state:     max_sessions × 50KB typical
Scratch:        n_streams × configured bytes
Audit ring:     capacity × 28 bytes

Reference (7B model, 10K sessions):
  Model: 56 GB (requires multi-GPU)
  Infrastructure: ~2.2 GB (negligible relative to model)
```

Every data structure declares its capacity at creation. LRU caches, counters, queues, stacks, ring buffers, bitsets — all bounded. Push to a full queue returns false. Counter at max clamps. No structure can grow past its declared capacity.

## Invariants

These hold at all times, in all states. Violation is a system bug.

1. **Softmax sum = D exactly.** Every softmax call, every row, integer equality. Not approximately.
2. **KB facts at integer addresses are exact.** Read at turn 1 or turn 1,000,000 returns what was asserted.
3. **Bounded primitives cannot exceed declared capacity.** No exceptions, no overflow, no wrap.
4. **Snapshot restore is bit-identical.** Save → modify → restore → memcmp matches original.
5. **Clone COW is invisible to parent.** Clone writes never modify parent state.
6. **Access-denied data is absent.** Query returns zero results, not redacted results.
7. **Grant denial happens before execution.** No partial side effects from denied operations.
8. **Integer arithmetic is deterministic across devices.** Same inputs, same result, every time.
9. **Prolog unification uses exact comparison.** No tolerance, no epsilon.
10. **Audit log is append-only and complete.** Every operation produces an entry.

## What's Not Here Yet

- Training loop (Adam, backward pass, compute graph)
- Actual GPU kernel dispatch (CPU fallback only — kernels are written, dispatch is not)
- Multi-device NVLink transfer (distributed layer is single-rank passthrough)
- FRU software recurrence for transcendentals
- Docker sandbox integration for exec
- Real HTTP client for net_fetch
- SHA-1 + base64 for WebSocket accept key
- Full eigenvalue decomposition (QR iteration) and SVD (Golub-Kahan)
- Registration of collections/sets builtins in dispatch table

## Builtin ID Map

| Range | Category | Count | Turn |
|-------|----------|-------|------|
| 0-24 | Arithmetic | 25 | 21 |
| 100-116 | Text | 17 | 21 |
| 200-214 | Mappings | 15 | 23 |
| 300-311 | Conversion | 12 | 23 |
| 400-407 | Linear Algebra | 8 | 24 |
| 420-427 | Statistics | 8 | 24 |
| 440-452 | Graph | 13 | 25 |
| 460-480 | Integer/Bit Ops | 21 | 25 |
| 490-499 | Time | 10 | 25 |
| 500-510 | Ops (grant-gated) | 11 | 34 |

Collections (36) and Sets (14) are implemented as library functions in turns 21-22 but not individually registered in the dispatch table.

## Testing

The integration test (`--test` flag) runs 13 checks:

1. Seed initialization — KB tree created with correct structure
2. KB operations — create parent and child KBs
3. Fact roundtrip — assert Q16 value, read back, verify exact match
4. Prolog engine — fire rules without error
5. Grammar engine — compile template, verify validated
6. Confidence propagation — combine two sources, verify > individual
7. Authentication — register user, authenticate, verify credential
8. Rate limiting — 5 allowed, 6th denied (exact threshold)
9. Health check — verify zero counters on fresh server
10. Runner creation — allocate poller, verify non-null ID
11. SRE scenario — end-to-end KB/fact/confidence/grammar/level-stats
12. Determinism — 100× memcmp across 9 operation categories
13. Builtin dispatch — verify table populated and entries registered

Chaos tests (run via deploy verification):
- Snapshot recovery: assert → snapshot → corrupt → restore → verify original
- Kill-restart: 10 facts → snapshot → restore to new KB → verify all match
- Concurrent write: 100 sequential facts → read all → verify exact
- Determinism after restart: full 100× suite passes

Determinism verification runs every kernel 100 times and byte-compares all outputs. Any difference is a FAIL — not noise, not drift, a bug. This test is meaningful only because integer arithmetic is deterministic.

## License

MIT
