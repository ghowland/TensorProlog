# TensorProlog

**Exact integer arithmetic GPU compute stack for LLM inference, training, and autonomous operation.**

TensorProlog replaces floating-point GPU computing with VDR (Value/Denominator/Remainder) integer triples. Every computation is exact. Every result is deterministic. Every intermediate value is inspectable. Softmax outputs sum to D=65536 by integer equality, not approximation. Checkpoints are bit-identical across platforms. Training on one machine produces the same model as training on another. Always.

The system combines an LLM inference engine with a Prolog logic engine, a knowledge base store, a grammar template engine, and autonomous runner management — all operating on exact integer arithmetic. The LLM handles judgment and framing. Everything else (data retrieval, computation, deduction, formatting, access control) is handled by deterministic integer subsystems at zero LLM token cost.

---

## What This Is

A complete GPU compute stack implementing:

- **VDR arithmetic**: Exact rational numbers as integer triples `{value: i32, remainder: i16, _pad: i16}` with implicit denominator 65536. No floats. No rounding. No platform-dependent results.
- **LLM inference**: Transformer forward pass in exact integer arithmetic. Attention weights sum to D exactly. Softmax via quadratic surrogate (shift-square-divide) or FRU-based exact exponential.
- **Knowledge base store**: 26-field KB structs at integer addresses. Facts are 40-byte structs indexed by kb_id + slot_id. O(1) access. Scoped visibility via integer comparison.
- **Prolog engine**: Depth-first search with backtracking over KB facts. Cross-multiply comparison for exact rational unification. Rules fire automatically at zero LLM cost.
- **Grammar engine**: Typed slot templates that produce structurally correct output by construction. Every bracket, pipe, header comes from the template. LLM fills content slots only.
- **Session management**: Snapshot/clone/kill lifecycle. Copy-on-write for clones. Bit-identical restore. Disposable clones with drift thresholds.
- **Runner system**: Four autonomous loop types (poller, processor, internal, batch) that operate continuously without human intervention.
- **Structural safety**: Access control via integer comparison before data access. Grant-gated operations with monotonic state transitions. Append-only audit trail.
- **Confidence propagation**: Exact VDR fractions from a declared knowability hierarchy. No hedging language — computed confidence with full derivation chain.

## What This Is Not

- Not a float GPU library. No float arithmetic anywhere in the computation path.
- Not a wrapper around CUDA/cuBLAS/cuDNN. Replaces them entirely with integer-native operations.
- Not an approximate or quantized system. Values are exact rational numbers, not truncated floats.
- Not production-scale yet. Current implementation targets research and validation. GPU kernels run on CPU fallback.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HOST (CPU)                               │
│                                                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Session   │  │ Runner    │  │ Grant    │  │ Snapshot  │  │
│  │ Manager   │  │ Scheduler │  │ Enforcer │  │ Manager   │  │
│  └─────┬─────┘  └─────┬─────┘  └─────┬────┘  └─────┬─────┘  │
│        │              │              │              │         │
│  ┌─────┴──────────────┴──────────────┴──────────────┴─────┐  │
│  │              Orchestration Layer                        │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌────────────────────────┴───────────────────────────────┐  │
│  │              Host-Device Bridge                         │  │
│  └────────────────────────┬───────────────────────────────┘  │
└───────────────────────────┼──────────────────────────────────┘
                            │
┌───────────────────────────┼──────────────────────────────────┐
│                     DEVICE (GPU / CPU fallback)              │
│                                                              │
│  ┌───┴──┐ ┌──────┐ ┌──────┐ ┌────────┐ ┌──────────┐        │
│  │ LLM  │ │ KB   │ │Prolog│ │Grammar │ │ Builtin  │        │
│  │Engine│ │Store │ │Engine│ │Engine  │ │ Executor │        │
│  └──────┘ └──────┘ └──────┘ └────────┘ └──────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Integer Arithmetic Layer (Q16/Q32/Q335)      │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

Five device-side engines. One host-side orchestration layer. One bridge. All communication is typed integers through declared interfaces.

---

## The Universal Cycle

Everything in the system runs one function — `vlp_cycle` — with different input/output bindings:

```
Phase 0: Fire Prolog rules (L3 path, zero LLM tokens)
         → If rules fully handle the input, render via grammar and return
Phase 1: Build LLM context (~300-600 tokens, bounded, constant regardless of turn)
Phase 2: LLM generates tokens, classified as:
         → COMMAND: parse + access check + grant check + execute + scratchpad
         → DIRECT_OUTPUT: load KB data + grammar render → output
         → PROSE: LLM judgment text → output
         → END_OF_TURN: break
Phase 3: Update counters, auto-snapshot check, recycle check
```

Interactive chat, polling runners, processor runners, batch runners, HTTP servers, WebSocket servers — all instantiate this same cycle with different triggers and output routing.

---

## Core Types

```zig
// The atomic unit of arithmetic — 8 bytes, packs in registers and cache lines
const Q16 = struct {
    v: i32,       // numerator
    r0: i16,      // remainder
    _pad: i16,    // alignment
    // Implicit denominator D = 65536. Never stored. Universal.
};

// The atomic unit of knowledge — 40 bytes
const VlpFact = struct {
    tag: VlpFactTag,           // what kind of value (12 types)
    value: Q16,                // the value
    provenance: VlpProvenance, // where it came from (28 bytes)
};

// The organizational unit — 256 bytes, cache-aligned
const VlpKB = struct {
    // 26 fields: identity (3), persistent (6), live (8),
    // structural (5), metadata (4), padding to 256 bytes
};
```

---

## Directory Structure

```
src/
├── vdr/                          # VDR arithmetic foundation (Turns 1-2)
│   ├── types.zig                 #   All shared enums: VlpStatus, VlpQBasis, VlpFactTag,
│   │                             #   VlpSourceType, VlpVisibility, VlpGrantClass, VlpGrantState,
│   │                             #   VlpSessionState, VlpTermType, VlpSlotType, VlpCommandType,
│   │                             #   VlpAuditAction, VlpRunnerType, VlpRunnerState, VlpTokenClass,
│   │                             #   VlpExecutionLevel, VlpMergePolicy, VlpProtocolType,
│   │                             #   VlpConnectionState, VlpReduceOp — 21 enums total
│   ├── q16.zig                   #   Q16 struct and all operations: add, sub, mul (widening i64
│   │                             #   product >> 16, remainder = product & 0xFFFF), div, compare,
│   │                             #   eql, fromFraction, toFraction, zero, one, negate, abs, sign,
│   │                             #   min, max, remainderMagnitude, compact, softmax (quadratic
│   │                             #   surrogate, sum = D = 65536 exactly), dotProduct
│   ├── q32.zig                   #   Q32: i64 value, two remainder levels, D = 2^32
│   ├── q335.zig                  #   Q335: 6×i64 limbs, four remainder levels, D = 2^335,
│   │                             #   240 bytes per value. Limb helpers: addLimbs, subLimbs,
│   │                             #   mulLimbs (schoolbook 6×6), shrLimbs, compareLimbs
│   └── reproject.zig             #   Q-basis conversion: q16ToQ32, q32ToQ16 (with remainder
│                                 #   capture), q16ToQ335, q335ToQ16, q32ToQ335, q335ToQ32
│
├── kb/                           # Knowledge base infrastructure (Turns 3-5)
│   ├── types.zig                 #   VlpProvenance (28 bytes), VlpFact (40 bytes), VlpKB (256
│   │                             #   bytes with 26 fields), KBCreateConfig, KBStoreConfig
│   ├── store.zig                 #   KBStore: contiguous array of VlpKB structs + fact array +
│   │                             #   text store + path index. init, createKB (returns sequential
│   │                             #   i32 ID), destroyKB, getKB, count, capacity, resolvePath
│   ├── fact.zig                  #   factAssert (checks frozen + bounds, writes at facts_offset +
│   │                             #   slot_id), factQuery (O(1) via two integer indices), factRetract
│   │                             #   (sets tag = empty), factSearch (linear scan by tag),
│   │                             #   factScopedSearch (walks parent chain — lexical scoping)
│   ├── tree.zig                  #   addChild, removeChild, getParent, getChildren, ancestorWalk
│   │                             #   (callback-based iteration up to root)
│   ├── path_index.zig            #   Open-addressing hash map: dotted path string → i32 kb_id.
│   │                             #   Linear probing, load factor < 0.7. insert, lookup, remove,
│   │                             #   computeHash
│   ├── text_store.zig            #   Append-only byte store: append (returns offset + length),
│   │                             #   read (returns slice). All KB names, paths, text values
│   │                             #   stored here, referenced by offset + length pairs
│   └── visibility.zig            #   checkAccess: walks parent chain checking visibility integer
│                                 #   at each level (PUBLIC/INTERNAL/OWNER_ONLY). Data absent,
│                                 #   not filtered. resolveVisibleKBs: enumerate visible KBs,
│                                 #   prune subtrees on failure
│
├── safety/                       # Access control and audit (Turns 5-6)
│   ├── types.zig                 #   VlpGrant (48 bytes: class, state, holder, target pattern,
│   │                             #   uses, expiry, creator), VlpAuditEntry (28 bytes),
│   │                             #   GrantCheckResult
│   ├── grant.zig                 #   GrantStore: create, check (4 integer comparisons: state ==
│   │                             #   ACTIVE AND not expired AND uses remaining AND target prefix
│   │                             #   match, then atomic decrement), revoke (permanent), list,
│   │                             #   cleanup (mark expired/exhausted)
│   └── audit.zig                 #   AuditRing: fixed-capacity ring buffer, write (append-only,
│                                 #   overwrites oldest), query (filter by user/action/time).
│                                 #   Every access check, grant check, fact assertion logged
│
├── confidence/                   # Provenance propagation (Turn 6)
│   ├── types.zig                 #   CONFIDENCE_TABLE: 11 Q16 constants mapping source types to
│   │                             #   exact fractions (VDR computation = 65536/65536, Prolog
│   │                             #   derivation = 65536/65536, database = 64225/65536, ...
│   │                             #   LLM generated = 19660/65536, unknown = 0/65536)
│   └── propagate.zig             #   assignFromSource (table lookup), combineAgreeing
│                                 #   (1 - ∏(1-Cᵢ) via integer complement multiply),
│                                 #   combineConflicting (with penalty), chain (C^N via
│                                 #   repeated exact multiply), propagate (walk derivation
│                                 #   chain recursively)
│
├── prolog/                       # Logic engine (Turns 7-9)
│   ├── types.zig                 #   VlpTerm (tagged union, 24 bytes: atom/variable/integer/
│   │                             #   vdr/text/list/compound), VlpBinding, BindingSet (with
│   │                             #   checkpoint/undo for backtracking), VlpRule (44 bytes),
│   │                             #   QueryConfig, QueryResults, PrologFired, PrologAction
│   │                             #   (assert_fact/retract_fact/direct_output), HygieneCandidate
│   ├── term.zig                  #   Constructors: makeAtom, makeVar, makeInt, makeVdr, makeList,
│   │                             #   makeCompound. termEql, containsVar (occurs check)
│   ├── unify.zig                 #   unify: recursive, depth-limited (100). Cases: atom-atom
│   │                             #   (integer equality), var-anything (occurs check then bind),
│   │                             #   VDR-VDR (a.v == b.v since same D), compound (functor match
│   │                             #   then recursive arg unification), list (head + tail recursive)
│   ├── query.zig                 #   Depth-first search with backtracking. Collects candidates
│   │                             #   via scopedSearch, attempts unification with each, recurses
│   │                             #   on body goals, collects successful binding sets
│   ├── rule.zig                  #   RuleStore: assertRule, retractRule, fireAll (evaluate all
│   │                             #   rules, return fired list without committing), fireAndCommit
│   │                             #   (fire + apply assert/retract actions), getRuleStats
│   └── hygiene.zig               #   hygieneScan: detect stale (>90 days unfired), failing
│                                 #   (<20% success after >5 fires), orphaned (references
│                                 #   revoked grant). Returns candidates, does not auto-delete
│
├── grammar/                      # Template engine (Turns 9-10)
│   ├── types.zig                 #   VlpGrammarSlot, VlpGrammar (28 bytes), VlpGrammarFill
│   │                             #   (tagged union for fill values), GrammarKBMapping
│   │                             #   (slot_index → kb_id + slot_id), RenderResult
│   ├── compile.zig               #   Parse template: scan for {name:type}, extract slots, build
│   │                             #   slot table, validate matching braces, type validity, name
│   │                             #   uniqueness. Set validated = true on success
│   ├── render.zig                #   render: walk template, memcpy literal ranges, render fills
│   │                             #   by type (VDR → decimal string, text → copy, integer →
│   │                             #   decimal, enum → validated copy). renderFromKB: fills from
│   │                             #   KB facts via slot → (kb_id, slot_id) mappings — data never
│   │                             #   enters token stream
│   ├── validate.zig              #   Structural validity check: if valid, every possible
│   │                             #   rendering is syntactically correct by construction
│   └── inherit.zig               #   Walk KB parent chain looking for grammar at slot.
│                                 #   First found wins — lexical scoping for grammars
│
├── primitives/                   # Bounded data structures (Turns 11-12)
│   ├── types.zig                 #   LRUEntry, CounterState — shared primitive types
│   ├── lru.zig                   #   Doubly-linked list + hash map. init(capacity 1-1000),
│   │                             #   get (promotes to MRU), put (evicts oldest at capacity),
│   │                             #   evictOldest, size, clear. Cannot exceed declared capacity
│   ├── counter.zig               #   init(min, max, initial). get, increment (clamps at bounds,
│   │                             #   never wraps), reset, atBound. Saturating arithmetic
│   ├── lock.zig                  #   Non-blocking flag. acquire → bool, release, query. Never
│   │                             #   blocks — coordination signal, not mutex
│   ├── queue.zig                 #   Circular buffer FIFO. push → bool (false if full), pop,
│   │                             #   peek, size, clear. Bounded at creation
│   ├── stack.zig                 #   Array + top index LIFO. push → bool (false if full), pop,
│   │                             #   peek, size, clear. Bounded at creation
│   ├── ring.zig                  #   Fixed-size sliding window. write (always succeeds,
│   │                             #   overwrites oldest), read(index), size, clear
│   └── bitset.zig                #   Bit array (1-10000 bits). set, clearBit, get → bool,
│                                 #   popcount → i32, clearAll
│
├── session/                      # Session lifecycle (Turns 13-14)
│   ├── types.zig                 #   VlpSession (~128 bytes: all counters, state, device refs,
│   │                             #   clone lineage, cached system prompt [2048]i32),
│   │                             #   SessionConfig, CloneConfig, VlpSnapshotHeader (magic
│   │                             #   "VLPS", version, region sizes, CRC32), VlpSnapshot,
│   │                             #   SnapshotDiff
│   ├── lifecycle.zig             #   sessionCreate, sessionDestroy, sessionClone (COW page
│   │                             #   table, shared persistent KBs, independent live state),
│   │                             #   sessionMerge (OURS/THEIRS/FAIL_ON_CONFLICT by timestamp
│   │                             #   comparison), sessionKill (immediate, no snapshot)
│   ├── cow.zig                   #   COWPageTable: read (source or private), writeBegin
│   │                             #   (copy-on-first-write), dirtyPages, resolve (copy all
│   │                             #   source to private for self-contained snapshot).
│   │                             #   Page size 4096 bytes (~100 facts)
│   └── snapshot.zig              #   snapshotSave (collect all state, pack contiguous, CRC32),
│                                 #   snapshotRestore (validate checksum — hard fail on
│                                 #   mismatch — overwrite all state), saveToFile, loadFromFile,
│                                 #   snapshotDiff (byte-compare regions), crc32
│
├── engine/                       # Universal cycle (Turns 15-17)
│   ├── context.zig               #   contextBuild: assemble 5 segments — system prompt (~200
│   │                             #   tokens, cached), scope reference (~5 tokens), scratchpad
│   │                             #   (0-50 tokens), turn summary (0-100 tokens), user input.
│   │                             #   Total bounded ~300-600 tokens regardless of turn number
│   ├── scratchpad.zig            #   Per-session ring buffer: writeResult, writeError,
│   │                             #   writeDenied, writeGrantDenied, clear, getContents
│   ├── token_classify.zig        #   classify: token_id → COMMAND_START / DIRECT_OUTPUT /
│   │                             #   END_OF_TURN / PROSE. Command start = token in command
│   │                             #   vocabulary range. Direct output = kb:// prefix token
│   ├── level_stats.zig           #   L1/L2/L3 counters. update(level, tokens).
│   │                             #   getAutoTriageRate → exact fraction (num/den)
│   ├── command_parse.zig         #   VlpCommand struct. commandParse: match first token →
│   │                             #   command type enum (~15 options), resolve dotted path via
│   │                             #   PathIndex, parse typed args
│   ├── command_exec.zig          #   commandExecute: access check → grant check (if
│   │                             #   operational) → dispatch by type (KB_ASSERT, KB_QUERY,
│   │                             #   PROLOG_QUERY, BUILTIN_CALL, GRAMMAR_RENDER, DIRECT_OUTPUT,
│   │                             #   OP_FILESYSTEM, etc.) → audit write
│   ├── auto_resolve.zig          #   checkAutoResolution: did fired rules fully handle input?
│   │                             #   Checks: has grammar ref AND confidence ≥ threshold (default
│   │                             #   90/100 = 58982 at Q16) AND has DIRECT_OUTPUT action
│   └── cycle.zig                 #   vlpCycle: THE function. Phase 0 (fire rules) → Phase 1
│                                 #   (context build) → Phase 2 (generate + dispatch loop) →
│                                 #   Phase 3 (post-cycle). Returns CycleResult with status,
│                                 #   tokens consumed, commands executed, level, recycle flag
│
├── llm/                          # LLM inference (Turns 18-20)
│   ├── model.zig                 #   Model struct: weight arrays per layer (ln1_gamma/beta,
│   │                             #   qkv_weights, out_proj, ln2_gamma/beta, mlp_up/down),
│   │                             #   embedding, lm_head. ModelConfig (n_layers, d_model,
│   │                             #   n_heads, d_head, vocab_size, mlp_dim). LLMEngine bundles
│   │                             #   model + KV cache + sampling config + scratchpad
│   ├── forward.zig               #   forward: embedding lookup → per-layer (layerNorm → QKV
│   │                             #   projection GEMM → attention → out_proj → residual → MLP
│   │                             #   norm → up-project → activation → down-project → residual)
│   │                             #   → final norm → lm_head projection. layerNorm: exact mean
│   │                             #   (sum/n) and exact variance via Q16
│   ├── softmax.zig               #   softmaxSurrogate: shift so min=0, square each, sum
│   │                             #   squares, divide each by sum. Last element absorbs rounding.
│   │                             #   Sum = D = 65536. Every call. Guaranteed by construction
│   ├── attention.zig             #   Per head: QKᵀ dot products → scale by d_head reciprocal →
│   │                             #   causal mask (col > row → zero, integer comparison) →
│   │                             #   softmax per row (sum = D) → weighted sum of V.
│   │                             #   verifySoftmaxSum counts rows where sum ≠ D (expect 0)
│   ├── kv_cache.zig              #   KV cache stored as KB facts. Slot formula: layer *
│   │                             #   max_seq * heads * 2 + pos * heads * 2 + head * 2 +
│   │                             #   kv_offset. append/loadRange/truncate. Snapshot includes
│   │                             #   cache, clone shares via COW, kill destroys cleanly
│   ├── generate.zig              #   GenerateState bundles model + cache + config. prefill
│   │                             #   (forward each token). generateToken (softmax + sample +
│   │                             #   forward). generateCommand (mask vocab to ~300 names,
│   │                             #   renormalize to sum=D, greedy sample until end marker).
│   │                             #   generateProse (unconstrained full vocabulary)
│   └── sampling.zig              #   SamplingConfig (temperature Q16, top_k, top_p Q16, greedy
│                                 #   flag). sampleGreedy (argmax by integer comparison).
│                                 #   sampleTopK (partial sort, normalize, cumulative scan).
│                                 #   sampleTopP (sort descending, accumulate to threshold).
│                                 #   applyTemperature (divTrunc by temp). RNG: LCG on i64
│
├── builtins/                     # 448 deterministic primitives (Turns 21-25)
│   ├── dispatch.zig              #   BuiltinTable: 512-entry function pointer array. register
│   │                             #   (id, name, fn_ptr, pure, input_count, output_type).
│   │                             #   dispatch (id, args): lookup + validate + call.
│   │                             #   registerAllBuiltins calls all category registrations
│   ├── text.zig                  #   17 functions: reverse, split, contains, replace, join,
│   │                             #   trim, upper, lower, startsWith, endsWith, indexOf,
│   │                             #   substring, repeat, padLeft, padRight, charAt, length.
│   │                             #   IDs 100-116
│   ├── arithmetic.zig            #   25 functions: add, sub, mul, div, pow (binary
│   │                             #   exponentiation), reciprocal, compare, equal, min, max,
│   │                             #   sign, isZero, floor, ceil, round, numerator, denominator,
│   │                             #   abs, negate, clamp, fromInt, toInt, lerp, midpoint,
│   │                             #   distance. IDs 0-24
│   ├── collections.zig           #   36 functions: sort (merge sort), sortBy, filter, map
│   │                             #   (UnaryOp enum), reduce (BinaryOp enum), groupBy,
│   │                             #   frequencies, distinct, flatten, chunk, zip, unzip, reverse,
│   │                             #   rotate, takeFirst/Last, dropFirst/Last, partition,
│   │                             #   interleave, enumerate, minBy, maxBy, scan (prefix scan,
│   │                             #   exact accumulation), all, any, none, count, findFirst/Last/
│   │                             #   All, binarySearch, merge, deduplicate, window,
│   │                             #   cartesianProduct
│   ├── sets.zig                  #   14 functions on sorted Q16 arrays: union, intersection,
│   │                             #   difference, symmetricDiff (two-pointer merge), isSubset,
│   │                             #   isSuperset, isDisjoint, contains (binary search), add
│   │                             #   (sorted insert), remove, equal, powerSet (max 20 elements),
│   │                             #   fromArray (sort + dedup)
│   ├── mappings.zig              #   15 functions: open-addressing hash table with linear
│   │                             #   probing. get, set, delete, containsKey, keys, values, size,
│   │                             #   merge (with MergePolicy), filterKeys, filterValues,
│   │                             #   mapValues, invert, clear, equal, fromArrays. IDs 200-214
│   ├── conversion.zig            #   14 functions: parseJson (recursive descent → KB facts),
│   │                             #   parseCsv, parseXml (stub), parseYaml (stub), toJson (KB →
│   │                             #   JSON string), toCsv, toFraction, fromFraction,
│   │                             #   formatNumber, parseNumber, vdrToDecimalString (integer
│   │                             #   division loop), decimalStringToVdr, baseConvert (2-36),
│   │                             #   timestampToFields (unix → y/m/d/h/m/s). IDs 300-311
│   ├── linalg.zig                #   8 functions: matVecMul (widening MAC), transpose,
│   │                             #   gaussianElim (fraction-free elimination), inverse
│   │                             #   (Gauss-Jordan with partial pivoting), determinant,
│   │                             #   gramSchmidt (exact orthogonalization), eigenvalues
│   │                             #   (2×2 quadratic, n>2 diagonal approx), svd (stub).
│   │                             #   IDs 400-407
│   ├── stats.zig                 #   9 functions: mean (exact sum/n), variance, median
│   │                             #   (selection sort + middle), bayes (prior × likelihood /
│   │                             #   evidence, renormalize to sum=D), normalize (largest
│   │                             #   absorbs rounding), histogram (exact bin via integer
│   │                             #   comparison), correlation (exact Pearson), covariance.
│   │                             #   IDs 420-427
│   ├── graph.zig                 #   13 functions: Graph struct (nodes/edges slices), addNode,
│   │                             #   removeNode, addEdge, removeEdge, bfs (queue-based), dfs
│   │                             #   (stack-based), shortestPath (Dijkstra with exact Q16
│   │                             #   weights), topologicalSort (Kahn's), connectedComponents,
│   │                             #   cycleDetect (DFS coloring), pageRankExact (iterative
│   │                             #   power method, damping 55705/65536, renormalize to sum=D),
│   │                             #   markovSteady. IDs 440-452
│   ├── integer_ops.zig           #   21 functions: intAdd/Sub/Mul (wrapping), intDiv/Mod,
│   │                             #   intAbs, intSign, intMin, intMax, intClamp, intPow,
│   │                             #   intFactorial, intChoose, bitAnd/Or/Xor/Not,
│   │                             #   bitShiftLeft/Right, bitPopcount (@popCount), bitReverse
│   │                             #   (@bitReverse). IDs 460-480
│   ├── time_ops.zig              #   10 functions: timestampNow, timestampDiff, timestampAdd,
│   │                             #   durationSeconds/Minutes/Hours/Days, durationCompare,
│   │                             #   durationFormat, timestampFields. IDs 490-499
│   └── register_*.zig            #   Registration files connecting builtins to dispatch table
│                                 #   at declared IDs. registerMappingBuiltins,
│                                 #   registerConversionBuiltins, registerLinalgBuiltins,
│                                 #   registerStatsBuiltins, registerGraphBuiltins,
│                                 #   registerIntegerOpsBuiltins, registerTimeBuiltins,
│                                 #   registerOpsBuiltins
│
├── seed/                         # Initial KB population (Turn 26)
│   ├── seed_init.zig             #   seedInit: creates KB tree (root → system → {oso,
│   │                             #   confidence, builtins, command_vocab, hygiene},
│   │                             #   root → templates → {sentences, formats}), calls all
│   │                             #   load functions, freezes seed KBs. Returns SeedIds struct
│   ├── oso_rules.zig             #   15 engineering principles as text facts (P01-P15)
│   ├── confidence_table.zig      #   11 Q16 values matching knowability spectrum
│   ├── command_vocab.zig         #   ~300 command names across all categories
│   ├── hygiene_rules.zig         #   3 self-maintenance rules (stale >90 days, failing <20%
│   │                             #   success, orphan references revoked grant)
│   ├── sentence_templates.zig    #   12 grammar templates for SRE domain
│   ├── format_grammars.zig       #   18 format templates (JSON, CSV, table, HTTP, WebSocket,
│   │                             #   SMTP, MQTT, health, metrics)
│   └── builtin_declarations.zig  #   36 representative IOSE declarations as facts
│
├── test_scenarios/               # Integration and determinism tests (Turn 27)
│   ├── sre_scenario.zig          #   Scripted SRE investigation replay: create incident KB,
│   │                             #   assert Prometheus facts, test confidence combination,
│   │                             #   compile grammar, render findings, verify level stats
│   └── determinism_tests.zig     #   100 runs each of Q16 arithmetic, softmax, collections,
│                                 #   sets, linalg, stats, graph, KB roundtrip, confidence —
│                                 #   all byte-compared via std.mem.sliceAsBytes. Any
│                                 #   difference = test failure
│
├── runner/                       # Autonomous execution loops (Turns 28-29)
│   ├── types.zig                 #   VlpRunner (72 bytes), VlpRunnerAction enum, config structs
│   │                             #   (PollerConfig, ProcessorConfig, InternalConfig,
│   │                             #   BatchConfig), RunnerStatus
│   ├── pool.zig                  #   ThreadPool: fixed worker threads, TaskQueue (circular
│   │                             #   buffer of 256 tasks with mutex), workerMain (pop-blocking
│   │                             #   with 10ms sleep on empty). RunnerTable: 64 runner slots
│   ├── poller.zig                #   pollerIteration: fire Prolog rules on scope KB, report
│   │                             #   results. pollerLoop: timer → iterate → error tracking →
│   │                             #   output routing to notification KB
│   ├── processor.zig             #   processorIteration: try rule-based compaction (L3), fall
│   │                             #   through to LLM (L1). processorRecycle: snapshot → kill →
│   │                             #   clone → restore. processorReconnect: exponential backoff
│   │                             #   1s → 60s cap
│   ├── internal.zig              #   internalIteration: call compute_fn. Timer loop, error
│   │                             #   tracking. Default stubs: rollingAverage, trendDetection,
│   │                             #   coverageGap
│   ├── batch.zig                 #   batchIteration: reap completed clones → merge results →
│   │                             #   spawn new clones (max 16 concurrent). Clone-per-task
│   │                             #   isolation
│   ├── runner_manager.zig        #   RunnerManager: table + pool. createPoller,
│   │                             #   startRunner, stopRunner, killRunner, recycleRunner,
│   │                             #   getStatus, shutdown
│   ├── runner_ops.zig            #   Convenience wrappers for runner creation
│   └── sre_deployment.zig        #   SreDeployment: creates 4 runners (prometheus processor,
│                                 #   deploy processor, triage poller, hygiene internal)
│
├── server/                       # Network server (Turns 30-32)
│   ├── types.zig                 #   VlpServer, ServerConnection (fd, session_id, credential,
│   │                             #   state, buffers), ServerCredential (user_id, visibility,
│   │                             #   grants[16], TTL), ServerConfig, ServerMetrics, HealthReport
│   ├── listener.zig              #   createListenSocket (socket + REUSEADDR + bind + listen),
│   │                             #   acceptLoop (accept → capacity check → slot allocation),
│   │                             #   socketRead/Write, closeSocket, findFreeSlot
│   ├── auth.zig                  #   authenticate: FNV-1a hash token → scan auth KB → load
│   │                             #   visibility + grants + status. Auth KB layout: user_id*4+0
│   │                             #   = token hash, +1 = visibility, +2 = grant KB ref, +3 =
│   │                             #   account status. credentialCheck: 2 integer comparisons
│   │                             #   (valid AND now < expires_at). credentialRevoke: valid=false
│   ├── handler.zig               #   handleConnection: read → parse → authenticate →
│   │                             #   handleRequestLoop (credential check → rate limit → read
│   │                             #   request → processRequest → sendResponse → keepalive).
│   │                             #   Routes: /health, /metrics, /kb/*, /query, 404
│   ├── rate_limit.zig            #   Per-user counter in KB: slot user_id*2 = counter, slot
│   │                             #   user_id*2+1 = window start. New window resets counter.
│   │                             #   counter ≥ max → denied with retry_after
│   ├── health.zig                #   collectHealth: read atomic counters. renderHealthJson:
│   │                             #   build JSON from integers, no LLM involved
│   ├── reaper.zig                #   reaperScan: iterate connections, 3 integer checks (idle
│   │                             #   timeout, credential expiry, turn budget)
│   ├── shutdown.zig              #   gracefulShutdown: set flag → close listen socket → drain
│   │                             #   connections with timeout → force-close → snapshot
│   │                             #   persistent sessions
│   └── server_main.zig           #   ServerRuntime: server + accept/reaper threads.
│                                 #   init, start, stop, isRunning, getHealth
│
├── protocol/                     # Wire format handling (Turns 33-34)
│   ├── http.zig                  #   Full HTTP/1.1 parser (state machine: method, path,
│   │                             #   version, headers, body). Extracts Content-Length,
│   │                             #   Connection, Content-Type, Authorization, Upgrade:websocket.
│   │                             #   Response assembly from grammar-rendered parts
│   ├── websocket.zig             #   Upgrade handshake, frame read/write (2-byte header,
│   │                             #   extended length 126/127, mask key, payload unmasking).
│   │                             #   Text/binary/ping/pong/close. Credential expiry → close
│   │                             #   frame code 4001
│   ├── grammars.zig              #   10 protocol grammar templates: HTTP status line, header,
│   │                             #   content-type, JSON body/error, WebSocket close, health,
│   │                             #   SMTP greeting/ok, MQTT connack
│   ├── protocol_router.zig       #   Peek connection, dispatch to HTTP or WebSocket
│   ├── smtp.zig                  #   SMTP state machine stub (greeting → ehlo → mail_from →
│   │                             #   rcpt_to → data → ehlo)
│   └── mqtt.zig                  #   MQTT stub (CONNECT/CONNACK/PUBLISH/PINGREQ/DISCONNECT)
│
├── ops/                          # Grant-gated operations (Turn 34)
│   ├── filesystem.zig            #   fsRead, fsWrite, fsAppend, fsDelete, fsStat — std.fs
│   │                             #   wrappers, each checks grant before any side effect.
│   │                             #   fsReadToKB: file → text store → fact assert with
│   │                             #   script provenance (confidence 62259/65536)
│   ├── network.zig               #   netFetch: stub returning placeholder. netFetchToKB:
│   │                             #   fetch → fact assert with rest_api provenance (55705/65536)
│   ├── execute.zig               #   execRun: stub. execRunToKB: exec → fact assert with
│   │                             #   script provenance
│   ├── compile_check.zig         #   compileCheck: balanced delimiter checker ({}, (), [])
│   ├── process.zig               #   procStart (stub), procKill, procStatus
│   └── ops_dispatch.zig          #   Builtin wrappers, registerOpsBuiltins (IDs 500-510),
│                                 #   all registered with pure=false
│
├── config/                       # System configuration (Turn 35)
│   ├── system_config.zig         #   SystemConfig: all fields with defaults (port 8080,
│   │                             #   64 connections, 100K KBs, 10M facts, 100K rules)
│   ├── cli.zig                   #   parseCli: --config, --port, --help, --version, --test
│   ├── config_file.zig           #   parseConfigFile: key=value line parser with indent nesting
│   ├── integration_test.zig      #   13 checks: seed init, KB CRUD, fact roundtrip, Prolog
│   │                             #   fire, grammar compile, confidence combine, auth flow,
│   │                             #   rate limiting, health check, runner creation, SRE
│   │                             #   scenario, determinism, builtin dispatch
│   └── main.zig                  #   Entry point: parse CLI, load config, run tests or start
│
├── gpu/                          # GPU kernels — CPU fallback (Turns 36-39)
│   ├── device.zig                #   DeviceState, DeviceProps. deviceInit creates CPU fallback.
│   │                             #   DeviceType enum: cpu_fallback/gpu_int8/gpu_int8_fru/
│   │                             #   gpu_native_qiu
│   ├── memory.zig                #   DeviceMemoryLayout: 13 regions (model_weights, kb_store,
│   │                             #   fact_store, rule_store, term_store, text_store,
│   │                             #   grammar_store, live_state, scratch, audit, grant_store,
│   │                             #   session_table, path_index). Sequential placement with
│   │                             #   256-byte alignment. CPU fallback = host buffer
│   ├── transfer.zig              #   hostToDevice/deviceToHost/deviceToDevice: memcpy for CPU
│   │                             #   fallback. Typed variants for Q16 arrays and KB structs
│   ├── profiling.zig             #   KernelStats, SessionStats, Profiler. verifyDeterminism:
│   │                             #   run N times, byte-compare all outputs
│   ├── benchmarks.zig            #   8 benchmarks: forward pass, softmax, attention, sort,
│   │                             #   layerNorm, prologUnify, elementwise, gemm
│   ├── determinism.zig           #   verifyDeterminismSoftmax/Gemm/Attention: 100 runs each,
│   │                             #   byte-compare. all_identical must be true
│   └── kernels/
│       ├── gemm.zig              #   q16Gemm: C = alpha*op(A)*op(B) + beta*C, widening i64
│       │                         #   MAC, scaled by alpha/D². Batched and strided variants
│       ├── softmax.zig           #   q16Softmax: shift-square-divide, last element absorbs,
│       │                         #   sum = D. q16SoftmaxBatched: per-row. verifySoftmaxSum
│       ├── elementwise.zig       #   q16Add/Sub/Mul/Div/Scale/Dot/Compare/Negate/Abs/Min/Max/
│       │                         #   Clamp/Fill/Copy/Sum — delegate to Q16 methods over arrays
│       ├── normalize.zig         #   q16LayerNorm: exact mean + variance, intSqrt for inv_std.
│       │                         #   q16RMSNorm: exact mean of squares, intSqrt for inv_rms
│       ├── activation.zig        #   q16ReLU (max(0,x)), q16GELU (linear sigmoid approx),
│       │                         #   q16SiLU (sigmoid approx × input)
│       ├── attention.zig         #   fusedAttentionForward: per-head QKᵀ → softmax → AV.
│       │                         #   fusedAttentionWithKVCache: single query vs cached K/V
│       ├── sort.zig              #   q16Sort (merge sort), q16ArgSort (selection sort
│       │                         #   descending), q16TopK (partial selection)
│       ├── prolog_kernel.zig     #   batchUnify: query vs N candidates, return match bitmap.
│       │                         #   batchCrossMultiplyCompare: N×M match matrix.
│       │                         #   parallelRuleEval: N rules × M facts fire matrix
│       └── reduction.zig         #   q16ReduceSum/Max/Min/ArgMax/ArgMin. allReduceSum/Max
│                                 #   stubs for distributed (single-rank passthrough)
│
├── deploy/                       # Multi-device and deployment (Turn 40)
│   ├── distributed.zig           #   Comm struct. allReduceSum/Max/Min: single-rank passthrough.
│   │                             #   Integer sum is associative → deterministic regardless of
│   │                             #   reduction topology. kbSync, snapshotBroadcast
│   ├── model_parallel.zig        #   ModelParallel: divide layers across shards. pipelineForward
│   │                             #   (stub: passthrough per layer)
│   ├── load_balancer.zig         #   LoadBalancer: round_robin and least_connections strategies.
│   │                             #   addBackend, route, markUnhealthy/Healthy
│   ├── prometheus_export.zig     #   exportPrometheus: all metrics as "metric_name value\n"
│   ├── chaos.zig                 #   testSnapshotRecovery, testKillRestart,
│   │                             #   testConcurrentWrite, testDeterminismAfterRestart
│   └── deploy_main.zig           #   deployAndVerify: integration → chaos → benchmarks →
│                                 #   determinism → print results
│
└── main.zig                      # Entry point (Turn 35)

config/
├── dev.yaml                      # Development configuration
├── test_local.yaml               # Local test configuration
├── test_gcp.yaml                 # GCP test configuration
└── production.yaml               # Production configuration

scripts/
├── setup_gcp.sh                  # GCP instance creation
├── upload_and_test.sh            # Upload source, run test phases
└── bench.sh                      # Run GPU benchmarks

build.zig                         # Build file with phased test targets
```

---

## Build Phases

The system builds in 6 phases with strict dependency ordering. Each phase's tests pass before the next begins.

| Phase | Turns | What It Delivers | Lines |
|-------|-------|-----------------|-------|
| **1 — Foundation** | 1-6 | Q16 types, KB store, fact CRUD, visibility, grants, audit, confidence | ~5,900 |
| **2 — Intelligence** | 7-14 | Prolog engine, grammar engine, bounded primitives, session lifecycle, snapshots | ~8,500 |
| **3 — Engine** | 15-27 | Universal cycle, LLM forward pass, 448 builtins, seed layer, SRE scenario test | ~12,900 |
| **4 — Operations** | 28-35 | Runner system, server, protocol handlers, ops, config, integration test | ~9,800 |
| **5 — GPU** | 36-39 | Device kernels (CPU fallback), profiling, benchmarks, determinism | ~4,200 |
| **6 — Deploy** | 40 | Multi-device, distributed, load balancer, chaos tests | ~1,200 |

**Total: ~42,200 lines across ~200 source files and ~70 test files.**

---

## Key Invariants

These hold at all times, in all states, on all devices. Violation is a bug.

1. **Softmax outputs sum to D exactly.** Integer equality, not tolerance.
2. **KB facts at integer addresses are exact.** Turn 1 or turn 1,000,000.
3. **Bounded primitives cannot exceed declared bounds.** LRU capacity 1000 → at most 1000 entries.
4. **Snapshot restore is bit-identical.** Save → modify → restore → state == saved. Every byte.
5. **Clone COW is invisible to parent.** Clone writes never modify parent state.
6. **Access-denied data is absent, not filtered.** Query returns zero results, not redacted results.
7. **Grant denial happens before execution.** No partial execution on denied operations.
8. **Integer arithmetic is deterministic across devices.** Same inputs → same outputs. Always.
9. **Prolog unification uses exact comparison.** No tolerance. No epsilon.
10. **Audit log is append-only and complete.** Every operation produces an entry.

---

## The Paper Series

This implementation is the Zig 0.16.0 realization of a 20-paper specification:

| Paper | Topic | What It Contributes |
|-------|-------|-------------------|
| VDR-1 | Arithmetic | The [V, D, R] triple, exact operations, linear algebra |
| VDR-4 | LM Architecture | Complete transformer in exact fractions |
| VDR-5 | Logic + Provenance | Prolog, scoped KBs, constraints, working data |
| VDR-6 | Primitives | 448 builtins, grants, command tokens, environments |
| VDR-7 | Lifecycle | 12 phases from data sourcing through retirement |
| VDR-8 | State + Sessions | 7 data primitives, dotted paths, snapshots, clones |
| VDR-10 | Foundations | IOSE model, 15 operational principles, number types |
| VDR-12 | Grammars | Template system, compaction, vocabulary filtering |
| VDR-14 | Consolidation | Complete system specification (all above unified) |
| VDR-15 | Token Economics | 85-97% token reduction, flat per-turn cost |
| VDR-16 | Safety | Three-layer structural safety, jailbreak impossibility for data |
| VDR-17 | Alignment | Structural HHH, credential model, session scoring |
| VDR-18 | Performance | GPU mapping, 150× per-token but 10× net at 95% reduction |
| VDR-19 | Self-Extension | Usage as training, seed layer, accumulation curve |
| VDR-20 | Deployment | Four runner types, coverage loops, owner interface |
| VDR-21 | FPGA | 10-core prototype, 53-instruction ISA, 884 tests |
| VDR-22 | ASIC | 5,120 QIUs, 4nm, 5.1T muls/sec, no float hardware |
| VDR-23 | FRU | Functional Remainder Unit, exact transcendentals |
| VDR-24 | LM Software | Application category: develop → snapshot → clone → improve |
| VDR-25 | Server Software | 44 protocols as LM Software services |

---

## TensorProlog API Surface

The system exposes ~580 functions across 23 modules:

| Module | Functions | What It Replaces |
|--------|-----------|-----------------|
| core | 11 | Runtime init, device management |
| memory | 10 | Typed device memory allocation |
| stream | 10 | Session-aware execution streams |
| session | 10 | Snapshot/clone/kill lifecycle *(new capability)* |
| launch | 5 | Kernel dispatch with scheduling hints |
| vdr_math | 17 | cuBLAS (~200+ functions) |
| attention | 3 | cuDNN attention (~50+ functions) |
| training | 10 | Training loops + AMP + loss scaling |
| kb | 14 | Knowledge base store *(new capability)* |
| kb_primitives | 30 | 7 bounded data structures *(new capability)* |
| prolog | 8 | Prolog engine on GPU *(new capability)* |
| grammar | 7 | Structural token generation *(new capability)* |
| runner | 8 | Autonomous execution loops *(new capability)* |
| safety | 6 | Structural access control *(replaces guardrails)* |
| confidence | 5 | Exact provenance *(new capability)* |
| distributed | 10 | NCCL (~100+ functions) |
| transform | 4 | cuFFT (~80+ functions) |
| linalg | 8 | cuSOLVER (~200+ functions) |
| stats | 8 | Exact statistics |
| numbertheory | 7 | Crypto primitives |
| functional_remainder | 8 | FRU transcendentals *(replaces SFU)* |
| builtins | 448 | Deterministic primitives *(replaces LLM token generation)* |
| profiling | 4 | Simplified diagnostics *(replaces Nsight)* |

**Eliminated:** ~3,400+ API functions from float precision variants, mixed-precision management, NaN handling, loss scaling, Transformer Engine, TensorRT calibration.

**Added:** ~90 functions enabling autonomous operation, persistent state, structural safety, and exact provenance — capabilities that don't exist in any form in CUDA.

---

## Running

```bash
# Build and run all Phase 1 tests
zig build test-phase1

# Run full integration test
zig build test-phase4

# Run SRE scenario and determinism verification
zig build test-phase3

# Start server
zig build run -- --config config/dev.yaml

# Run benchmarks
zig build bench
```

---

## Current Status

Turns 1-40 implemented. Known stubs requiring real implementations:

1. KV cache stores vector sums as single facts (needs vector-typed facts)
2. Mapping builtin wrappers need KB-backed VlpMap integration
3. Eigenvalues for n>2 returns diagonal approximation (needs QR iteration)
4. SVD returns identity U/Vt (needs Golub-Kahan)
5. `netFetch` is a stub (needs HTTP client)
6. `execRun` is a stub (needs subprocess via `std.process.Child`)
7. Processor recycle uses placeholder session ID allocation
8. WebSocket accept key uses simple hash (needs SHA-1 + base64)
9. Distributed operations are single-rank passthroughs (needs transport)
10. Model parallel forward is passthrough (needs cross-device transfer)
11. GELU/SiLU use linear approximations (FRU-based versions planned)
12. `build.zig` needs uncommenting and path adjustment

~180 of 448 target builtins registered. Collections and sets need dispatch registration. Polynomial, finite field, denominator management categories not yet implemented.

---

## Design Principles

- **No float anywhere.** Every number is an exact integer or exact rational.
- **Prefer i32.** Use i64 only where range requires it.
- **Runtime over comptime.** Keep things inspectable and debuggable.
- **Bounded at creation.** Every data structure declares its maximum size.
- **No heap in hot paths.** Pre-allocated slabs, arena allocators.
- **Fixed-size structs.** KB = 256 bytes. Fact = 40 bytes. Rule = 44 bytes. All aligned.
- **Dependencies flow downward.** No module imports from a later build phase.
- **Tests ship with source.** Every turn delivers implementation and tests together.
- **Fix forward.** Bugs found in turn N get fixed in turn N, not by modifying turn M.

---

## License

MIT

---

## Related Work

- [VDR and VDR-LLM-Prolog series](https://sireus.cloud/vdr-llm-prolog/) — The 34-paper specification this implements on the GPU
- [TensorProlog tech spec](docs/tech_spec.md) — Technical specification
- [TensorProlog cookbook](docs/cookbook.md) — ML practitioner workflow reference

