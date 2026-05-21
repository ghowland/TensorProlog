**vlp_gpu_shared.zig** — no function changes, one comment update:

```zig
// Change comment at FACT_VALUE_R0:
pub const FACT_VALUE_R0: i32 = 2;   // packed: r0 in lower 16 bits, r1 in upper 16 bits
```

---

**vlp_kernel.zig** — `op_unify_candidates` VDR match block:

```zig
// In op_unify_candidates, replace the VDR comparison block:

    } else if (q_type == @intFromEnum(shared.TermType.vdr)) {
        const fact_r0_r1 = fact_store.data[@intCast(fb + shared.FACT_VALUE_R0)];
        const query_r0_r1 = p(shared.P_FIELD_5);
        matched = (fact_v == p(shared.P_FIELD_4)) and (fact_r0_r1 == query_r0_r1);
    }
```

---

**vlp_gpu_params.zig** — `unifyCandidates`:

```zig
pub fn unifyCandidates(n_cand: i32, q_type: i32, q_atom: i32, q_int: i32, q_vdr_v: i32, q_vdr_r0_r1: i32, q_func: i32, q_argc: i32, q_argoff: i32, max_bind: i32) ParamsBuffer {
    var buf = initParams(.unify_candidates);
    buf[@intCast(shared.P_FIELD_0)] = n_cand;
    buf[@intCast(shared.P_FIELD_1)] = q_type;
    buf[@intCast(shared.P_FIELD_2)] = q_atom;
    buf[@intCast(shared.P_FIELD_3)] = q_int;
    buf[@intCast(shared.P_FIELD_4)] = q_vdr_v;
    buf[@intCast(shared.P_FIELD_5)] = q_vdr_r0_r1; // packed: r0 lower 16, r1 upper 16
    buf[@intCast(shared.P_FIELD_6)] = q_func;
    buf[@intCast(shared.P_FIELD_7)] = q_argc;
    buf[@intCast(shared.P_FIELD_8)] = q_argoff;
    buf[@intCast(shared.P_FIELD_9)] = max_bind;
    return buf;
}
```

---

**vlp_prolog.zig** — `unifyCandidatesGpu`, the line that builds params:

```zig
// In unifyCandidatesGpu, replace the params construction:

    var params_buf = gpu_params.unifyCandidates(
        n,
        @intFromEnum(query_term.term_type),
        query_term.primary_id,
        query_term.primary_id,
        query_term.vdr_value.v,
        query_term.vdr_value.toInts()[1], // packed r0|r1
        query_term.primary_id,
        @intCast(query_term.secondary_aux),
        query_term.secondary_offset,
        shared.MAX_BINDINGS_PER,
    );
```

---

**vlp_types.zig** — `Fact.toInts` and `Fact.fromInts`:

```zig
    pub fn toInts(self: Fact) [10]i32 {
        const val = self.value.toInts();
        const prov = self.provenance.toInts();
        return .{ @intFromEnum(self.tag), val[0], val[1], prov[0], prov[1], prov[2], prov[3], prov[4], prov[5], prov[6] };
    }

    pub fn fromInts(ints: [10]i32) Fact {
        return .{
            .tag = @enumFromInt(ints[0]),
            .value = Q16.fromInts(.{ ints[1], ints[2] }),
            .provenance = Provenance.fromInts(.{ ints[3], ints[4], ints[5], ints[6], ints[7], ints[8], ints[9] }),
        };
    }
```

No change needed — these already delegate to `Q16.toInts`/`fromInts` which now packs r1. Confirming they're correct as-is.

---

**vlp_types.zig** — `Provenance.toInts` confidence field:

```zig
    pub fn toInts(self: Provenance) [7]i32 {
        const conf = self.confidence.toInts();
        return .{
            self.source_type,
            self.source_kb_id,
            self.source_slot_id,
            conf[0],        // confidence.v
            conf[1],        // confidence r0|r1 packed
            self.timestamp,
            self.derivation_rule_id,
        };
    }

    pub fn fromInts(ints: [7]i32) Provenance {
        return .{
            .source_type = ints[0],
            .source_kb_id = ints[1],
            .source_slot_id = ints[2],
            .confidence = Q16.fromInts(.{ ints[3], ints[4] }),
            .timestamp = ints[5],
            .derivation_rule_id = ints[6],
        };
    }
```

Already correct — delegates to `Q16.toInts`/`fromInts`. No change needed.

---

That's everything. 4 actual code changes:
1. `vlp_gpu_shared.zig` — comment
2. `vlp_kernel.zig` — `op_unify_candidates` VDR block
3. `vlp_gpu_params.zig` — `unifyCandidates` parameter name
4. `vlp_prolog.zig` — `unifyCandidatesGpu` params construction