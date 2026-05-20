// ============================================================
// src/llm/kv_cache.zig
// ============================================================

const q16_mod = @import("../vdr/q16.zig");
const Q16 = q16_mod.Q16;

const kb_types = @import("../kb/types.zig");
const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");

const VlpFact = kb_types.VlpFact;
const VlpFactTag = kb_types.VlpFactTag;
const VlpSourceType = kb_types.VlpSourceType;
const VlpProvenance = kb_types.VlpProvenance;
const KBStore = kb_store_mod.KBStore;

pub const KVCacheConfig = struct {
    n_layers: i32,
    max_seq_len: i32,
    n_heads: i32,
    d_head: i32,
    parent_kb_id: i32,
};

pub const KVCache = struct {
    kb_id: i32,
    n_layers: i32,
    max_seq_len: i32,
    n_heads: i32,
    d_head: i32,
    store: *KBStore,
    current_len: i32,

    pub fn init(store: *KBStore, config: KVCacheConfig) KVCache {
        const max_facts: i32 = config.n_layers * config.max_seq_len * config.n_heads * 2;
        const kb_id = store.createKB(.{
            .name = "kv_cache",
            .parent_id = config.parent_kb_id,
            .visibility = .internal,
            .owner = "system",
            .max_facts = max_facts,
            .max_rules = 0,
            .max_children = 0,
        });
        return .{
            .kb_id = kb_id,
            .n_layers = config.n_layers,
            .max_seq_len = config.max_seq_len,
            .n_heads = config.n_heads,
            .d_head = config.d_head,
            .store = store,
            .current_len = 0,
        };
    }

    pub fn slotIndex(self: *const KVCache, layer: i32, position: i32, head: i32, is_value: bool) i32 {
        const base = layer * self.max_seq_len * self.n_heads * 2;
        const pos_offset = position * self.n_heads * 2;
        const head_offset = head * 2;
        const kv_offset: i32 = if (is_value) 1 else 0;
        return base + pos_offset + head_offset + kv_offset;
    }

    pub fn append(
        self: *KVCache,
        layer: i32,
        position: i32,
        k_vecs: []const Q16,
        v_vecs: []const Q16,
    ) void {
        const d: usize = @intCast(self.d_head);
        const nh: usize = @intCast(self.n_heads);

        for (0..nh) |h| {
            const head_start = h * d;
            const hi: i32 = @intCast(h);

            var k_sum: i64 = 0;
            for (0..d) |dd| {
                k_sum += @intCast(k_vecs[head_start + dd].v);
            }

            const k_fact = VlpFact{
                .tag = .vector,
                .value = .{
                    .v = @intCast(k_sum),
                    .r0 = 0,
                },
                .provenance = makeKVProvenance(self.kb_id, self.slotIndex(layer, position, hi, false)),
            };
            _ = fact_mod.factAssert(self.store, self.kb_id, self.slotIndex(layer, position, hi, false), &k_fact);

            var v_sum: i64 = 0;
            for (0..d) |dd| {
                v_sum += @intCast(v_vecs[head_start + dd].v);
            }

            const v_fact = VlpFact{
                .tag = .vector,
                .value = .{
                    .v = @intCast(v_sum),
                    .r0 = 0,
                },
                .provenance = makeKVProvenance(self.kb_id, self.slotIndex(layer, position, hi, true)),
            };
            _ = fact_mod.factAssert(self.store, self.kb_id, self.slotIndex(layer, position, hi, true), &v_fact);
        }

        if (position >= self.current_len) {
            self.current_len = position + 1;
        }
    }

    pub fn loadRange(
        self: *KVCache,
        layer: i32,
        start_pos: i32,
        end_pos: i32,
        k_out: []Q16,
        v_out: []Q16,
    ) void {
        const nh: usize = @intCast(self.n_heads);
        const d: usize = @intCast(self.d_head);
        const sp: usize = @intCast(start_pos);
        const ep: usize = @intCast(end_pos);

        for (sp..ep) |pos| {
            const pi: i32 = @intCast(pos);
            const pos_offset = (pos - sp) * nh * d;

            for (0..nh) |h| {
                const hi: i32 = @intCast(h);
                const head_offset = h * d;

                const k_slot = self.slotIndex(layer, pi, hi, false);
                const k_fact = fact_mod.factQuery(self.store, self.kb_id, k_slot);
                if (k_fact) |kf| {
                    for (0..d) |dd| {
                        k_out[pos_offset + head_offset + dd] = .{
                            .v = @divTrunc(kf.value.v, @as(i32, @intCast(d))),
                            .r0 = 0,
                        };
                    }
                }

                const v_slot = self.slotIndex(layer, pi, hi, true);
                const v_fact = fact_mod.factQuery(self.store, self.kb_id, v_slot);
                if (v_fact) |vf| {
                    for (0..d) |dd| {
                        v_out[pos_offset + head_offset + dd] = .{
                            .v = @divTrunc(vf.value.v, @as(i32, @intCast(d))),
                            .r0 = 0,
                        };
                    }
                }
            }
        }
    }

    pub fn truncate(self: *KVCache, position: i32) void {
        const nl: usize = @intCast(self.n_layers);
        const ms: usize = @intCast(self.max_seq_len);
        const nh: usize = @intCast(self.n_heads);

        for (0..nl) |layer| {
            const li: i32 = @intCast(layer);
            const pi_start: usize = @intCast(position);
            for (pi_start..ms) |pos| {
                const pi: i32 = @intCast(pos);
                for (0..nh) |h| {
                    const hi: i32 = @intCast(h);
                    _ = fact_mod.factRetract(self.store, self.kb_id, self.slotIndex(li, pi, hi, false));
                    _ = fact_mod.factRetract(self.store, self.kb_id, self.slotIndex(li, pi, hi, true));
                }
            }
        }

        if (position < self.current_len) {
            self.current_len = position;
        }
    }

    pub fn currentLength(self: *const KVCache) i32 {
        return self.current_len;
    }
};

fn makeKVProvenance(kb_id: i32, slot_id: i32) VlpProvenance {
    return .{
        .source_type = .vdr_computation,
        .source_kb_id = kb_id,
        .source_slot_id = slot_id,
        .confidence = .{ .v = Q16.D, .r0 = 0 },
        .timestamp = 0,
        .derivation_rule_id = -1,
    };
}
