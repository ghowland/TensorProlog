// ============================================================
// src/prolog/hygiene.zig
// ============================================================

const prolog_types = @import("types.zig");
const rule_mod = @import("rule.zig");
const safety_types = @import("../safety/types.zig");
const grant_mod = @import("../safety/grant.zig");

const VlpRule = prolog_types.VlpRule;
const HygieneCandidate = prolog_types.HygieneCandidate;
const RuleStore = rule_mod.RuleStore;
const GrantStore = grant_mod.GrantStore;

pub fn hygieneScan(
    rs: *const RuleStore,
    gs: *const GrantStore,
    stale_seconds: i32,
    min_success_num: i32,
    min_success_den: i32,
    now: i32,
    out: []HygieneCandidate,
) i32 {
    var found: i32 = 0;
    const max: i32 = @intCast(out.len);
    var i: i32 = 0;
    while (i < rs.rule_count) : (i += 1) {
        if (found >= max) break;
        const r = &rs.rules[@intCast(i)];
        if (!r.alive) continue;

        if (isStale(r, stale_seconds, now)) {
            out[@intCast(found)] = .{ .rule_id = r.id, .reason = .stale, .detail = now - r.last_fired };
            found += 1;
            continue;
        }

        if (isFailing(r, min_success_num, min_success_den)) {
            out[@intCast(found)] = .{ .rule_id = r.id, .reason = .failing, .detail = r.success_count };
            found += 1;
            continue;
        }

        if (isOrphaned(r, rs, gs)) {
            out[@intCast(found)] = .{ .rule_id = r.id, .reason = .orphaned, .detail = r.id };
            found += 1;
            continue;
        }
    }
    return found;
}

fn isStale(r: *const VlpRule, stale_seconds: i32, now: i32) bool {
    if (r.fire_count == 0 and r.created_at > 0) {
        return (now - r.created_at) > stale_seconds;
    }
    if (r.last_fired == 0) return false;
    return (now - r.last_fired) > stale_seconds;
}

fn isFailing(r: *const VlpRule, min_num: i32, min_den: i32) bool {
    const total = r.success_count + r.failure_count;
    if (total < 5) return false;
    return r.success_count * min_den < total * min_num;
}

fn isOrphaned(r: *const VlpRule, rs: *const RuleStore, gs: *const GrantStore) bool {
    if (r.action_count <= 0 or r.action_offset < 0) return false;
    var ai: i32 = 0;
    while (ai < r.action_count) : (ai += 1) {
        const idx: usize = @intCast(r.action_offset + ai);
        if (idx >= rs.actions.len) continue;
        const act = &rs.actions[idx];
        if (act.fact.provenance.derivation_rule_id >= 0) {
            const gid = act.fact.provenance.derivation_rule_id;
            if (gid >= 0 and gid < gs.count) {
                if (gs.grants[@intCast(gid)].state == .revoked) return true;
            }
        }
    }
    return false;
}
