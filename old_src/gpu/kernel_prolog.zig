// ============================================================
// src/gpu/kernels/prolog_kernel.zig
// ============================================================

const prolog_types = @import("../../prolog/types.zig");
const VlpTerm = prolog_types.VlpTerm;
const VlpTermType = prolog_types.VlpTermType;
const VlpBinding = prolog_types.VlpBinding;

pub const UnifyBatchResult = struct {
    matched: [256]bool,
    n_candidates: i32,
    n_matched: i32,
};

pub fn batchUnify(
    query: *const VlpTerm,
    candidates: []const VlpTerm,
    n_candidates: i32,
    result: *UnifyBatchResult,
) VlpStatus {
    const nc: usize = @intCast(n_candidates);
    result.n_candidates = n_candidates;
    result.n_matched = 0;

    for (0..nc) |i| {
        result.matched[i] = unifyTerms(query, &candidates[i]);
        if (result.matched[i]) result.n_matched += 1;
    }

    return .ok;
}

fn unifyTerms(a: *const VlpTerm, b: *const VlpTerm) bool {
    if (a.term_type == .variable or b.term_type == .variable) return true;

    if (a.term_type != b.term_type) return false;

    switch (a.term_type) {
        .atom => return a.data.atom_id == b.data.atom_id,
        .integer => return a.data.int_value == b.data.int_value,
        .vdr => {
            const av: i64 = @intCast(a.data.vdr_value.v);
            const bv: i64 = @intCast(b.data.vdr_value.v);
            return av == bv;
        },
        .text => {
            return a.data.text_offset == b.data.text_offset and
                a.data.text_length == b.data.text_length;
        },
        .compound => {
            if (a.data.functor_id != b.data.functor_id) return false;
            if (a.data.args_count != b.data.args_count) return false;
            return true;
        },
        .list => return true,
        else => return false,
    }
}

pub fn batchCrossMultiplyCompare(
    query_values: []const Q16,
    candidate_values: []const Q16,
    n_query: i32,
    n_candidates: i32,
    match_matrix: []bool,
) VlpStatus {
    const nq: usize = @intCast(n_query);
    const nc: usize = @intCast(n_candidates);

    for (0..nq) |qi| {
        for (0..nc) |ci| {
            const av: i64 = @intCast(query_values[qi].v);
            const bv: i64 = @intCast(candidate_values[ci].v);
            match_matrix[qi * nc + ci] = (av == bv);
        }
    }

    return .ok;
}

pub const ScopeFilterResult = struct {
    visible: [1024]bool,
    n_visible: i32,
};

pub fn scopeFilter(
    kb_visibilities: []const i8,
    kb_parents: []const i32,
    n_kbs: i32,
    session_visibility: i8,
    session_user_id: i32,
    kb_owners: []const i32,
    result: *ScopeFilterResult,
) VlpStatus {
    const nk: usize = @intCast(n_kbs);
    result.n_visible = 0;

    for (0..nk) |i| {
        const vis = kb_visibilities[i];
        var accessible = false;

        if (vis == 0) {
            accessible = true;
        } else if (vis == 1) {
            accessible = (session_visibility <= 1);
        } else if (vis == 2) {
            accessible = (kb_owners[i] == session_user_id);
        }

        if (accessible) {
            var parent = kb_parents[i];
            var depth: i32 = 0;
            while (parent >= 0 and depth < 100) {
                const pi: usize = @intCast(parent);
                if (pi >= nk) break;
                const pvis = kb_visibilities[pi];
                if (pvis == 2 and kb_owners[pi] != session_user_id) {
                    accessible = false;
                    break;
                }
                if (pvis == 1 and session_visibility > 1) {
                    accessible = false;
                    break;
                }
                parent = kb_parents[pi];
                depth += 1;
            }
        }

        if (i < result.visible.len) {
            result.visible[i] = accessible;
            if (accessible) result.n_visible += 1;
        }
    }

    return .ok;
}

pub fn parallelRuleEval(
    rule_heads: []const VlpTerm,
    fact_values: []const VlpTerm,
    n_rules: i32,
    n_facts: i32,
    fire_matrix: []bool,
) VlpStatus {
    const nr: usize = @intCast(n_rules);
    const nf: usize = @intCast(n_facts);

    for (0..nr) |r| {
        var any_match = false;
        for (0..nf) |f| {
            const matched = unifyTerms(&rule_heads[r], &fact_values[f]);
            fire_matrix[r * nf + f] = matched;
            if (matched) any_match = true;
        }
        _ = any_match;
    }

    return .ok;
}
