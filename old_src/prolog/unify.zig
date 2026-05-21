// ============================================================
// src/prolog/unify.zig
// ============================================================

const prolog_types = @import("types.zig");
const term_mod = @import("term.zig");

const VlpTerm = prolog_types.VlpTerm;
const VlpTermType = prolog_types.VlpTermType;
const BindingSet = prolog_types.BindingSet;
const Q16 = prolog_types.Q16;

pub fn unify(a_raw: VlpTerm, b_raw: VlpTerm, bindings: *BindingSet, term_store: []const VlpTerm, depth: i32) bool {
    if (depth <= 0) return false;

    const a = term_mod.resolve(a_raw, bindings);
    const b = term_mod.resolve(b_raw, bindings);

    if (a.ttype == .variable and b.ttype == .variable) {
        if (a.d.var_id == b.d.var_id) return true;
        return bindings.bind(a.d.var_id, b);
    }

    if (a.ttype == .variable) {
        if (term_mod.containsVar(term_store, termIndex(b, term_store), a.d.var_id)) return false;
        return bindings.bind(a.d.var_id, b);
    }

    if (b.ttype == .variable) {
        if (term_mod.containsVar(term_store, termIndex(a, term_store), b.d.var_id)) return false;
        return bindings.bind(b.d.var_id, a);
    }

    if (a.ttype != b.ttype) return false;

    return switch (a.ttype) {
        .atom => a.d.atom_id == b.d.atom_id,
        .integer => a.d.int_value == b.d.int_value,
        .vdr => Q16.eql(a.d.vdr_value, b.d.vdr_value),
        .text => a.d.text.offset == b.d.text.offset and a.d.text.length == b.d.text.length,
        .compound => unifyCompound(a, b, bindings, term_store, depth),
        .list => unifyList(a, b, bindings, term_store, depth),
        else => false,
    };
}

fn unifyCompound(a: VlpTerm, b: VlpTerm, bindings: *BindingSet, term_store: []const VlpTerm, depth: i32) bool {
    const ac = a.d.compound;
    const bc = b.d.compound;
    if (ac.functor_id != bc.functor_id) return false;
    if (ac.args_count != bc.args_count) return false;
    var i: i32 = 0;
    while (i < ac.args_count) : (i += 1) {
        const ai: usize = @intCast(ac.args_offset + i);
        const bi: usize = @intCast(bc.args_offset + i);
        if (ai >= term_store.len or bi >= term_store.len) return false;
        if (!unify(term_store[ai], term_store[bi], bindings, term_store, depth - 1)) return false;
    }
    return true;
}

fn unifyList(a: VlpTerm, b: VlpTerm, bindings: *BindingSet, term_store: []const VlpTerm, depth: i32) bool {
    const al = a.d.list;
    const bl = b.d.list;
    if (al.head < 0 and bl.head < 0) return true;
    if (al.head < 0 or bl.head < 0) return false;
    const ah: usize = @intCast(al.head);
    const bh: usize = @intCast(bl.head);
    if (ah >= term_store.len or bh >= term_store.len) return false;
    if (!unify(term_store[ah], term_store[bh], bindings, term_store, depth - 1)) return false;
    if (al.tail < 0 and bl.tail < 0) return true;
    if (al.tail < 0 or bl.tail < 0) return false;
    const at: usize = @intCast(al.tail);
    const bt: usize = @intCast(bl.tail);
    if (at >= term_store.len or bt >= term_store.len) return false;
    return unify(term_store[at], term_store[bt], bindings, term_store, depth - 1);
}

fn termIndex(t: VlpTerm, store: []const VlpTerm) i32 {
    for (store, 0..) |s, i| {
        if (term_mod.termEql(s, t)) return @intCast(i);
    }
    return -1;
}
