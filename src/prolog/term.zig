// ============================================================
// src/prolog/term.zig
// ============================================================

const prolog_types = @import("types.zig");
const VlpTerm = prolog_types.VlpTerm;
const VlpTermType = prolog_types.VlpTermType;
const TermData = prolog_types.TermData;
const Q16 = prolog_types.Q16;
const TextSpan = prolog_types.TextSpan;
const ListPair = prolog_types.ListPair;
const CompoundRef = prolog_types.CompoundRef;

pub fn makeAtom(id: i32) VlpTerm {
    return .{ .ttype = .atom, .d = .{ .atom_id = id } };
}

pub fn makeVar(id: i32) VlpTerm {
    return .{ .ttype = .variable, .d = .{ .var_id = id } };
}

pub fn makeInt(val: i32) VlpTerm {
    return .{ .ttype = .integer, .d = .{ .int_value = val } };
}

pub fn makeVdr(val: Q16) VlpTerm {
    return .{ .ttype = .vdr, .d = .{ .vdr_value = val } };
}

pub fn makeText(offset: i32, length: i16) VlpTerm {
    return .{ .ttype = .text, .d = .{ .text = .{ .offset = offset, .length = length } } };
}

pub fn makeList(head: i32, tail: i32) VlpTerm {
    return .{ .ttype = .list, .d = .{ .list = .{ .head = head, .tail = tail } } };
}

pub fn makeCompound(functor_id: i32, args_offset: i32, args_count: i16) VlpTerm {
    return .{ .ttype = .compound, .d = .{ .compound = .{ .functor_id = functor_id, .args_offset = args_offset, .args_count = args_count } } };
}

pub fn termEql(a: VlpTerm, b: VlpTerm) bool {
    if (a.ttype != b.ttype) return false;
    return switch (a.ttype) {
        .atom => a.d.atom_id == b.d.atom_id,
        .variable => a.d.var_id == b.d.var_id,
        .integer => a.d.int_value == b.d.int_value,
        .vdr => Q16.eql(a.d.vdr_value, b.d.vdr_value),
        .text => a.d.text.offset == b.d.text.offset and a.d.text.length == b.d.text.length,
        .list => a.d.list.head == b.d.list.head and a.d.list.tail == b.d.list.tail,
        .compound => a.d.compound.functor_id == b.d.compound.functor_id and a.d.compound.args_offset == b.d.compound.args_offset and a.d.compound.args_count == b.d.compound.args_count,
        .vector, .matrix, .pair => false,
    };
}

pub fn containsVar(term_store: []const VlpTerm, idx: i32, var_id: i32) bool {
    if (idx < 0 or idx >= @as(i32, @intCast(term_store.len))) return false;
    const t = term_store[@intCast(idx)];
    return switch (t.ttype) {
        .variable => t.d.var_id == var_id,
        .compound => {
            const c = t.d.compound;
            var i: i32 = 0;
            while (i < c.args_count) : (i += 1) {
                if (containsVar(term_store, c.args_offset + i, var_id)) return true;
            }
            return false;
        },
        .list => {
            if (containsVar(term_store, t.d.list.head, var_id)) return true;
            return containsVar(term_store, t.d.list.tail, var_id);
        },
        else => false,
    };
}

pub fn resolve(t: VlpTerm, bindings: *const prolog_types.BindingSet) VlpTerm {
    if (t.ttype != .variable) return t;
    const bound = bindings.lookup(t.d.var_id) orelse return t;
    if (bound.ttype == .variable) return resolve(bound, bindings);
    return bound;
}
