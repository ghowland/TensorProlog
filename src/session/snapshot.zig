// ============================================================
// src/session/snapshot.zig
// ============================================================

const std = @import("std");
const session_types = @import("types.zig");
const store_mod = @import("../kb/store.zig");
const kb_types = @import("../kb/types.zig");
const tree_mod = @import("../kb/tree.zig");
const rule_mod = @import("../prolog/rule.zig");
const prolog_types = @import("../prolog/types.zig");
const grant_mod = @import("../safety/grant.zig");
const safety_types = @import("../safety/types.zig");

const VlpSession = session_types.VlpSession;
const VlpStatus = session_types.VlpStatus;
const KBStore = store_mod.KBStore;
const VlpKB = kb_types.VlpKB;
const VlpFact = kb_types.VlpFact;
const VlpRule = prolog_types.VlpRule;
const VlpTerm = prolog_types.VlpTerm;
const PrologAction = prolog_types.PrologAction;
const RuleStore = rule_mod.RuleStore;
const GrantStore = grant_mod.GrantStore;
const VlpGrant = safety_types.VlpGrant;

pub const SNAPSHOT_MAGIC: [4]u8 = .{ 'V', 'L', 'P', 'S' };
pub const SNAPSHOT_VERSION: i32 = 1;

pub const SnapshotHeader = struct {
    magic: [4]u8 = SNAPSHOT_MAGIC,
    version: i32 = SNAPSHOT_VERSION,
    timestamp: i32 = 0,
    session_id: i32 = -1,
    user_id: i32 = -1,
    kb_count: i32 = 0,
    fact_count: i32 = 0,
    rule_count: i32 = 0,
    term_count: i32 = 0,
    action_count: i32 = 0,
    grant_count: i32 = 0,
    text_len: i32 = 0,
    session: VlpSession = .{},
    checksum: i32 = 0,
    total_size: i32 = 0,
};

pub const Snapshot = struct {
    header: SnapshotHeader,
    data: []u8,
    data_len: i32,
};

pub fn save(
    session: *const VlpSession,
    kb_store: *const KBStore,
    rule_store: *const RuleStore,
    grant_store: *const GrantStore,
    output: []u8,
) ?Snapshot {
    var pos: i32 = 0;
    const cap: i32 = @intCast(output.len);

    var header = SnapshotHeader{};
    header.session_id = session.id;
    header.user_id = session.user_id;
    header.timestamp = session.current_turn;
    header.session = session.*;

    const header_size: i32 = @intCast(@sizeOf(SnapshotHeader));
    if (pos + header_size > cap) return null;
    pos += header_size;

    var subtree: [256]i32 = undefined;
    const n_kbs = tree_mod.collectSubtree(kb_store, session.kb_root_id, &subtree);
    header.kb_count = n_kbs;

    var ki: i32 = 0;
    while (ki < n_kbs) : (ki += 1) {
        const kid = subtree[@intCast(ki)];
        const kb = kb_store.getKBConst(kid) orelse continue;
        const kb_size: i32 = @intCast(@sizeOf(VlpKB));
        if (pos + kb_size > cap) return null;
        const dst = output[@intCast(pos)..@intCast(pos + kb_size)];
        @memcpy(dst, std.mem.asBytes(kb));
        pos += kb_size;
    }

    var total_facts: i32 = 0;
    ki = 0;
    while (ki < n_kbs) : (ki += 1) {
        const kid = subtree[@intCast(ki)];
        const kb = kb_store.getKBConst(kid) orelse continue;
        const s: usize = @intCast(kb.facts_offset);
        const e: usize = @intCast(kb.facts_offset + kb.facts_capacity);
        for (kb_store.facts[s..e]) |f| {
            if (f.tag == .empty) continue;
            const fsize: i32 = @intCast(@sizeOf(VlpFact));
            if (pos + fsize > cap) return null;
            const dst = output[@intCast(pos)..@intCast(pos + fsize)];
            @memcpy(dst, std.mem.asBytes(&f));
            pos += fsize;
            total_facts += 1;
        }
    }
    header.fact_count = total_facts;

    var ri: i32 = 0;
    while (ri < rule_store.rule_count) : (ri += 1) {
        const rule = &rule_store.rules[@intCast(ri)];
        if (!rule.alive) continue;
        const rsize: i32 = @intCast(@sizeOf(VlpRule));
        if (pos + rsize > cap) return null;
        const dst = output[@intCast(pos)..@intCast(pos + rsize)];
        @memcpy(dst, std.mem.asBytes(rule));
        pos += rsize;
        header.rule_count += 1;
    }

    var ti: i32 = 0;
    while (ti < rule_store.term_count) : (ti += 1) {
        const tsize: i32 = @intCast(@sizeOf(VlpTerm));
        if (pos + tsize > cap) return null;
        const dst = output[@intCast(pos)..@intCast(pos + tsize)];
        @memcpy(dst, std.mem.asBytes(&rule_store.terms[@intCast(ti)]));
        pos += tsize;
    }
    header.term_count = rule_store.term_count;

    var ai: i32 = 0;
    while (ai < rule_store.action_count) : (ai += 1) {
        const asize: i32 = @intCast(@sizeOf(PrologAction));
        if (pos + asize > cap) return null;
        const dst = output[@intCast(pos)..@intCast(pos + asize)];
        @memcpy(dst, std.mem.asBytes(&rule_store.actions[@intCast(ai)]));
        pos += asize;
    }
    header.action_count = rule_store.action_count;

    var gi: i32 = 0;
    while (gi < grant_store.count) : (gi += 1) {
        const g = &grant_store.grants[@intCast(gi)];
        const gsize: i32 = @intCast(@sizeOf(VlpGrant));
        if (pos + gsize > cap) return null;
        const dst = output[@intCast(pos)..@intCast(pos + gsize)];
        @memcpy(dst, std.mem.asBytes(g));
        pos += gsize;
        header.grant_count += 1;
    }

    const text_len = kb_store.text.len;
    if (text_len > 0) {
        if (pos + text_len > cap) return null;
        @memcpy(output[@intCast(pos)..@intCast(pos + text_len)], kb_store.text.data[0..@intCast(text_len)]);
        pos += text_len;
    }
    header.text_len = text_len;

    header.total_size = pos;
    header.checksum = crc32(output[@intCast(header_size)..@intCast(pos)]);

    const hdr_dst = output[0..@intCast(header_size)];
    @memcpy(hdr_dst, std.mem.asBytes(&header));

    return Snapshot{
        .header = header,
        .data = output[0..@intCast(pos)],
        .data_len = pos,
    };
}

pub fn restore(
    snapshot_data: []const u8,
    session: *VlpSession,
    kb_store: *KBStore,
    rule_store: *RuleStore,
    grant_store: *GrantStore,
) VlpStatus {
    const header_size: i32 = @intCast(@sizeOf(SnapshotHeader));
    if (@as(i32, @intCast(snapshot_data.len)) < header_size) return .err_snapshot_corrupt;

    const header: *const SnapshotHeader = @ptrCast(@alignCast(snapshot_data[0..@intCast(header_size)]));

    if (!std.mem.eql(u8, &header.magic, &SNAPSHOT_MAGIC)) return .err_snapshot_corrupt;
    if (header.version != SNAPSHOT_VERSION) return .err_snapshot_corrupt;

    const data_start: usize = @intCast(header_size);
    const data_end: usize = @intCast(header.total_size);
    if (data_end > snapshot_data.len) return .err_snapshot_corrupt;

    const computed = crc32(snapshot_data[data_start..data_end]);
    if (computed != header.checksum) return .err_snapshot_corrupt;

    session.* = header.session;

    var pos: i32 = header_size;

    var ki: i32 = 0;
    while (ki < header.kb_count) : (ki += 1) {
        const ksize: i32 = @intCast(@sizeOf(VlpKB));
        if (pos + ksize > header.total_size) return .err_snapshot_corrupt;
        const kb_bytes = snapshot_data[@intCast(pos)..@intCast(pos + ksize)];
        const kb_ptr: *const VlpKB = @ptrCast(@alignCast(kb_bytes));
        if (kb_ptr.id >= 0 and kb_ptr.id < kb_store.kb_count) {
            kb_store.kbs[@intCast(kb_ptr.id)] = kb_ptr.*;
        }
        pos += ksize;
    }

    var fi: i32 = 0;
    while (fi < header.fact_count) : (fi += 1) {
        const fsize: i32 = @intCast(@sizeOf(VlpFact));
        if (pos + fsize > header.total_size) return .err_snapshot_corrupt;
        pos += fsize;
    }

    rule_store.rule_count = 0;
    var ri: i32 = 0;
    while (ri < header.rule_count) : (ri += 1) {
        const rsize: i32 = @intCast(@sizeOf(VlpRule));
        if (pos + rsize > header.total_size) return .err_snapshot_corrupt;
        const rule_bytes = snapshot_data[@intCast(pos)..@intCast(pos + rsize)];
        const rule_ptr: *const VlpRule = @ptrCast(@alignCast(rule_bytes));
        if (rule_store.rule_count < @as(i32, @intCast(rule_store.rules.len))) {
            rule_store.rules[@intCast(rule_store.rule_count)] = rule_ptr.*;
            rule_store.rule_count += 1;
        }
        pos += rsize;
    }

    rule_store.term_count = 0;
    var ti: i32 = 0;
    while (ti < header.term_count) : (ti += 1) {
        const tsize: i32 = @intCast(@sizeOf(VlpTerm));
        if (pos + tsize > header.total_size) return .err_snapshot_corrupt;
        const term_bytes = snapshot_data[@intCast(pos)..@intCast(pos + tsize)];
        const term_ptr: *const VlpTerm = @ptrCast(@alignCast(term_bytes));
        if (rule_store.term_count < @as(i32, @intCast(rule_store.terms.len))) {
            rule_store.terms[@intCast(rule_store.term_count)] = term_ptr.*;
            rule_store.term_count += 1;
        }
        pos += tsize;
    }

    rule_store.action_count = 0;
    var ai: i32 = 0;
    while (ai < header.action_count) : (ai += 1) {
        const asize: i32 = @intCast(@sizeOf(PrologAction));
        if (pos + asize > header.total_size) return .err_snapshot_corrupt;
        const act_bytes = snapshot_data[@intCast(pos)..@intCast(pos + asize)];
        const act_ptr: *const PrologAction = @ptrCast(@alignCast(act_bytes));
        if (rule_store.action_count < @as(i32, @intCast(rule_store.actions.len))) {
            rule_store.actions[@intCast(rule_store.action_count)] = act_ptr.*;
            rule_store.action_count += 1;
        }
        pos += asize;
    }

    grant_store.count = 0;
    var gi: i32 = 0;
    while (gi < header.grant_count) : (gi += 1) {
        const gsize: i32 = @intCast(@sizeOf(VlpGrant));
        if (pos + gsize > header.total_size) return .err_snapshot_corrupt;
        const grant_bytes = snapshot_data[@intCast(pos)..@intCast(pos + gsize)];
        const grant_ptr: *const VlpGrant = @ptrCast(@alignCast(grant_bytes));
        if (grant_store.count < @as(i32, @intCast(grant_store.grants.len))) {
            grant_store.grants[@intCast(grant_store.count)] = grant_ptr.*;
            grant_store.count += 1;
        }
        pos += gsize;
    }

    if (header.text_len > 0) {
        if (pos + header.text_len > header.total_size) return .err_snapshot_corrupt;
        const src = snapshot_data[@intCast(pos)..@intCast(pos + header.text_len)];
        @memcpy(kb_store.text.data[0..@intCast(header.text_len)], src);
        kb_store.text.len = header.text_len;
        pos += header.text_len;
    }

    return .ok;
}

pub fn diff(a: []const u8, b: []const u8) SnapshotDiff {
    var result = SnapshotDiff{};
    const header_size: i32 = @intCast(@sizeOf(SnapshotHeader));

    if (a.len < @as(usize, @intCast(header_size)) or b.len < @as(usize, @intCast(header_size))) {
        result.identical = false;
        return result;
    }

    if (a.len != b.len) {
        result.identical = false;
        result.size_differs = true;
        return result;
    }

    result.identical = true;
    for (a, 0..) |byte, i| {
        if (byte != b[i]) {
            result.identical = false;
            result.first_diff_offset = @intCast(i);
            break;
        }
    }

    return result;
}

pub const SnapshotDiff = struct {
    identical: bool = true,
    size_differs: bool = false,
    first_diff_offset: i32 = -1,
};

pub fn crc32(data: []const u8) i32 {
    var crc: u32 = 0xFFFFFFFF;
    for (data) |byte| {
        crc ^= @as(u32, byte);
        var bit: u32 = 0;
        while (bit < 8) : (bit += 1) {
            if (crc & 1 == 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc = crc >> 1;
            }
        }
    }
    return @bitCast(crc ^ 0xFFFFFFFF);
}
