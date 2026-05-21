// ============================================================
// src/kb/store.zig
// ============================================================

const kb_types = @import("types.zig");
const text_mod = @import("text_store.zig");
const path_mod = @import("path_index.zig");

const VlpKB = kb_types.VlpKB;
const VlpFact = kb_types.VlpFact;
const VlpVisibility = kb_types.VlpVisibility;
const KBCreateConfig = kb_types.KBCreateConfig;
const TextStore = text_mod.TextStore;
const TextRef = text_mod.TextRef;
const PathIndex = path_mod.PathIndex;

pub const KBStore = struct {
    kbs: []VlpKB,
    kb_count: i32,

    facts: []VlpFact,
    fact_next: i32,

    children: []i32,
    children_next: i32,

    text: TextStore,
    paths: PathIndex,

    pub fn init(
        kbs: []VlpKB,
        facts: []VlpFact,
        text_buf: []u8,
        pi_keys: []i32,
        pi_vals: []i32,
        pi_occ: []bool,
        children: []i32,
    ) KBStore {
        for (kbs) |*k| k.* = VlpKB{};
        for (facts) |*f| f.* = VlpFact{};
        @memset(children, -1);
        return .{
            .kbs = kbs,
            .kb_count = 0,
            .facts = facts,
            .fact_next = 0,
            .children = children,
            .children_next = 0,
            .text = TextStore.init(text_buf),
            .paths = PathIndex.init(pi_keys, pi_vals, pi_occ),
        };
    }

    pub fn createKB(self: *KBStore, cfg: KBCreateConfig) ?i32 {
        if (self.kb_count >= @as(i32, @intCast(self.kbs.len))) return null;
        if (self.fact_next + cfg.max_facts > @as(i32, @intCast(self.facts.len))) return null;
        if (self.children_next + cfg.max_children > @as(i32, @intCast(self.children.len))) return null;

        const id = self.kb_count;
        var kb = &self.kbs[@intCast(id)];

        const name_ref = self.text.append(cfg.name) orelse return null;
        kb.name_offset = name_ref.offset;
        kb.name_length = name_ref.length;

        var path_buf: [1024]u8 = undefined;
        const plen = self.buildPath(cfg.parent_id, cfg.name, &path_buf);
        const path_ref = self.text.append(path_buf[0..plen]) orelse return null;
        kb.path_offset = path_ref.offset;
        kb.path_length = path_ref.length;

        if (!self.paths.insert(path_buf[0..plen], id)) return null;

        kb.id = id;
        kb.facts_offset = self.fact_next;
        kb.facts_capacity = cfg.max_facts;
        self.fact_next += cfg.max_facts;

        kb.children_offset = self.children_next;
        kb.children_capacity = @intCast(cfg.max_children);
        self.children_next += cfg.max_children;

        kb.parent_id = cfg.parent_id;
        kb.visibility = cfg.visibility;
        kb.owner_id = cfg.owner_id;
        kb.alive = true;

        self.kb_count += 1;

        if (cfg.parent_id >= 0) self.addChild(cfg.parent_id, id);

        return id;
    }

    pub fn destroyKB(self: *KBStore, kb_id: i32) bool {
        const kb = self.getKB(kb_id) orelse return false;
        const kids = self.childrenSlice(kb_id);
        for (kids) |cid| {
            if (cid >= 0 and cid < self.kb_count and self.kbs[@intCast(cid)].alive) {
                self.kbs[@intCast(cid)].parent_id = kb.parent_id;
                if (kb.parent_id >= 0) self.addChild(kb.parent_id, cid);
            }
        }
        if (kb.parent_id >= 0) self.removeChild(kb.parent_id, kb_id);
        const s: usize = @intCast(kb.facts_offset);
        const e: usize = @intCast(kb.facts_offset + kb.facts_capacity);
        for (self.facts[s..e]) |*f| f.* = VlpFact{};
        var kbm = &self.kbs[@intCast(kb_id)];
        kbm.alive = false;
        return true;
    }

    pub fn getKB(self: *KBStore, kb_id: i32) ?*VlpKB {
        if (kb_id < 0 or kb_id >= self.kb_count) return null;
        const kb = &self.kbs[@intCast(kb_id)];
        if (!kb.alive) return null;
        return kb;
    }

    pub fn getKBConst(self: *const KBStore, kb_id: i32) ?*const VlpKB {
        if (kb_id < 0 or kb_id >= self.kb_count) return null;
        const kb = &self.kbs[@intCast(kb_id)];
        if (!kb.alive) return null;
        return kb;
    }

    pub fn resolvePath(self: *const KBStore, path: []const u8) ?i32 {
        const id = self.paths.lookup(path) orelse return null;
        if (id < 0 or id >= self.kb_count) return null;
        if (!self.kbs[@intCast(id)].alive) return null;
        return id;
    }

    pub fn getName(self: *const KBStore, kb_id: i32) ?[]const u8 {
        const kb = self.getKBConst(kb_id) orelse return null;
        return self.text.read(.{ .offset = kb.name_offset, .length = kb.name_length });
    }

    pub fn getPath(self: *const KBStore, kb_id: i32) ?[]const u8 {
        const kb = self.getKBConst(kb_id) orelse return null;
        return self.text.read(.{ .offset = kb.path_offset, .length = kb.path_length });
    }

    pub fn childrenSlice(self: *const KBStore, kb_id: i32) []const i32 {
        const kb = self.getKBConst(kb_id) orelse return &[_]i32{};
        if (kb.children_count <= 0) return &[_]i32{};
        const s: usize = @intCast(kb.children_offset);
        const e: usize = @intCast(kb.children_offset + @as(i32, kb.children_count));
        return self.children[s..e];
    }

    pub fn aliveCount(self: *const KBStore) i32 {
        var c: i32 = 0;
        for (self.kbs[0..@intCast(self.kb_count)]) |*kb| {
            if (kb.alive) c += 1;
        }
        return c;
    }

    fn addChild(self: *KBStore, parent_id: i32, child_id: i32) void {
        if (parent_id < 0 or parent_id >= self.kb_count) return;
        var p = &self.kbs[@intCast(parent_id)];
        if (p.children_count >= p.children_capacity) return;
        const idx: usize = @intCast(p.children_offset + @as(i32, p.children_count));
        self.children[idx] = child_id;
        p.children_count += 1;
    }

    fn removeChild(self: *KBStore, parent_id: i32, child_id: i32) void {
        if (parent_id < 0 or parent_id >= self.kb_count) return;
        var p = &self.kbs[@intCast(parent_id)];
        const s: usize = @intCast(p.children_offset);
        const n: usize = @intCast(p.children_count);
        for (s..s + n) |i| {
            if (self.children[i] == child_id) {
                self.children[i] = self.children[s + n - 1];
                self.children[s + n - 1] = -1;
                p.children_count -= 1;
                return;
            }
        }
    }

    fn buildPath(self: *const KBStore, parent_id: i32, name: []const u8, buf: []u8) usize {
        if (parent_id < 0) {
            @memcpy(buf[0..name.len], name);
            return name.len;
        }
        const pp = self.getPath(parent_id) orelse {
            @memcpy(buf[0..name.len], name);
            return name.len;
        };
        const total = pp.len + 1 + name.len;
        if (total > buf.len) {
            @memcpy(buf[0..name.len], name);
            return name.len;
        }
        @memcpy(buf[0..pp.len], pp);
        buf[pp.len] = '.';
        @memcpy(buf[pp.len + 1 .. pp.len + 1 + name.len], name);
        return total;
    }
};
