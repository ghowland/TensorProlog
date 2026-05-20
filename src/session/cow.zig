// ============================================================
// src/session/cow.zig
// ============================================================

pub const PAGE_SIZE: i32 = 4096;

pub const COWPage = struct {
    source: [*]u8,
    private: ?[*]u8,
    dirty: bool = false,
    private_backing: ?[]u8 = null,
};

pub const COWPageTable = struct {
    pages: []COWPage,
    n_pages: i32,
    private_pool: []u8,
    private_next: i32,

    pub fn init(pages: []COWPage, source_base: [*]u8, n_pages: i32, private_pool: []u8) COWPageTable {
        var i: i32 = 0;
        while (i < n_pages) : (i += 1) {
            pages[@intCast(i)] = .{
                .source = source_base + @as(usize, @intCast(i * PAGE_SIZE)),
                .private = null,
                .dirty = false,
            };
        }
        return .{
            .pages = pages,
            .n_pages = n_pages,
            .private_pool = private_pool,
            .private_next = 0,
        };
    }

    pub fn readPage(self: *const COWPageTable, page_id: i32) ?[*]const u8 {
        if (page_id < 0 or page_id >= self.n_pages) return null;
        const p = &self.pages[@intCast(page_id)];
        if (p.dirty) {
            if (p.private) |priv| return priv;
        }
        return p.source;
    }

    pub fn writePage(self: *COWPageTable, page_id: i32) ?[*]u8 {
        if (page_id < 0 or page_id >= self.n_pages) return null;
        var p = &self.pages[@intCast(page_id)];
        if (p.dirty) {
            return p.private;
        }
        const priv = self.allocPrivate() orelse return null;
        const src_slice = p.source[0..@intCast(PAGE_SIZE)];
        @memcpy(priv[0..@intCast(PAGE_SIZE)], src_slice);
        p.private = priv;
        p.dirty = true;
        return priv;
    }

    pub fn isDirty(self: *const COWPageTable, page_id: i32) bool {
        if (page_id < 0 or page_id >= self.n_pages) return false;
        return self.pages[@intCast(page_id)].dirty;
    }

    pub fn dirtyCount(self: *const COWPageTable) i32 {
        var c: i32 = 0;
        var i: i32 = 0;
        while (i < self.n_pages) : (i += 1) {
            if (self.pages[@intCast(i)].dirty) c += 1;
        }
        return c;
    }

    pub fn dirtyPages(self: *const COWPageTable, out: []i32) i32 {
        var c: i32 = 0;
        var i: i32 = 0;
        while (i < self.n_pages) : (i += 1) {
            if (c >= @as(i32, @intCast(out.len))) break;
            if (self.pages[@intCast(i)].dirty) {
                out[@intCast(c)] = i;
                c += 1;
            }
        }
        return c;
    }

    pub fn resolve(self: *COWPageTable) void {
        var i: i32 = 0;
        while (i < self.n_pages) : (i += 1) {
            var p = &self.pages[@intCast(i)];
            if (!p.dirty) {
                const priv = self.allocPrivate() orelse continue;
                const src_slice = p.source[0..@intCast(PAGE_SIZE)];
                @memcpy(priv[0..@intCast(PAGE_SIZE)], src_slice);
                p.private = priv;
                p.dirty = true;
            }
        }
    }

    pub fn applyToSource(self: *COWPageTable) void {
        var i: i32 = 0;
        while (i < self.n_pages) : (i += 1) {
            const p = &self.pages[@intCast(i)];
            if (p.dirty) {
                if (p.private) |priv| {
                    const dst = p.source[0..@intCast(PAGE_SIZE)];
                    @memcpy(dst, priv[0..@intCast(PAGE_SIZE)]);
                }
            }
        }
    }

    fn allocPrivate(self: *COWPageTable) ?[*]u8 {
        const needed = PAGE_SIZE;
        if (self.private_next + needed > @as(i32, @intCast(self.private_pool.len))) return null;
        const offset: usize = @intCast(self.private_next);
        self.private_next += needed;
        return self.private_pool[offset..].ptr;
    }
};

pub fn pageCount(byte_size: i32) i32 {
    return @divTrunc(byte_size + PAGE_SIZE - 1, PAGE_SIZE);
}

pub fn pageForOffset(byte_offset: i32) i32 {
    return @divTrunc(byte_offset, PAGE_SIZE);
}

pub fn offsetInPage(byte_offset: i32) i32 {
    return @mod(byte_offset, PAGE_SIZE);
}
