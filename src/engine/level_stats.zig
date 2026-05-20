// ============================================================
// src/engine/level_stats.zig
// ============================================================

const vdr_types = @import("../vdr/types.zig");
const VlpExecutionLevel = vdr_types.VlpExecutionLevel;

pub const LevelStats = struct {
    l1_count: i64 = 0,
    l2_count: i64 = 0,
    l3_count: i64 = 0,
    l1_tokens: i64 = 0,
    l2_tokens: i64 = 0,

    pub fn update(self: *LevelStats, level: VlpExecutionLevel, tokens: i32) void {
        switch (level) {
            .l1 => {
                self.l1_count += 1;
                self.l1_tokens += @as(i64, tokens);
            },
            .l2 => {
                self.l2_count += 1;
                self.l2_tokens += @as(i64, tokens);
            },
            .l3 => self.l3_count += 1,
        }
    }

    pub fn totalCount(self: *const LevelStats) i64 {
        return self.l1_count + self.l2_count + self.l3_count;
    }

    pub fn autoTriageRate(self: *const LevelStats) struct { num: i64, den: i64 } {
        const total = self.totalCount();
        if (total == 0) return .{ .num = 0, .den = 0 };
        return .{ .num = self.l3_count, .den = total };
    }

    pub fn avgTokensPerTurn(self: *const LevelStats) struct { num: i64, den: i64 } {
        const total = self.totalCount();
        if (total == 0) return .{ .num = 0, .den = 0 };
        return .{ .num = self.l1_tokens + self.l2_tokens, .den = total };
    }

    pub fn reset(self: *LevelStats) void {
        self.l1_count = 0;
        self.l2_count = 0;
        self.l3_count = 0;
        self.l1_tokens = 0;
        self.l2_tokens = 0;
    }

    pub fn merge(self: *LevelStats, other: *const LevelStats) void {
        self.l1_count += other.l1_count;
        self.l2_count += other.l2_count;
        self.l3_count += other.l3_count;
        self.l1_tokens += other.l1_tokens;
        self.l2_tokens += other.l2_tokens;
    }
};
