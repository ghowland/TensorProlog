// ============================================================
// src/gpu/profiling.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;

pub const KernelStats = struct {
    kernel_id: i32,
    elapsed_ns: i64,
    integer_ops: i64,
    memory_bytes_read: i64,
    memory_bytes_written: i64,
    warp_occupancy_percent: i32,
    kb_cache_hits: i64,
    kb_cache_misses: i64,
    remainder_overflows: i64,
};

pub const SessionStats = struct {
    total_turns: i64,
    total_facts_asserted: i64,
    total_facts_retracted: i64,
    total_rules_fired: i64,
    total_prolog_queries: i64,
    total_kb_accesses: i64,
    total_primitive_calls: i64,
    total_grammar_renders: i64,
    total_llm_tokens: i64,
    total_command_tokens: i64,
    l1_count: i64,
    l2_count: i64,
    l3_count: i64,
    auto_triage_num: i64,
    auto_triage_den: i64,
};

pub const Profiler = struct {
    active: bool,
    start_ns: i64,
    kernel_stats: [64]KernelStats,
    kernel_count: i32,
    session_stats: SessionStats,

    pub fn init() Profiler {
        return .{
            .active = false,
            .start_ns = 0,
            .kernel_stats = undefined,
            .kernel_count = 0,
            .session_stats = std.mem.zeroes(SessionStats),
        };
    }

    pub fn start(self: *Profiler) void {
        self.active = true;
        self.start_ns = std.time.nanoTimestamp();
        self.kernel_count = 0;
    }

    pub fn stop(self: *Profiler) void {
        self.active = false;
    }

    pub fn recordKernel(self: *Profiler, kernel_id: i32, elapsed_ns: i64, int_ops: i64, bytes_read: i64, bytes_written: i64) void {
        if (!self.active) return;
        if (self.kernel_count >= 64) return;
        const idx: usize = @intCast(self.kernel_count);
        self.kernel_stats[idx] = .{
            .kernel_id = kernel_id,
            .elapsed_ns = elapsed_ns,
            .integer_ops = int_ops,
            .memory_bytes_read = bytes_read,
            .memory_bytes_written = bytes_written,
            .warp_occupancy_percent = 100,
            .kb_cache_hits = 0,
            .kb_cache_misses = 0,
            .remainder_overflows = 0,
        };
        self.kernel_count += 1;
    }

    pub fn getKernelStats(self: *const Profiler, kernel_id: i32) ?KernelStats {
        const kc: usize = @intCast(self.kernel_count);
        for (0..kc) |i| {
            if (self.kernel_stats[i].kernel_id == kernel_id) return self.kernel_stats[i];
        }
        return null;
    }

    pub fn totalElapsedNs(self: *const Profiler) i64 {
        if (self.start_ns == 0) return 0;
        return std.time.nanoTimestamp() - self.start_ns;
    }
};
