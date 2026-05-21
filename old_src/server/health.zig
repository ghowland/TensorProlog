// ============================================================
// src/server/health.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");
const server_types = @import("types.zig");
const runner_types = @import("../runner/types.zig");
const runner_pool = @import("../runner/pool.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;
const Server = server_types.Server;

pub const HealthReport = struct {
    active_connections: i32,
    total_accepted: i64,
    total_rejected: i64,
    total_requests: i64,
    active_sessions: i32,
    total_facts: i32,
    total_rules: i32,
    l1_count: i64,
    l2_count: i64,
    l3_count: i64,
    runner_count: i32,
    runner_states: [16]RunnerHealth,
};

pub const RunnerHealth = struct {
    id: i32,
    runner_type: runner_types.VlpRunnerType,
    state: runner_types.VlpRunnerState,
    iterations: i64,
    errors_consecutive: i32,
    errors_total: i64,
    recycle_count: i32,
};

pub fn collectHealth(server: *const Server) HealthReport {
    var report = HealthReport{
        .active_connections = server.n_active.load(.seq_cst),
        .total_accepted = server.metrics.total_connections_accepted,
        .total_rejected = server.metrics.total_connections_rejected,
        .total_requests = server.metrics.total_requests_served,
        .active_sessions = server.metrics.active_sessions,
        .total_facts = 0,
        .total_rules = 0,
        .l1_count = 0,
        .l2_count = 0,
        .l3_count = 0,
        .runner_count = 0,
        .runner_states = undefined,
    };

    const kb_count = server.store.count();
    _ = kb_count;

    return report;
}

pub fn renderHealthJson(report: *const HealthReport, output: []u8) i32 {
    var pos: usize = 0;

    pos += copyStr(output[pos..], "{\"status\":\"ok\"");
    pos += copyStr(output[pos..], ",\"active_connections\":");
    pos += i32ToAscii(report.active_connections, output[pos..]);
    pos += copyStr(output[pos..], ",\"total_accepted\":");
    pos += i64ToAscii(report.total_accepted, output[pos..]);
    pos += copyStr(output[pos..], ",\"total_rejected\":");
    pos += i64ToAscii(report.total_rejected, output[pos..]);
    pos += copyStr(output[pos..], ",\"total_requests\":");
    pos += i64ToAscii(report.total_requests, output[pos..]);
    pos += copyStr(output[pos..], ",\"active_sessions\":");
    pos += i32ToAscii(report.active_sessions, output[pos..]);
    pos += copyStr(output[pos..], ",\"total_facts\":");
    pos += i32ToAscii(report.total_facts, output[pos..]);
    pos += copyStr(output[pos..], ",\"total_rules\":");
    pos += i32ToAscii(report.total_rules, output[pos..]);

    pos += copyStr(output[pos..], ",\"levels\":{\"l1\":");
    pos += i64ToAscii(report.l1_count, output[pos..]);
    pos += copyStr(output[pos..], ",\"l2\":");
    pos += i64ToAscii(report.l2_count, output[pos..]);
    pos += copyStr(output[pos..], ",\"l3\":");
    pos += i64ToAscii(report.l3_count, output[pos..]);
    pos += copyStr(output[pos..], "}");

    const total_levels = report.l1_count + report.l2_count + report.l3_count;
    if (total_levels > 0) {
        pos += copyStr(output[pos..], ",\"auto_triage\":\"");
        pos += i64ToAscii(report.l3_count, output[pos..]);
        pos += copyStr(output[pos..], "/");
        pos += i64ToAscii(total_levels, output[pos..]);
        pos += copyStr(output[pos..], "\"");
    }

    pos += copyStr(output[pos..], ",\"runners\":[");
    const rc: usize = @intCast(report.runner_count);
    for (0..rc) |i| {
        if (i > 0) {
            pos += copyStr(output[pos..], ",");
        }
        pos += copyStr(output[pos..], "{\"id\":");
        pos += i32ToAscii(report.runner_states[i].id, output[pos..]);
        pos += copyStr(output[pos..], ",\"state\":");
        pos += i32ToAscii(@intFromEnum(report.runner_states[i].state), output[pos..]);
        pos += copyStr(output[pos..], ",\"iterations\":");
        pos += i64ToAscii(report.runner_states[i].iterations, output[pos..]);
        pos += copyStr(output[pos..], ",\"errors\":");
        pos += i32ToAscii(report.runner_states[i].errors_consecutive, output[pos..]);
        pos += copyStr(output[pos..], ",\"recycles\":");
        pos += i32ToAscii(report.runner_states[i].recycle_count, output[pos..]);
        pos += copyStr(output[pos..], "}");
    }
    pos += copyStr(output[pos..], "]");

    pos += copyStr(output[pos..], "}");

    return @intCast(pos);
}

fn copyStr(dest: []u8, src: []const u8) usize {
    const n = @min(src.len, dest.len);
    @memcpy(dest[0..n], src[0..n]);
    return n;
}

fn i32ToAscii(val: i32, output: []u8) usize {
    if (output.len == 0) return 0;
    if (val == 0) {
        output[0] = '0';
        return 1;
    }
    var v: i64 = @intCast(val);
    var pos: usize = 0;
    if (v < 0) {
        output[pos] = '-';
        pos += 1;
        v = -v;
    }
    var buf: [12]u8 = undefined;
    var len: usize = 0;
    while (v > 0) {
        buf[len] = @intCast(@as(u8, @intCast(@mod(v, 10))) + '0');
        len += 1;
        v = @divTrunc(v, 10);
    }
    for (0..len) |i| {
        if (pos >= output.len) break;
        output[pos] = buf[len - 1 - i];
        pos += 1;
    }
    return pos;
}

fn i64ToAscii(val: i64, output: []u8) usize {
    if (output.len == 0) return 0;
    if (val == 0) {
        output[0] = '0';
        return 1;
    }
    var v = val;
    var pos: usize = 0;
    if (v < 0) {
        output[pos] = '-';
        pos += 1;
        v = -v;
    }
    var buf: [20]u8 = undefined;
    var len: usize = 0;
    while (v > 0) {
        buf[len] = @intCast(@as(u8, @intCast(@mod(v, 10))) + '0');
        len += 1;
        v = @divTrunc(v, 10);
    }
    for (0..len) |i| {
        if (pos >= output.len) break;
        output[pos] = buf[len - 1 - i];
        pos += 1;
    }
    return pos;
}
