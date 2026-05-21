// ============================================================
// src/deploy/monitoring.zig
// ============================================================

const server_types_mon = @import("../server/types.zig");
const health_mod_mon = @import("../server/health.zig");
const runner_types_mon = @import("../runner/types.zig");
const pool_mod_mon = @import("../runner/pool.zig");

pub const MonitoringConfig = struct {
    export_interval_ms: i32,
    prometheus_prefix: [32]u8,
    prometheus_prefix_len: i32,
    include_runner_details: bool,
    include_kb_details: bool,
};

pub fn defaultMonitoringConfig() MonitoringConfig {
    var cfg = MonitoringConfig{
        .export_interval_ms = 15000,
        .prometheus_prefix = undefined,
        .prometheus_prefix_len = 0,
        .include_runner_details = true,
        .include_kb_details = false,
    };
    const prefix = "tensorprolog_";
    @memcpy(cfg.prometheus_prefix[0..prefix.len], prefix);
    cfg.prometheus_prefix_len = @intCast(prefix.len);
    return cfg;
}

pub fn exportPrometheus(server: *const server_types_mon.Server, config: *const MonitoringConfig, output: []u8) i32 {
    const report = health_mod_mon.collectHealth(server);
    var pos: usize = 0;
    const prefix = config.prometheus_prefix[0..@intCast(config.prometheus_prefix_len)];

    pos += writeMetric(output[pos..], prefix, "active_connections", report.active_connections);
    pos += writeMetric(output[pos..], prefix, "total_accepted", @intCast(report.total_accepted));
    pos += writeMetric(output[pos..], prefix, "total_rejected", @intCast(report.total_rejected));
    pos += writeMetric(output[pos..], prefix, "total_requests", @intCast(report.total_requests));
    pos += writeMetric(output[pos..], prefix, "active_sessions", report.active_sessions);
    pos += writeMetric(output[pos..], prefix, "total_facts", report.total_facts);
    pos += writeMetric(output[pos..], prefix, "total_rules", report.total_rules);
    pos += writeMetric(output[pos..], prefix, "l1_count", @intCast(report.l1_count));
    pos += writeMetric(output[pos..], prefix, "l2_count", @intCast(report.l2_count));
    pos += writeMetric(output[pos..], prefix, "l3_count", @intCast(report.l3_count));

    const total = report.l1_count + report.l2_count + report.l3_count;
    if (total > 0) {
        pos += writeMetric(output[pos..], prefix, "auto_triage_numerator", @intCast(report.l3_count));
        pos += writeMetric(output[pos..], prefix, "auto_triage_denominator", @intCast(total));
    }

    if (config.include_runner_details) {
        const rc: usize = @intCast(report.runner_count);
        for (0..rc) |i| {
            pos += writeMetricLabeled(output[pos..], prefix, "runner_iterations", report.runner_states[i].id, @intCast(report.runner_states[i].iterations));
            pos += writeMetricLabeled(output[pos..], prefix, "runner_errors", report.runner_states[i].id, @intCast(report.runner_states[i].errors_total));
            pos += writeMetricLabeled(output[pos..], prefix, "runner_recycles", report.runner_states[i].id, report.runner_states[i].recycle_count);
        }
    }

    return @intCast(pos);
}

pub fn exportHealthJson(server: *const server_types_mon.Server, output: []u8) i32 {
    const report = health_mod_mon.collectHealth(server);
    return health_mod_mon.renderHealthJson(&report, output);
}

fn writeMetric(output: []u8, prefix: []const u8, name: []const u8, value: i32) usize {
    var pos: usize = 0;
    pos += copyStrM(output[pos..], prefix);
    pos += copyStrM(output[pos..], name);
    pos += copyStrM(output[pos..], " ");
    pos += i32ToAsciiM(value, output[pos..]);
    pos += copyStrM(output[pos..], "\n");
    return pos;
}

fn writeMetricLabeled(output: []u8, prefix: []const u8, name: []const u8, label_val: i32, value: i32) usize {
    var pos: usize = 0;
    pos += copyStrM(output[pos..], prefix);
    pos += copyStrM(output[pos..], name);
    pos += copyStrM(output[pos..], "{id=\"");
    pos += i32ToAsciiM(label_val, output[pos..]);
    pos += copyStrM(output[pos..], "\"} ");
    pos += i32ToAsciiM(value, output[pos..]);
    pos += copyStrM(output[pos..], "\n");
    return pos;
}

fn copyStrM(dest: []u8, src: []const u8) usize {
    const n = @min(src.len, dest.len);
    @memcpy(dest[0..n], src[0..n]);
    return n;
}

fn i32ToAsciiM(val: i32, output: []u8) usize {
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
