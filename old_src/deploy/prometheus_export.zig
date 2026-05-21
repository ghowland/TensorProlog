// ============================================================
// src/deploy/prometheus_export.zig
// ============================================================

const health_mod = @import("../server/health.zig");
const server_types = @import("../server/types.zig");

pub fn exportPrometheus(server: *const server_types.Server, output: []u8) i32 {
    const report = health_mod.collectHealth(server);
    var pos: usize = 0;

    pos += writeMetric(output[pos..], "tensorprolog_active_connections", report.active_connections);
    pos += writeMetric(output[pos..], "tensorprolog_total_accepted", @intCast(report.total_accepted));
    pos += writeMetric(output[pos..], "tensorprolog_total_rejected", @intCast(report.total_rejected));
    pos += writeMetric(output[pos..], "tensorprolog_total_requests", @intCast(report.total_requests));
    pos += writeMetric(output[pos..], "tensorprolog_active_sessions", report.active_sessions);
    pos += writeMetric(output[pos..], "tensorprolog_total_facts", report.total_facts);
    pos += writeMetric(output[pos..], "tensorprolog_total_rules", report.total_rules);
    pos += writeMetric(output[pos..], "tensorprolog_l1_count", @intCast(report.l1_count));
    pos += writeMetric(output[pos..], "tensorprolog_l2_count", @intCast(report.l2_count));
    pos += writeMetric(output[pos..], "tensorprolog_l3_count", @intCast(report.l3_count));

    const total = report.l1_count + report.l2_count + report.l3_count;
    if (total > 0) {
        pos += writeMetricFrac(output[pos..], "tensorprolog_auto_triage_numerator", @intCast(report.l3_count));
        pos += writeMetricFrac(output[pos..], "tensorprolog_auto_triage_denominator", @intCast(total));
    }

    return @intCast(pos);
}

fn writeMetric(output: []u8, name: []const u8, value: i32) usize {
    var pos: usize = 0;
    pos += copyStr(output[pos..], name);
    pos += copyStr(output[pos..], " ");
    pos += i32ToAscii(value, output[pos..]);
    pos += copyStr(output[pos..], "\n");
    return pos;
}

fn writeMetricFrac(output: []u8, name: []const u8, value: i32) usize {
    return writeMetric(output, name, value);
}
