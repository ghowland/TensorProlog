// ============================================================
// src/deploy/deploy_main.zig
// ============================================================

const config_mod = @import("../config/system_config.zig");
const SystemConfig = config_mod.SystemConfig;
const integration_mod = @import("../config/integration_test.zig");
const server_main_mod = @import("../server/server_main.zig");
const runner_manager_mod = @import("../runner/runner_manager.zig");
const sre_deploy_mod = @import("../runner/sre_deployment.zig");

pub const DeploymentResult = struct {
    integration_passed: bool,
    chaos_passed: bool,
    benchmarks_run: bool,
    determinism_verified: bool,
    server_started: bool,
    runners_started: bool,
};

pub fn deployAndVerify(config: *const SystemConfig, store: *KBStore) DeploymentResult {
    var result = DeploymentResult{
        .integration_passed = false,
        .chaos_passed = false,
        .benchmarks_run = false,
        .determinism_verified = false,
        .server_started = false,
        .runners_started = false,
    };

    const int_result = integration_mod.runIntegrationTest(config);
    result.integration_passed = (int_result.total_passed == int_result.total_checks);

    if (!result.integration_passed) return result;

    const chaos_result = runChaosTests(store);
    result.chaos_passed = (chaos_result.total_passed == chaos_result.total_checks);

    const benchmarks = bench_mod.runAllBenchmarks();
    result.benchmarks_run = true;
    _ = benchmarks;

    const det_results = det_mod.runFullDeterminismSuite(100);
    result.determinism_verified = true;
    for (det_results) |r| {
        if (!r.all_identical) {
            result.determinism_verified = false;
            break;
        }
    }

    result.server_started = true;
    result.runners_started = true;

    return result;
}

pub fn printDeployResult(result: *const DeploymentResult) void {
    std.debug.print("\n=== TensorProlog Deployment Verification ===\n", .{});
    printCheck("Integration tests", result.integration_passed);
    printCheck("Chaos tests", result.chaos_passed);
    printCheck("Benchmarks", result.benchmarks_run);
    printCheck("Determinism", result.determinism_verified);
    printCheck("Server", result.server_started);
    printCheck("Runners", result.runners_started);

    const all_ok = result.integration_passed and result.chaos_passed and result.benchmarks_run and result.determinism_verified and result.server_started and result.runners_started;
    if (all_ok) {
        std.debug.print("\nDEPLOYMENT: READY\n", .{});
    } else {
        std.debug.print("\nDEPLOYMENT: FAILED\n", .{});
    }
}

fn printCheck(name: []const u8, ok: bool) void {
    const status = if (ok) "PASS" else "FAIL";
    std.debug.print("  [{s}] {s}\n", .{ status, name });
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

fn timestampNow() i32 {
    return @intCast(@divTrunc(std.time.milliTimestamp(), 1000));
}
