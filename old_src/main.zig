// ============================================================
// src/main.zig
// ============================================================

const std = @import("std");

const cli_mod = @import("config/cli.zig");
const config_file_mod = @import("config/config_file.zig");
const integration_mod = @import("config/integration_test.zig");

pub fn main() !void {
    const argv = std.os.argv;
    var arg_slices: [64][]const u8 = undefined;
    const argc = @min(argv.len, arg_slices.len);
    for (0..argc) |i| {
        arg_slices[i] = std.mem.span(argv[i]);
    }

    var args = cli_mod.parseCli(arg_slices[0..argc]);

    if (args.show_help) {
        cli_mod.printHelp();
        return;
    }

    if (args.show_version) {
        cli_mod.printVersion();
        return;
    }

    if (args.config_file_path_len > 0) {
        const path = args.config_file_path[0..@intCast(args.config_file_path_len)];
        _ = config_file_mod.parseConfigFile(path, &args.config);
    }

    if (args.run_tests) {
        const result = integration_mod.runIntegrationTest(&args.config);
        integration_mod.printIntegrationResult(&result);
        if (result.total_passed < result.total_checks) {
            std.process.exit(1);
        }
        return;
    }

    std.debug.print("TensorProlog v0.1.0\n", .{});
    std.debug.print("Port: {d}\n", .{args.config.server_port});
    std.debug.print("Max connections: {d}\n", .{args.config.server_max_connections});
    std.debug.print("Max runners: {d}\n", .{args.config.max_runners});
    std.debug.print("Max KBs: {d}\n", .{args.config.max_total_kbs});
    std.debug.print("Model: {d} layers, d_model={d}, {d} heads, vocab={d}\n", .{
        args.config.model_n_layers,
        args.config.model_d_model,
        args.config.model_n_heads,
        args.config.model_vocab_size,
    });
    std.debug.print("\nStarting server...\n", .{});
}

// ============================================================
// build.zig
// ============================================================

// const std = @import("std");
//
// pub fn build(b: *std.Build) void {
//     const target = b.standardTargetOptions(.{});
//     const optimize = b.standardOptimizeOption(.{});
//
//     const exe = b.addExecutable(.{
//         .name = "tensorprolog",
//         .root_source_file = b.path("src/main.zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     b.installArtifact(exe);
//
//     const run_cmd = b.addRunArtifact(exe);
//     run_cmd.step.dependOn(b.getInstallStep());
//     if (b.args) |args| run_cmd.addArgs(args);
//     const run_step = b.step("run", "Run TensorProlog");
//     run_step.dependOn(&run_cmd.step);
//
//     const phase1_tests = b.addTest(.{
//         .root_source_file = b.path("test/test_q16.zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     const phase2_tests = b.addTest(.{
//         .root_source_file = b.path("test/test_prolog.zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     const phase3_tests = b.addTest(.{
//         .root_source_file = b.path("test/test_engine.zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     const phase4_tests = b.addTest(.{
//         .root_source_file = b.path("test/test_server.zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//
//     const test_step = b.step("test", "Run all tests");
//     test_step.dependOn(&b.addRunArtifact(phase1_tests).step);
//     test_step.dependOn(&b.addRunArtifact(phase2_tests).step);
//     test_step.dependOn(&b.addRunArtifact(phase3_tests).step);
//     test_step.dependOn(&b.addRunArtifact(phase4_tests).step);
//
//     const test_phase1 = b.step("test-phase1", "Phase 1: Foundation");
//     test_phase1.dependOn(&b.addRunArtifact(phase1_tests).step);
//     const test_phase2 = b.step("test-phase2", "Phase 2: Intelligence");
//     test_phase2.dependOn(&b.addRunArtifact(phase2_tests).step);
//     const test_phase3 = b.step("test-phase3", "Phase 3: Inference + Builtins");
//     test_phase3.dependOn(&b.addRunArtifact(phase3_tests).step);
//     const test_phase4 = b.step("test-phase4", "Phase 4: Operations");
//     test_phase4.dependOn(&b.addRunArtifact(phase4_tests).step);
// }
