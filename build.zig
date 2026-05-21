// ============================================================
// build.zig
// Two artifacts: PTX kernel (nvptx64) + host binary (native).
// PTX embedded into host binary via @embedFile.
// ============================================================

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ============================================================
    // Step 1: Build PTX kernel
    // ============================================================

    const ptx_target = b.resolveTargetQuery(.{
        .cpu_arch = .nvptx64,
        .os_tag = .freestanding,
        .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_75 },
    });

    const kernel = b.addObject(.{
        .name = "vlp_kernel",
        .root_source_file = b.path("src/vlp_kernel.zig"),
        .target = ptx_target,
        .optimize = .Debug, // ReleaseFast strips dead code; Debug keeps overflow checks
    });

    // The kernel imports vlp_gpu_shared for constants
    kernel.addImport("vlp_gpu_shared", b.addModule(.{
        .name = "vlp_gpu_shared_kernel",
        .root_source_file = b.path("src/vlp_gpu_shared.zig"),
        .target = ptx_target,
        .optimize = .Debug,
    }));

    // Emit PTX as a file we can embed
    const ptx_emit = kernel.getEmittedBin(); // .ptx file

    // ============================================================
    // Step 2: Build host binary
    // ============================================================

    // Shared module — compiled for host
    const gpu_shared_mod = b.addModule(.{
        .name = "vlp_gpu_shared",
        .root_source_file = b.path("src/vlp_gpu_shared.zig"),
    });

    const gpu_params_mod = b.addModule(.{
        .name = "vlp_gpu_params",
        .root_source_file = b.path("src/vlp_gpu_params.zig"),
    });
    gpu_params_mod.addImport("vlp_types", b.addModule(.{
        .name = "vlp_types_for_params",
        .root_source_file = b.path("src/vlp_types.zig"),
    }));

    const types_mod = b.addModule(.{
        .name = "vlp_types",
        .root_source_file = b.path("src/vlp_types.zig"),
    });
    types_mod.addImport("vlp_gpu_shared", gpu_shared_mod);

    const device_mem_mod = b.addModule(.{
        .name = "vlp_device_memory",
        .root_source_file = b.path("src/vlp_device_memory.zig"),
    });
    device_mem_mod.addImport("vlp_types", types_mod);

    const exe = b.addExecutable(.{
        .name = "vlp",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Embed PTX into host binary — bridge reads this via @embedFile
    exe.addAnonymousImport("vlp_kernel_ptx", .{
        .root_source_file = ptx_emit,
    });

    // Add all modules to the executable
    exe.root_module.addImport("vlp_gpu_shared", gpu_shared_mod);
    exe.root_module.addImport("vlp_gpu_params", gpu_params_mod);
    exe.root_module.addImport("vlp_types", types_mod);
    exe.root_module.addImport("vlp_device_memory", device_mem_mod);

    // Link CUDA driver library
    exe.linkSystemLibrary("cuda");
    exe.linkLibC();

    b.installArtifact(exe);

    // ============================================================
    // Step 3: Tests
    // ============================================================

    const tests = b.addTest(.{
        .root_source_file = b.path("src/vlp_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    tests.root_module.addImport("vlp_gpu_shared", gpu_shared_mod);
    tests.root_module.addImport("vlp_types", types_mod);
    tests.linkSystemLibrary("cuda");
    tests.linkLibC();

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run VLP tests");
    test_step.dependOn(&run_tests.step);
}
