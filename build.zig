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

    const gpu_shared_kernel = b.createModule(.{
        .root_source_file = b.path("src/vlp_gpu_shared.zig"),
        .target = ptx_target,
        .optimize = .Debug,
    });

    const kernel_module = b.createModule(.{
        .root_source_file = b.path("src/vlp_kernel.zig"),
        .target = ptx_target,
        .optimize = .Debug,
    });
    kernel_module.addImport("vlp_gpu_shared", gpu_shared_kernel);

    const ptx_kernel = b.addObject(.{
        .name = "vlp_kernel",
        .root_module = kernel_module,
    });

    // Install the PTX assembly output
    const install_ptx = b.addInstallFile(ptx_kernel.getEmittedAsm(), "vlp_kernel.ptx");
    b.getInstallStep().dependOn(&install_ptx.step);

    // ============================================================
    // Step 2: Build host binary
    // ============================================================

    const gpu_shared_host = b.createModule(.{
        .root_source_file = b.path("src/vlp_gpu_shared.zig"),
    });

    const types_mod = b.createModule(.{
        .root_source_file = b.path("src/vlp_types.zig"),
    });
    types_mod.addImport("vlp_gpu_shared", gpu_shared_host);

    const device_mem_mod = b.createModule(.{
        .root_source_file = b.path("src/vlp_device_memory.zig"),
    });
    device_mem_mod.addImport("vlp_types", types_mod);

    const gpu_params_mod = b.createModule(.{
        .root_source_file = b.path("src/vlp_gpu_params.zig"),
    });
    gpu_params_mod.addImport("vlp_types", types_mod);
    gpu_params_mod.addImport("vlp_gpu_shared", gpu_shared_host);

    const host_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    host_module.addImport("vlp_gpu_shared", gpu_shared_host);
    host_module.addImport("vlp_types", types_mod);
    host_module.addImport("vlp_device_memory", device_mem_mod);
    host_module.addImport("vlp_gpu_params", gpu_params_mod);

    const exe = b.addExecutable(.{
        .name = "vlp",
        .root_module = host_module,
    });

    // Embed PTX into host binary — bridge reads via @embedFile
    host_module.addAnonymousImport("vlp_kernel_ptx", .{
        .root_source_file = ptx_kernel.getEmittedAsm(),
    });

    // Link CUDA driver library
    exe.linkSystemLibrary("cuda");
    exe.linkLibC();

    b.installArtifact(exe);

    // ============================================================
    // Step 3: Tests
    // ============================================================

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/vlp_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_module.addImport("vlp_gpu_shared", gpu_shared_host);
    test_module.addImport("vlp_types", types_mod);

    const tests = b.addTest(.{
        .name = "vlp_test",
        .root_module = test_module,
    });
    tests.linkSystemLibrary("cuda");
    tests.linkLibC();

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run VLP tests");
    test_step.dependOn(&run_tests.step);
}
