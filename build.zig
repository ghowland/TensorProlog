const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Step 1: PTX kernel
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

    const install_ptx = b.addInstallFile(ptx_kernel.getEmittedAsm(), "vlp_kernel.ptx");
    b.getInstallStep().dependOn(&install_ptx.step);

    // Step 2: Test binary
    const test_module = b.createModule(.{
        .root_source_file = b.path("src/test_ptx.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_module.addAnonymousImport("vlp_kernel_ptx", .{
        .root_source_file = ptx_kernel.getEmittedAsm(),
    });

    const test_exe = b.addExecutable(.{
        .name = "test_ptx",
        .root_module = test_module,
    });

    b.installArtifact(test_exe);

    const run_cmd = b.addRunArtifact(test_exe);
    const run_step = b.step("test", "Run PTX GPU tests");
    run_step.dependOn(&run_cmd.step);
}
