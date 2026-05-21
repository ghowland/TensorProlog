// ============================================================
// build.zig
// Compiles vlp_kernel.zig to SPIR-V, embeds in host binary,
// compiles all host modules, links Vulkan.
// ============================================================

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Step 1: Compile vlp_kernel.zig to SPIR-V ──

    const spirv_target = b.resolveTargetQuery(.{
        .cpu_arch = .spirv32,
        .os_tag = .vulkan,
        .cpu_model = .{ .explicit = &std.Target.spirv.cpu.vulkan_v1_2 },
        .ofmt = .spirv,
    });

    const kernel = b.addObject(.{
        .name = "vlp_kernel",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vlp_kernel_minimal.zig"),
            .target = spirv_target,
            .optimize = .ReleaseFast,
        }),
        .use_llvm = false,
        .use_lld = false,
    });

    // ── Step 2: Shared GPU module (imported by both kernel and host) ──

    const gpu_shared_module = b.createModule(.{
        .root_source_file = b.path("src/vlp_gpu_shared.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ── Step 3: Host modules ──

    // vlp_types imports gpu_shared
    const types_module = b.createModule(.{
        .root_source_file = b.path("src/vlp_types.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vlp_gpu_shared", .module = gpu_shared_module },
        },
    });

    // vlp_gpu_params imports gpu_shared
    const params_module = b.createModule(.{
        .root_source_file = b.path("src/vlp_gpu_params.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vlp_gpu_shared", .module = gpu_shared_module },
        },
    });

    // vlp_device_memory imports vlp_types
    const device_memory_module = b.createModule(.{
        .root_source_file = b.path("src/vlp_device_memory.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vlp_types", .module = types_module },
            .{ .name = "vlp_gpu_shared", .module = gpu_shared_module },
        },
    });

    // vlp_bridge imports types, params, device_memory, gpu_shared
    // and embeds the compiled SPIR-V kernel
    const bridge_module = b.createModule(.{
        .root_source_file = b.path("src/vlp_bridge.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "vlp_types", .module = types_module },
            .{ .name = "vlp_gpu_params", .module = params_module },
            .{ .name = "vlp_gpu_shared", .module = gpu_shared_module },
            .{ .name = "vlp_device_memory", .module = device_memory_module },
        },
    });

    // Embed compiled SPIR-V into bridge module
    bridge_module.addAnonymousImport("vlp_kernel_spv", .{
        .root_source_file = kernel.getEmittedBin(),
    });

    // ── Step 4: Main executable ──

    const exe = b.addExecutable(.{
        .name = "vlp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vlp_gpu_shared", .module = gpu_shared_module },
                .{ .name = "vlp_types", .module = types_module },
                .{ .name = "vlp_gpu_params", .module = params_module },
                .{ .name = "vlp_device_memory", .module = device_memory_module },
                .{ .name = "vlp_bridge", .module = bridge_module },
            },
        }),
    });

    // Link Vulkan loader
    exe.root_module.linkSystemLibrary("vulkan", .{});

    b.installArtifact(exe);

    // ── Run step ──

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step("run", "Build and run VLP");
    run_step.dependOn(&run_cmd.step);

    // ── Test step ──

    const test_exe = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/vlp_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "vlp_gpu_shared", .module = gpu_shared_module },
                .{ .name = "vlp_types", .module = types_module },
                .{ .name = "vlp_bridge", .module = bridge_module },
            },
        }),
    });

    const test_step = b.step("test", "Run VLP tests");
    test_step.dependOn(&b.addRunArtifact(test_exe).step);
}
