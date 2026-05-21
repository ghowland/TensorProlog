const std = @import("std");

const ptx_raw = @embedFile("vlp_kernel_ptx");
const ptx_source = ptx_raw ++ [_]u8{0};

const CUdevice = c_int;
const CUcontext = ?*anyopaque;
const CUmodule = ?*anyopaque;
const CUfunction = ?*anyopaque;
const CUdeviceptr = u64;
const CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;
const HMODULE = ?*anyopaque;
const FARPROC = ?*anyopaque;

extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) HMODULE;
extern "kernel32" fn GetProcAddress(module: HMODULE, name: [*:0]const u8) FARPROC;

fn cuda(comptime T: type, h: HMODULE, name: [*:0]const u8) ?T {
    const p = GetProcAddress(h, name);
    if (p == null) return null;
    return @ptrCast(p);
}

pub fn main() void {
    std.debug.print("=== VDR-Prolog PTX Test ===\n\n", .{});

    const nv = LoadLibraryA("nvcuda.dll");
    if (nv == null) {
        std.debug.print("FAIL: LoadLibraryA nvcuda.dll\n", .{});
        return;
    }
    std.debug.print("[OK] nvcuda.dll loaded\n", .{});

    const FnInit = *const fn (c_uint) callconv(.c) CUresult;
    const FnDeviceGet = *const fn (*CUdevice, c_int) callconv(.c) CUresult;
    const FnDeviceGetCount = *const fn (*c_int) callconv(.c) CUresult;
    const FnDeviceGetName = *const fn ([*]u8, c_int, CUdevice) callconv(.c) CUresult;
    const FnCtxCreate = *const fn (*CUcontext, c_uint, CUdevice) callconv(.c) CUresult;
    const FnCtxDestroy = *const fn (CUcontext) callconv(.c) CUresult;
    const FnCtxSync = *const fn () callconv(.c) CUresult;
    // const FnModuleLoadData = *const fn (*CUmodule, [*]const u8) callconv(.c) CUresult;
    const FnModuleUnload = *const fn (CUmodule) callconv(.c) CUresult;
    const FnModuleGetFunc = *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult;
    const FnMemAlloc = *const fn (*CUdeviceptr, usize) callconv(.c) CUresult;
    const FnMemFree = *const fn (CUdeviceptr) callconv(.c) CUresult;
    const FnMemcpyHtoD = *const fn (CUdeviceptr, [*]const u8, usize) callconv(.c) CUresult;
    const FnMemcpyDtoH = *const fn ([*]u8, CUdeviceptr, usize) callconv(.c) CUresult;
    const FnMemsetD32 = *const fn (CUdeviceptr, c_uint, usize) callconv(.c) CUresult;
    const FnLaunch = *const fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, ?*anyopaque, [*]?*anyopaque, ?*anyopaque) callconv(.c) CUresult;

    const cuInit = cuda(FnInit, nv, "cuInit") orelse {
        std.debug.print("FAIL: cuInit not found\n", .{});
        return;
    };
    const cuDeviceGet = cuda(FnDeviceGet, nv, "cuDeviceGet") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    const cuDeviceGetCount = cuda(FnDeviceGetCount, nv, "cuDeviceGetCount") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    const cuDeviceGetName = cuda(FnDeviceGetName, nv, "cuDeviceGetName") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    const cuCtxCreate = cuda(FnCtxCreate, nv, "cuCtxCreate_v2") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    const cuCtxDestroy = cuda(FnCtxDestroy, nv, "cuCtxDestroy_v2") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    const cuCtxSync = cuda(FnCtxSync, nv, "cuCtxSynchronize") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    // const cuModuleLoadData = cuda(FnModuleLoadData, nv, "cuModuleLoadData") orelse {
    //     std.debug.print("FAIL\n", .{});
    //     return;
    // };
    const FnModuleLoadDataEx = *const fn (*CUmodule, [*]const u8, c_uint, [*]c_int, [*]?*anyopaque) callconv(.c) CUresult;
    const cuModuleLoadDataEx = cuda(FnModuleLoadDataEx, nv, "cuModuleLoadDataEx") orelse {
        std.debug.print("FAIL: no cuModuleLoadDataEx\n", .{});
        return;
    };

    const cuModuleUnload = cuda(FnModuleUnload, nv, "cuModuleUnload") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    const cuModuleGetFunc = cuda(FnModuleGetFunc, nv, "cuModuleGetFunction") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    const cuMemAlloc = cuda(FnMemAlloc, nv, "cuMemAlloc_v2") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    const cuMemFree = cuda(FnMemFree, nv, "cuMemFree_v2") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    const cuMemcpyHtoD = cuda(FnMemcpyHtoD, nv, "cuMemcpyHtoD_v2") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    const cuMemcpyDtoH = cuda(FnMemcpyDtoH, nv, "cuMemcpyDtoH_v2") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    const cuMemsetD32 = cuda(FnMemsetD32, nv, "cuMemsetD32_v2") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };
    const cuLaunch = cuda(FnLaunch, nv, "cuLaunchKernel") orelse {
        std.debug.print("FAIL\n", .{});
        return;
    };

    std.debug.print("[OK] All 16 CUDA functions loaded\n", .{});

    if (cuInit(0) != CUDA_SUCCESS) {
        std.debug.print("FAIL: cuInit\n", .{});
        return;
    }
    std.debug.print("[OK] cuInit\n", .{});

    var dev_count: c_int = 0;
    _ = cuDeviceGetCount(&dev_count);
    std.debug.print("[OK] {} device(s)\n", .{dev_count});

    var device: CUdevice = 0;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
        std.debug.print("FAIL: cuDeviceGet\n", .{});
        return;
    }

    var name_buf: [256]u8 = [_]u8{0} ** 256;
    _ = cuDeviceGetName(&name_buf, 256, device);
    const name_len = std.mem.indexOfScalar(u8, &name_buf, 0) orelse 256;
    std.debug.print("[OK] {s}\n", .{name_buf[0..name_len]});

    var ctx: CUcontext = null;
    if (cuCtxCreate(&ctx, 0, device) != CUDA_SUCCESS) {
        std.debug.print("FAIL: cuCtxCreate\n", .{});
        return;
    }
    std.debug.print("[OK] Context\n", .{});

    std.debug.print("[DBG] PTX first 80 bytes: {s}\n", .{ptx_source[0..@min(80, ptx_source.len)]});
    std.debug.print("[DBG] PTX size: {} last byte: {}\n", .{ ptx_source.len, ptx_source[ptx_source.len - 1] });

    // var mod: CUmodule = null;
    // const lr = cuModuleLoadData(&mod, ptx_source.ptr);
    // if (lr != CUDA_SUCCESS) {
    //     std.debug.print("FAIL: cuModuleLoadData {}\n", .{lr});
    //     _ = cuCtxDestroy(ctx);
    //     return;
    // }
    // std.debug.print("[OK] PTX loaded ({} bytes)\n", .{ptx_source.len});

    var mod: CUmodule = null;
    var error_log: [4096]u8 = [_]u8{0} ** 4096;
    const error_log_ptr: ?*anyopaque = @ptrCast(&error_log);
    var error_log_size_val: usize = 4096;
    const error_log_size_ptr: ?*anyopaque = @ptrCast(&error_log_size_val);
    var jit_options = [_]c_int{ 5, 6 };
    var jit_values = [_]?*anyopaque{ error_log_ptr, error_log_size_ptr };
    const lr = cuModuleLoadDataEx(&mod, ptx_source.ptr, 2, &jit_options, &jit_values);
    if (lr != CUDA_SUCCESS) {
        const log_len = std.mem.indexOfScalar(u8, &error_log, 0) orelse 4096;
        std.debug.print("FAIL: cuModuleLoadDataEx {}\n", .{lr});
        std.debug.print("JIT: {s}\n", .{error_log[0..log_len]});
        _ = cuCtxDestroy(ctx);
        return;
    }
    std.debug.print("[OK] PTX loaded ({} bytes)\n", .{ptx_source.len});

    var func: CUfunction = null;
    if (cuModuleGetFunc(&func, mod, "vlp_kernel_$_main") != CUDA_SUCCESS) {
        std.debug.print("FAIL: GetFunction\n", .{});
        _ = cuModuleUnload(mod);
        _ = cuCtxDestroy(ctx);
        return;
    }
    std.debug.print("[OK] Kernel: vlp_kernel_$_main\n", .{});

    var dp: [15]CUdeviceptr = .{0} ** 15;
    for (0..15) |i| {
        if (cuMemAlloc(&dp[i], 4096) != CUDA_SUCCESS) {
            std.debug.print("FAIL: alloc {}\n", .{i});
            return;
        }
        _ = cuMemsetD32(dp[i], 0, 1024);
    }
    std.debug.print("[OK] 15 buffers\n", .{});

    const fire = struct {
        fn f(fn_launch: FnLaunch, fn_sync: FnCtxSync, d: *[15]CUdeviceptr) bool {
            var ap: [15]CUdeviceptr = d.*;
            var ka: [15]?*anyopaque = undefined;
            for (0..15) |i| ka[i] = @ptrCast(&ap[i]);
            if (fn_launch(null, 1, 1, 1, 256, 1, 1, 4096, null, &ka, null) != CUDA_SUCCESS) {
                std.debug.print("FAIL: launch\n", .{});
                return false;
            }
            if (fn_sync() != CUDA_SUCCESS) {
                std.debug.print("FAIL: sync\n", .{});
                return false;
            }
            return true;
        }
    }.f;
    _ = fire;

    // Need to pass func to launch — use a simpler approach
    const doLaunch = struct {
        fn go(fn_launch: FnLaunch, fn_sync: FnCtxSync, kfunc: CUfunction, d: *[15]CUdeviceptr) bool {
            var ap: [15]CUdeviceptr = d.*;
            var ka: [15]?*anyopaque = undefined;
            for (0..15) |i| ka[i] = @ptrCast(&ap[i]);
            if (fn_launch(kfunc, 1, 1, 1, 256, 1, 1, 4096, null, &ka, null) != CUDA_SUCCESS) {
                std.debug.print("FAIL: launch\n", .{});
                return false;
            }
            if (fn_sync() != CUDA_SUCCESS) {
                std.debug.print("FAIL: sync\n", .{});
                return false;
            }
            return true;
        }
    }.go;

    // TEST 1: residual_add
    std.debug.print("\n--- Test 1: residual_add ---\n", .{});
    {
        var a = [_]i32{ 10, 20, 30, 40, 50, 60, 70, 80 };
        var b = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        var p = [_]i32{0} ** 64;
        p[0] = 10;
        p[1] = 8;
        _ = cuMemcpyHtoD(dp[9], @ptrCast(&a), 32);
        _ = cuMemcpyHtoD(dp[10], @ptrCast(&b), 32);
        _ = cuMemcpyHtoD(dp[12], @ptrCast(&p), 256);
        if (!doLaunch(cuLaunch, cuCtxSync, func, &dp)) return;
        var r: [8]i32 = undefined;
        _ = cuMemcpyDtoH(@ptrCast(&r), dp[9], 32);
        if (std.mem.eql(i32, &r, &[_]i32{ 11, 22, 33, 44, 55, 66, 77, 88 }))
            std.debug.print("[PASS] residual_add\n", .{})
        else
            std.debug.print("[FAIL] got {any}\n", .{r});
    }

    // TEST 2: abs
    std.debug.print("\n--- Test 2: abs ---\n", .{});
    {
        var a = [_]i32{ -100, 200, -300, 0 };
        var p = [_]i32{0} ** 64;
        p[0] = 19;
        p[1] = 4;
        p[2] = 0;
        p[3] = 0;
        p[4] = 0;
        _ = cuMemcpyHtoD(dp[9], @ptrCast(&a), 16);
        _ = cuMemcpyHtoD(dp[12], @ptrCast(&p), 256);
        if (!doLaunch(cuLaunch, cuCtxSync, func, &dp)) return;
        var r: [4]i32 = undefined;
        _ = cuMemcpyDtoH(@ptrCast(&r), dp[10], 16);
        if (std.mem.eql(i32, &r, &[_]i32{ 100, 200, 300, 0 }))
            std.debug.print("[PASS] abs\n", .{})
        else
            std.debug.print("[FAIL] got {any}\n", .{r});
    }

    // TEST 3: binary add
    std.debug.print("\n--- Test 3: binary add ---\n", .{});
    {
        var ab = [_]i32{ 100, 200, 300, 400, 1, 2, 3, 4 };
        var p = [_]i32{0} ** 64;
        p[0] = 20;
        p[1] = 4;
        p[2] = 0;
        p[3] = 0;
        p[4] = 4;
        p[5] = 0;
        _ = cuMemcpyHtoD(dp[9], @ptrCast(&ab), 32);
        _ = cuMemcpyHtoD(dp[12], @ptrCast(&p), 256);
        if (!doLaunch(cuLaunch, cuCtxSync, func, &dp)) return;
        var r: [4]i32 = undefined;
        _ = cuMemcpyDtoH(@ptrCast(&r), dp[10], 16);
        if (std.mem.eql(i32, &r, &[_]i32{ 101, 202, 303, 404 }))
            std.debug.print("[PASS] binary add\n", .{})
        else
            std.debug.print("[FAIL] got {any}\n", .{r});
    }

    // TEST 4: softmax unity
    std.debug.print("\n--- Test 4: softmax unity ---\n", .{});
    {
        var logits = [_]i32{ 10000, 20000, 5000, 30000, 15000 };
        var p = [_]i32{0} ** 64;
        p[0] = 4;
        p[1] = 5;
        p[2] = 1;
        p[3] = 65536;
        _ = cuMemcpyHtoD(dp[9], @ptrCast(&logits), 20);
        _ = cuMemcpyHtoD(dp[12], @ptrCast(&p), 256);
        if (!doLaunch(cuLaunch, cuCtxSync, func, &dp)) return;
        var r: [5]i32 = undefined;
        _ = cuMemcpyDtoH(@ptrCast(&r), dp[9], 20);
        var sum: i64 = 0;
        for (r) |v| sum += v;
        std.debug.print("  probs: {any}\n  sum: {}\n", .{ r, sum });
        if (sum == 65536) std.debug.print("[PASS] softmax unity\n", .{}) else std.debug.print("[FAIL]\n", .{});
    }

    // TEST 5: determinism
    std.debug.print("\n--- Test 5: determinism ---\n", .{});
    {
        var logits = [_]i32{ 10000, 20000, 5000, 30000, 15000 };
        var p = [_]i32{0} ** 64;
        p[0] = 4;
        p[1] = 5;
        p[2] = 1;
        p[3] = 65536;
        var r1: [5]i32 = undefined;
        var r2: [5]i32 = undefined;
        _ = cuMemcpyHtoD(dp[9], @ptrCast(&logits), 20);
        _ = cuMemcpyHtoD(dp[12], @ptrCast(&p), 256);
        if (!doLaunch(cuLaunch, cuCtxSync, func, &dp)) return;
        _ = cuMemcpyDtoH(@ptrCast(&r1), dp[9], 20);
        _ = cuMemcpyHtoD(dp[9], @ptrCast(&logits), 20);
        _ = cuMemcpyHtoD(dp[12], @ptrCast(&p), 256);
        if (!doLaunch(cuLaunch, cuCtxSync, func, &dp)) return;
        _ = cuMemcpyDtoH(@ptrCast(&r2), dp[9], 20);
        if (std.mem.eql(i32, &r1, &r2)) std.debug.print("[PASS] deterministic\n", .{}) else std.debug.print("[FAIL]\n", .{});
    }

    // TEST 6: Q16 mul
    std.debug.print("\n--- Test 6: Q16 mul ---\n", .{});
    {
        var ab = [_]i32{ 32768, 32768 };
        var p = [_]i32{0} ** 64;
        p[0] = 20;
        p[1] = 1;
        p[2] = 2;
        p[3] = 0;
        p[4] = 1;
        p[5] = 0;
        _ = cuMemcpyHtoD(dp[9], @ptrCast(&ab), 8);
        _ = cuMemcpyHtoD(dp[12], @ptrCast(&p), 256);
        if (!doLaunch(cuLaunch, cuCtxSync, func, &dp)) return;
        var r: [1]i32 = undefined;
        _ = cuMemcpyDtoH(@ptrCast(&r), dp[10], 4);
        if (r[0] == 16384) std.debug.print("[PASS] Q16 mul\n", .{}) else std.debug.print("[FAIL] got {}\n", .{r[0]});
    }

    std.debug.print("\n=== Done ===\n", .{});
    for (&dp) |*p| if (p.* != 0) {
        _ = cuMemFree(p.*);
    };
    _ = cuModuleUnload(mod);
    _ = cuCtxDestroy(ctx);
}
