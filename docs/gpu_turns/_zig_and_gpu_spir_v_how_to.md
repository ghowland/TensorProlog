```
================================================================================
Zig 0.16.0 SPIR-V / GPU Technical Reference
Everything a new implementor needs to know. No search required.
================================================================================

================================================================================
1. WHAT ZIG CAN TARGET ON GPU
================================================================================

Zig has a self-hosted SPIR-V backend (default) and an LLVM SPIR-V backend
(opt-in via -fllvm). The self-hosted backend is more mature for Vulkan targets.

Supported GPU targets:
  spirv32-vulkan    — 32-bit SPIR-V for Vulkan (most common)
  spirv64-vulkan    — 64-bit SPIR-V for Vulkan
  spirv64-opencl    — 64-bit SPIR-V for OpenCL (more permissive pointer model)
  nvptx64-cuda      — NVIDIA PTX via LLVM
  amdgcn-amdhsa     — AMD GCN via LLVM

For Vulkan compute shaders, use spirv32-vulkan with vulkan_v1_2 CPU model.

================================================================================
2. HOW TO COMPILE A ZIG SHADER
================================================================================

CLI:
  zig build-obj -target spirv32-vulkan -mcpu vulkan_v1_2 -ofmt=spirv -fno-llvm shader.zig

build.zig (0.16.0 syntax):
  const spirv_target = b.resolveTargetQuery(.{
      .cpu_arch = .spirv32,
      .os_tag = .vulkan,
      .cpu_model = .{ .explicit = &std.Target.spirv.cpu.vulkan_v1_2 },
      .ofmt = .spirv,
  });
  const shader = b.addObject(.{
      .name = "my_shader",
      .root_module = b.createModule(.{
          .root_source_file = b.path("shader.zig"),
          .target = spirv_target,
          .optimize = .ReleaseFast,
      }),
      .use_llvm = false,
      .use_lld = false,
  });

Embed in host executable:
  exe.root_module.addAnonymousImport(
      "my_shader_spv",
      .{ .root_source_file = shader.getEmittedBin() },
  );

Host loads it:
  const spv align(@alignOf(u32)) = @embedFile("my_shader_spv").*;

SPIR-V must be 32-bit aligned when fed to Vulkan. The dereference + align
trick (.*) converts the embed from a pointer to a value with correct alignment.

================================================================================
3. WHAT A ZIG COMPUTE SHADER LOOKS LIKE
================================================================================

const std = @import("std");
const gpu = std.gpu;

// Storage buffer bindings — extern + addrspace
const InputBlock = extern struct {
    data: [256]i32,
};
const OutputBlock = extern struct {
    data: [256]i32,
};

extern var input: InputBlock addrspace(.storage_buffer);
extern var output: OutputBlock addrspace(.storage_buffer);

// Entry point — callconv(.spirv_kernel) emits GLCompute
export fn main() callconv(.spirv_kernel) void {
    // Execution property — sets local_size (workgroup size)
    @setExecProperty(.local_size, .{ 256, 1, 1 });

    // Descriptor set and binding annotations
    gpu.binding(&input, 0, 0);   // set 0, binding 0
    gpu.binding(&output, 0, 1);  // set 0, binding 1

    // Built-in invocation ID
    const gid = gpu.global_invocation_id;
    const idx: u32 = gid[0];

    if (idx < 256) {
        output.data[idx] = input.data[idx] * 2;
    }
}

Key elements:
  - callconv(.spirv_kernel) → generates GLCompute entry point
  - addrspace(.storage_buffer) → SPIR-V StorageBuffer storage class
  - addrspace(.uniform) → SPIR-V Uniform storage class
  - addrspace(.input) → SPIR-V Input storage class (built-ins)
  - addrspace(.output) → SPIR-V Output storage class
  - addrspace(.shared) → SPIR-V Workgroup storage class
  - addrspace(.push_constant) → SPIR-V PushConstant storage class
  - addrspace(.global) → SPIR-V PhysicalStorageBuffer (Vulkan 1.2+)
  - gpu.binding(&var, set, binding) → OpDecorate DescriptorSet + Binding
  - gpu.location(&var, loc) → OpDecorate Location (vertex/fragment only)
  - gpu.global_invocation_id → @Vector(3, u32), built-in GlobalInvocationId
  - gpu.local_invocation_id → @Vector(3, u32), built-in LocalInvocationId
  - gpu.workgroup_id → @Vector(3, u32), built-in WorkgroupId
  - gpu.workgroup_size → @Vector(3, u32), built-in WorkgroupSize
  - gpu.num_workgroups → @Vector(3, u32), built-in NumWorkgroups
  - @setExecProperty(.local_size, .{X, Y, Z}) → OpExecutionMode LocalSize

================================================================================
4. WHAT DOES NOT WORK IN ZIG SPIR-V (as of 0.16.0)
================================================================================

HARD BLOCKS — will not compile or will fail spirv-val:

  a) Recursion. SPIR-V has no call stack. Functions can exist but cannot
     call themselves or form cycles. The compiler may inline small functions
     but will reject recursive ones.

  b) Dynamic allocation. No allocator, no heap, no mmap, no sbrk.
     All memory must be statically sized or provided via bound buffers.

  c) Function pointers. No OpFunctionPointerINTEL without extensions.
     Cannot store or call through function pointers. No vtables.

  d) std library. Almost nothing from std works. No std.mem.Allocator,
     no std.ArrayList, no std.HashMap, no std.io, no std.fs, no std.fmt.
     Only std.math (partially), std.gpu, and basic builtins.

  e) Error unions / optionals with complex payloads. Simple error returns
     may work. Complex error handling (try/catch chains) generates code
     patterns the backend may not support.

  f) Printing / logging. No stdout, no debug output. Kernels are silent.
     Communicate results through buffer writes only.

CURRENT BUGS / LIMITATIONS (may be fixed in future Zig versions):

  g) storage_buffer pointer indexing. Accessing [*]addrspace(.storage_buffer)
     via runtime index fails with "cannot perform arithmetic on pointers
     with address space 'storage_buffer'". Workaround: use fixed-size arrays
     inside extern structs instead of pointer-to-many.
     Tracking: ziglang/zig#25638

  h) PhysicalStorageBuffer alignment. Loads from .global (PhysicalStorageBuffer)
     require Aligned memory access decoration. The backend does not always
     emit this. Tracking: ziglang/zig#23212

  i) Windows build system. SPIR-V output via build.zig fails on Windows
     with "NotOpenForWriting" due to a linker file handle bug. Works on
     Linux. Tracking: ziglang/zig#23883

  j) spirv-val failures. The backend occasionally emits SPIR-V that passes
     compilation but fails Vulkan validation. Common: missing layout
     decorations on struct members, missing Aligned on PhysicalStorageBuffer
     accesses. Run spirv-val on every .spv file before loading into Vulkan.

  k) Array concatenation (++ operator) in shaders generates code that some
     GPU drivers reject (ACO compiler error on AMD: "Unimplemented intrinsic
     instr: @store_deref"). Avoid ++ in shader code. Use loops.

  l) The SPIR-V backend passes ~50% of Zig's behavior tests for Vulkan
     targets and ~75% for OpenCL targets. Many failing tests exercise
     features not meant for GPU (allocators, error returns, etc). But
     some are genuine backend gaps.

================================================================================
5. THE PRACTICAL PATH: GLSL KERNELS + ZIG HOST
================================================================================

Given the limitations in section 4, the recommended architecture is:

  GPU kernels: written in GLSL, compiled to SPIR-V with glslc
  Host code: written in Zig, uses vulkan-zig bindings
  Bridge: Zig host creates Vulkan resources, loads .spv files, dispatches

This gives you:
  - Proven GLSL→SPIR-V path (glslc is mature, spirv-val clean)
  - Full Zig host with allocators, error handling, threading, filesystem
  - Clean boundary: host manages state, GPU does parallel math
  - Future migration: when Zig SPIR-V backend matures, port GLSL→Zig
    one kernel at a time. Same descriptor sets, same buffer layouts.

GLSL compute shader example (equivalent to the Zig shader in section 3):

  #version 450
  layout(local_size_x = 256) in;
  layout(set = 0, binding = 0) buffer Input { int data[]; } input_buf;
  layout(set = 0, binding = 1) buffer Output { int data[]; } output_buf;
  void main() {
      uint idx = gl_GlobalInvocationID.x;
      if (idx < 256) {
          output_buf.data[idx] = input_buf.data[idx] * 2;
      }
  }

Compile: glslc --target-env=vulkan1.2 -o shader.spv shader.comp

build.zig integration:
  const glslc_cmd = b.addSystemCommand(&.{ "glslc", "--target-env=vulkan1.2", "-o" });
  const spv = glslc_cmd.addOutputFileArg("shader.spv");
  glslc_cmd.addFileArg(b.path("shaders/shader.comp"));
  exe.root_module.addAnonymousImport("shader_spv", .{ .root_source_file = spv });

================================================================================
6. VULKAN-ZIG BINDINGS
================================================================================

Repository: github.com/Snektron/vulkan-zig
Compat: master branch tracks zig master (0.17-dev). For 0.16.0 use the
        commit that was current at 0.16.0 release (the commit whose
        generator uses std.process.Init — this is the 0.16 API).

build.zig.zon dependencies:
  .vulkan = .{ .url = "git+https://github.com/Snektron/vulkan-zig#<commit>", .hash = "..." },
  .vulkan_headers = .{ .url = "git+https://github.com/KhronosGroup/Vulkan-Headers#<commit>", .hash = "..." },

build.zig setup:
  const vulkan_dep = b.dependency("vulkan", .{
      .registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml"),
  });
  const vulkan_mod = vulkan_dep.module("vulkan-zig");
  exe.root_module.addImport("vulkan", vulkan_mod);

Key API patterns:

  Loading (no static linking — all dynamic):
    var lib = std.DynLib.open("libvulkan.so.1");  // Linux
    const get_proc = lib.lookup("vkGetInstanceProcAddr");
    const vkb = vk.BaseWrapper(.{}).load(get_proc);

  Dispatch tables (compile-time feature selection):
    const InstanceDispatch = vk.InstanceWrapper(.{
        .destroyInstance = true,
        .enumeratePhysicalDevices = true,
        .createDevice = true,
        // ... only enable functions you use
    });

  Creating compute pipeline:
    const module = vkd.createShaderModule(.{
        .code_size = spv.len,
        .p_code = @ptrCast(&spv),
    });
    const pipeline_info = vk.ComputePipelineCreateInfo{
        .stage = .{ .stage = .{ .compute_bit = true }, .module = module, .p_name = "main" },
        .layout = pipeline_layout,
    };
    vkd.createComputePipelines(.null_handle, 1, @ptrCast(&pipeline_info), null, @ptrCast(&pipeline));

  Dispatch:
    vkd.cmdBindPipeline(cmd, .compute, pipeline);
    vkd.cmdBindDescriptorSets(cmd, .compute, layout, 0, 1, @ptrCast(&desc_set), 0, undefined);
    vkd.cmdDispatch(cmd, group_x, group_y, group_z);

  Memory:
    Host-visible + coherent for staging / mapped access.
    Device-local for performance-critical buffers.
    vkd.mapMemory → direct pointer access (if host-visible).

================================================================================
7. ADDRESS SPACES AND WHAT THEY MEAN
================================================================================

Vulkan SPIR-V has distinct address spaces. Zig exposes them via addrspace().
A pointer in one address space CANNOT be cast to another (no OpPtrCastToGeneric
in Vulkan — that's OpenCL only).

  .storage_buffer   — Large read-write buffers. SSBO in GLSL. Most data lives here.
                      Logical pointers — no pointer arithmetic allowed (Vulkan default).
                      Can index via fixed-size arrays inside extern structs.
                      Cannot use [*]addrspace(.storage_buffer) with runtime index (bug #25638).

  .uniform          — Small read-only buffers. UBO in GLSL. Dispatch params go here.
                      Typically 64 KB max (minMaxUniformBufferRange).
                      Broadcast-optimized — all invocations read same value cheaply.

  .push_constant    — Tiny (128-256 bytes) inline constants. No buffer needed.
                      Fastest path for small per-dispatch parameters.
                      Alternative to uniform buffer for params < 128 bytes.

  .shared           — Workgroup-local memory. shared in GLSL.
                      16-48 KB typical (32 KB guaranteed by Vulkan spec).
                      Shared across invocations within ONE workgroup.
                      NOT visible across workgroups.
                      Declare as: var scratch: [1024]i32 addrspace(.shared) = undefined;

  .input            — Built-in inputs (invocation ID, workgroup ID, etc).
                      Read-only. Populated by hardware.

  .output           — Built-in outputs (vertex position, frag color).
                      Not used in compute shaders.

  .global           — PhysicalStorageBuffer64. Real pointers with arithmetic.
                      Requires bufferDeviceAddress feature (Vulkan 1.2 core, ~95% coverage).
                      Use for pointer-rich data structures that need runtime indexing.
                      Alignment decorations required on every load/store (bug #23212).

For compute shaders, you primarily use: storage_buffer, uniform, shared, input.

================================================================================
8. WHAT THE GPU CAN AND CANNOT DO — DECISION MATRIX
================================================================================

CAN do on GPU (dispatch as compute shader):
  ✓ Matrix multiply (GEMM) — embarrassingly parallel
  ✓ Element-wise arithmetic on arrays — one invocation per element
  ✓ Parallel reduction (sum, max, argmax) — tree reduction in shared memory
  ✓ Parallel scan/filter — each invocation checks one element
  ✓ Attention score computation — independent per head×position
  ✓ Softmax — per-row reduction + normalization
  ✓ Layer normalization — per-token mean/variance reduction
  ✓ Bitonic sort — compare-and-swap network, no recursion
  ✓ Parallel unification — each invocation tests one candidate
  ✓ Atomic counters — atomicAdd for result counting

CANNOT do on GPU (must be host-side):
  ✗ Recursion (Prolog backtracking, tree traversal)
  ✗ Dynamic memory allocation (growing arrays, hash maps)
  ✗ String manipulation (concatenation, formatting, parsing)
  ✗ Function pointer dispatch (builtin tables, vtables)
  ✗ File I/O, network I/O, process spawning
  ✗ Complex control flow (deeply nested conditionals, variable-length loops
    where the bound isn't known at dispatch time)
  ✗ Error handling with try/catch
  ✗ Anything requiring std library

MARGINAL — can do but shouldn't (host is faster for small N):
  ~ Fact scan with < 256 elements — dispatch overhead > scan time
  ~ Unification with < 32 candidates — one warp isn't worth the setup
  ~ Sampling over vocabulary — sequential scan, CPU does it in microseconds
  ~ Single fact read/write — one buffer transfer > one memcpy
  ~ Access control checks — 5-10 integer comparisons, nanoseconds

================================================================================
9. GPU KERNEL DESIGN RULES
================================================================================

Rule 1: One entry point per .spv module.
  Vulkan compute shaders have exactly one entry point named "main" (GLSL)
  or an exported function with callconv(.spirv_kernel) (Zig). You cannot
  have two kernels in one module. One .comp/.zig file → one .spv → one VkPipeline.

Rule 2: All data through descriptor sets.
  Kernels cannot access host memory directly. All input/output goes through
  bound storage buffers, uniform buffers, or push constants. The host writes
  data to buffers before dispatch and reads results after.

Rule 3: Workgroup size is compile-time.
  layout(local_size_x = N) in GLSL or @setExecProperty(.local_size, .{N,1,1})
  in Zig. Cannot be changed at dispatch time (specialization constants can
  override, but the options must be compiled in). Choose 256 as default —
  good occupancy on most GPUs, evenly divides common data sizes.

Rule 4: No cross-workgroup communication during execution.
  Workgroups execute independently. Shared memory is per-workgroup.
  To communicate across workgroups, write to storage buffer, dispatch a
  barrier, dispatch the next kernel. The host orchestrates the sequence.

Rule 5: Atomic operations for cross-invocation counters.
  atomicAdd on storage buffer is the standard pattern for counting results.
  Works across workgroups. Available for i32 and u32 on all Vulkan devices.
  i64 atomics require the Int64Atomics feature (not universal).

Rule 6: Bounds check everything.
  GPU has no segfault. Out-of-bounds buffer access is undefined behavior.
  Every invocation must check: if (gid >= n_elements) return;
  The host may dispatch more invocations than elements (rounded up to
  workgroup size). The extra invocations must exit immediately.

Rule 7: No variable-length loops without a hard bound.
  Every loop in a kernel must have a compile-time or dispatch-param maximum.
  while (condition) is dangerous — if condition never becomes false, the GPU
  hangs. Use: for (0..max_iterations) |i| { if (done) break; }

Rule 8: Pipeline barriers between dependent dispatches.
  If kernel B reads what kernel A wrote, there must be a
  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT → VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
  barrier with VK_ACCESS_SHADER_WRITE_BIT → VK_ACCESS_SHADER_READ_BIT
  between the two dispatches. Without this, B may read stale data.

Rule 9: Host read barrier after last kernel.
  If the host needs to read GPU results, the last dispatch needs a
  COMPUTE_SHADER → HOST barrier with SHADER_WRITE → HOST_READ access flags.

Rule 10: Status reporting via buffer, not return values.
  Kernels return void. To report errors: write an error code to
  status_buffer[global_invocation_id.x]. Host reads status buffer after
  dispatch. Convention: 0 = success, nonzero = error code from ErrorCode enum.

================================================================================
10. THE HOST-DEVICE DATA CONTRACT
================================================================================

All data crossing the boundary is extern struct with explicit padding.
Layout rules:
  - i32 fields aligned to 4 bytes
  - i64 fields aligned to 8 bytes
  - i16 fields aligned to 2 bytes (but prefer i32 for GPU — 16-bit may be slow)
  - i8 fields aligned to 1 byte (pad to 4 for GPU buffers)
  - Arrays: element stride must match sizeof(element)
  - No pointers. Offsets (i32 index into a flat array) instead.
  - Total struct size should be multiple of 4 for buffer alignment

Structs shared with GPU (must be extern struct):
  Q16 (8 bytes), Fact (40 bytes), Kb (256 bytes), Term (24 bytes),
  Rule (48 bytes), Binding (8 bytes), Grammar (28 bytes),
  GrammarSlot, GrammarFill, GrammarKbMapping,
  all dispatch params in vlp_gpu_params.zig

Structs host-only (can be regular struct):
  Session, Runner, Grant, AuditEntry, Command,
  SearchResult, QueryResult, FireResult, CompileResult,
  all engine structs (KbStore, PrologEngine, etc)

Addressing convention:
  Flat arrays in storage buffers. Element N is at byte offset N * sizeof(element).
  KB N is at byte offset N * 256.
  Fact N is at byte offset N * 40.
  Term N is at byte offset N * 24.
  The host computes byte offsets. The GPU kernel receives base_offset as a param
  and indexes from there: element = buffer[base_offset + gid].

================================================================================
11. Zig 0.16.0 BREAKING CHANGES FROM 0.15.x THAT AFFECT THIS PROJECT
================================================================================

  a) std.process.Init — new "Juicy Main" API. pub fn main() may need to
     accept init: std.process.Init parameter. vulkan-zig master uses this.
     0.15.x vulkan-zig does NOT compile on 0.16.0 and vice versa.

  b) std.DynLib may have moved or changed API. Verify the open/lookup
     signatures against 0.16.0 std.

  c) Package management: dependencies go in zig-pkg/ directory (local),
     not just .zig-cache. build.zig.zon hash format changed. Re-run
     zig fetch --save to regenerate hashes after version change.

  d) std.io reworked — GenericReader, AnyReader, FixedBufferStream removed.
     If any module uses these for file I/O, update to new std.Io API.

  e) std.Thread.Pool removed. If you were using it, switch to manual
     std.Thread.spawn (which the runner module already does).

  f) @Type replaced with individual type-creating builtins.
     Doesn't affect this project (no comptime type construction on GPU path).

  g) Vectors and arrays no longer support in-memory coercion.
     If any code coerces [4]i32 to @Vector(4, i32), add explicit conversion.

  h) Forbid runtime vector indexes. vec[runtime_idx] is now illegal.
     Use @shuffle or convert to array first: const arr: [N]T = vec;

================================================================================
12. TOOLCHAIN VERSIONS TO PIN
================================================================================

  Zig:             0.16.0 (stable, released April 14, 2026)
  vulkan-zig:      pin to specific commit hash compatible with 0.16.0
  Vulkan-Headers:  pin to specific commit hash (v1.3.283 or later)
  glslc:           from Vulkan SDK or apt install glslc
  spirv-val:       from SPIRV-Tools, run on every .spv before loading

Download URLs:
  Windows: https://ziglang.org/download/0.16.0/zig-x86_64-windows-0.16.0.zip
  Linux:   https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz

Pin commits in build.zig.zon with #<sha> fragment. Do not use bare branch
references — they drift.

================================================================================
13. TESTING GPU CODE
================================================================================

  1. Compile shader: zig build (or glslc)
  2. Validate: spirv-val shader.spv — must pass with zero errors
  3. Create Vulkan instance with validation layers enabled
  4. Load shader module, create pipeline
  5. Allocate input/output buffers with known test data
  6. Dispatch, fence, read back
  7. Compare output to expected values — INTEGER EQUALITY, no tolerance
  8. If mismatch: the shader is wrong. Not "close enough." Wrong. Fix it.

For the VLP system specifically:
  - testSoftmaxSumInvariant: sum of output row must == D (65536). Exactly.
  - testDeterminism: run N times, compare bit-by-bit. Any difference is a bug.
  - testSnapshotRoundtrip: save, modify, restore, compare. Bit-identical.
  - All VDR arithmetic: a.v * b.v intermediate in i64, result in i32.
    Cross-check host computation against GPU computation. Must match.

================================================================================
14. COMMON PITFALLS
================================================================================

  1. Forgetting @alignOf(u32) on embedded SPIR-V. Vulkan rejects misaligned .spv.
  2. Dispatching more invocations than status_buffer can hold. Silent overflow.
  3. Reading scratch buffer before dispatch fence completes. Race condition.
  4. Using regular struct instead of extern struct for GPU-shared data. Wrong layout.
  5. Assuming shared memory > 32 KB. Query device properties first.
  6. Not resetting result_counts before dispatch. Stale counts from previous dispatch.
  7. Passing slot-relative offsets when kernel expects absolute offsets. Wrong data.
  8. Forgetting pipeline barrier between dependent dispatches. Stale reads.
  9. Using ++ (array concatenation) in shader. Driver compiler may crash (AMD).
  10. Calling bridge.dispatch from multiple threads. Command buffer corruption.
  11. Text operations in kernel. No strings on GPU. Format on host.
  12. Variable-length loop without bound. GPU hang.
  13. Assuming i64 atomics are available. Check Int64Atomics feature first.
  14. Not pinning dependency commits. Build breaks next week when master moves.
  15. Using float anywhere in the VDR pipeline. The whole point is integer-only.
```
