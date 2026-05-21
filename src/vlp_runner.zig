// ============================================================
// vlp_runner.zig
// Runner scheduler — host-side autonomous execution.
// Poller, processor, internal, batch runner types.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const session_mod = @import("vlp_session.zig");
const inference_mod = @import("vlp_inference.zig");

// ============================================================
// Runner configurations
// ============================================================

pub const PollerConfig = struct {
    session: types.SessionHandle,
    interval_ms: i32,
    max_consecutive_errors: i32 = 5,
    kb_id: i32 = -1, // KB to fire rules against
};

pub const ProcessorConfig = struct {
    session: types.SessionHandle,
    source_type: SourceConnectionType,
    source_url: [512]u8 = [_]u8{0} ** 512,
    source_url_len: i32 = 0,
    max_turns_before_recycle: i32 = 200,
    max_consecutive_errors: i32 = 5,
    backoff_initial_ms: i32 = 1000,
    backoff_max_ms: i32 = 60000,
};

pub const SourceConnectionType = enum(i32) {
    http_poll = 0,
    websocket = 1,
    webhook = 2,
    grpc_stream = 3,
};

pub const InternalConfig = struct {
    session: types.SessionHandle,
    interval_ms: i32,
    compute_kb_id: i32,
};

pub const BatchConfig = struct {
    session: types.SessionHandle,
    task_queue_kb_id: i32,
    task_queue_slot: i32 = 0,
    max_concurrent: i32 = 4,
};

// ============================================================
// Runner status (returned by getStatus)
// ============================================================

pub const RunnerStatus = struct {
    state: types.RunnerState,
    runner_type: types.RunnerType,
    iterations_completed: i64,
    errors_consecutive: i32,
    errors_total: i64,
    last_iteration_ms: i32,
    last_iteration_timestamp: i32,
    recycle_count: i32,
};

// ============================================================
// Internal runner context — per-runner mutable state
// ============================================================

const RunnerContext = struct {
    runner_idx: i32,
    session_handle: types.SessionHandle,
    config_type: types.RunnerType,
    stop_requested: bool,
    kill_requested: bool,

    // Type-specific config
    interval_ms: i32,
    kb_id: i32,
    max_turns_recycle: i32,
    max_errors: i32,
    backoff_ms: i32,
    backoff_max_ms: i32,
    max_concurrent: i32,
    task_queue_kb_id: i32,
    task_queue_slot: i32,

    // Backoff state
    current_backoff_ms: i32,
};

// ============================================================
// Runner Scheduler
// ============================================================

pub const RunnerScheduler = struct {
    allocator: std.mem.Allocator,
    session_mgr: *session_mod.SessionManager,
    inference: *inference_mod.InferenceEngine,

    runners: []types.Runner,
    contexts: []RunnerContext,
    runner_active: []bool,
    runner_capacity: i32,
    runner_count: i32,
    next_runner_id: i32,

    // Thread handles for running runners
    threads: []?std.Thread,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(
    allocator: std.mem.Allocator,
    session_mgr: *session_mod.SessionManager,
    inference: *inference_mod.InferenceEngine,
    max_runners: i32,
) RunnerScheduler {
    const cap: usize = @intCast(max_runners);
    const runners = allocator.alloc(types.Runner, cap) catch &.{};
    const ctxs = allocator.alloc(RunnerContext, cap) catch &.{};
    const active = allocator.alloc(bool, cap) catch &.{};
    const threads = allocator.alloc(?std.Thread, cap) catch &.{};

    if (active.len > 0) @memset(active, false);
    if (threads.len > 0) @memset(threads, null);

    return .{
        .allocator = allocator,
        .session_mgr = session_mgr,
        .inference = inference,
        .runners = runners,
        .contexts = ctxs,
        .runner_active = active,
        .runner_capacity = max_runners,
        .runner_count = 0,
        .next_runner_id = 0,
        .threads = threads,
    };
}

pub fn deinit(self: *RunnerScheduler) void {
    // Stop all running runners
    for (self.runner_active, 0..) |active, i| {
        if (active) {
            self.contexts[i].kill_requested = true;
        }
    }
    // Wait for threads
    for (self.threads) |*t| {
        if (t.*) |thread| {
            thread.join();
            t.* = null;
        }
    }
    if (self.runners.len > 0) self.allocator.free(self.runners);
    if (self.contexts.len > 0) self.allocator.free(self.contexts);
    if (self.runner_active.len > 0) self.allocator.free(self.runner_active);
    if (self.threads.len > 0) self.allocator.free(self.threads);
}

// ============================================================
// Create runners
// ============================================================

pub fn createPoller(self: *RunnerScheduler, config: *const PollerConfig) ?types.RunnerHandle {
    const idx = self.allocSlot() orelse return null;
    const id = self.next_runner_id;
    self.next_runner_id += 1;

    self.runners[@intCast(idx)] = buildRunner(id, .poller, config.session.id, config.interval_ms, config.max_consecutive_errors);
    self.contexts[@intCast(idx)] = .{
        .runner_idx = idx,
        .session_handle = config.session,
        .config_type = .poller,
        .stop_requested = false,
        .kill_requested = false,
        .interval_ms = config.interval_ms,
        .kb_id = config.kb_id,
        .max_turns_recycle = 0,
        .max_errors = config.max_consecutive_errors,
        .backoff_ms = 0,
        .backoff_max_ms = 0,
        .max_concurrent = 0,
        .task_queue_kb_id = -1,
        .task_queue_slot = 0,
        .current_backoff_ms = 0,
    };

    return .{ .id = id, .index = idx };
}

pub fn createProcessor(self: *RunnerScheduler, config: *const ProcessorConfig) ?types.RunnerHandle {
    const idx = self.allocSlot() orelse return null;
    const id = self.next_runner_id;
    self.next_runner_id += 1;

    self.runners[@intCast(idx)] = buildRunner(id, .processor, config.session.id, 0, config.max_consecutive_errors);
    self.runners[@intCast(idx)].max_turns_before_recycle = config.max_turns_before_recycle;
    self.contexts[@intCast(idx)] = .{
        .runner_idx = idx,
        .session_handle = config.session,
        .config_type = .processor,
        .stop_requested = false,
        .kill_requested = false,
        .interval_ms = 0,
        .kb_id = -1,
        .max_turns_recycle = config.max_turns_before_recycle,
        .max_errors = config.max_consecutive_errors,
        .backoff_ms = config.backoff_initial_ms,
        .backoff_max_ms = config.backoff_max_ms,
        .max_concurrent = 0,
        .task_queue_kb_id = -1,
        .task_queue_slot = 0,
        .current_backoff_ms = config.backoff_initial_ms,
    };

    return .{ .id = id, .index = idx };
}

pub fn createInternal(self: *RunnerScheduler, config: *const InternalConfig) ?types.RunnerHandle {
    const idx = self.allocSlot() orelse return null;
    const id = self.next_runner_id;
    self.next_runner_id += 1;

    self.runners[@intCast(idx)] = buildRunner(id, .internal, config.session.id, config.interval_ms, 3);
    self.contexts[@intCast(idx)] = .{
        .runner_idx = idx,
        .session_handle = config.session,
        .config_type = .internal,
        .stop_requested = false,
        .kill_requested = false,
        .interval_ms = config.interval_ms,
        .kb_id = config.compute_kb_id,
        .max_turns_recycle = 0,
        .max_errors = 3,
        .backoff_ms = 0,
        .backoff_max_ms = 0,
        .max_concurrent = 0,
        .task_queue_kb_id = -1,
        .task_queue_slot = 0,
        .current_backoff_ms = 0,
    };

    return .{ .id = id, .index = idx };
}

pub fn createBatch(self: *RunnerScheduler, config: *const BatchConfig) ?types.RunnerHandle {
    const idx = self.allocSlot() orelse return null;
    const id = self.next_runner_id;
    self.next_runner_id += 1;

    self.runners[@intCast(idx)] = buildRunner(id, .batch, config.session.id, 0, 3);
    self.contexts[@intCast(idx)] = .{
        .runner_idx = idx,
        .session_handle = config.session,
        .config_type = .batch,
        .stop_requested = false,
        .kill_requested = false,
        .interval_ms = 0,
        .kb_id = -1,
        .max_turns_recycle = 0,
        .max_errors = 3,
        .backoff_ms = 0,
        .backoff_max_ms = 0,
        .max_concurrent = config.max_concurrent,
        .task_queue_kb_id = config.task_queue_kb_id,
        .task_queue_slot = config.task_queue_slot,
        .current_backoff_ms = 0,
    };

    return .{ .id = id, .index = idx };
}

// ============================================================
// Control
// ============================================================

pub fn start(self: *RunnerScheduler, handle: types.RunnerHandle) types.Status {
    const idx: usize = @intCast(handle.index);
    if (idx >= self.runner_active.len or !self.runner_active[idx]) {
        return types.Status.err(.runner, .runner_error_threshold, handle.id);
    }

    self.runners[idx].state = .running;
    self.contexts[idx].stop_requested = false;
    self.contexts[idx].kill_requested = false;

    // Spawn thread for this runner
    const ctx = &self.contexts[idx];
    const runner = &self.runners[idx];
    self.threads[idx] = std.Thread.spawn(.{}, runnerThread, .{ self, ctx, runner }) catch
        return types.Status.err(.runner, .runner_error_threshold, handle.id);

    return types.Status.ok();
}

pub fn stop(self: *RunnerScheduler, handle: types.RunnerHandle) types.Status {
    const idx: usize = @intCast(handle.index);
    if (idx >= self.contexts.len) return types.Status.err(.runner, .runner_error_threshold, handle.id);

    self.contexts[idx].stop_requested = true;

    // Wait for thread to finish
    if (self.threads[idx]) |thread| {
        thread.join();
        self.threads[idx] = null;
    }

    self.runners[idx].state = .stopped;
    return types.Status.ok();
}

pub fn kill(self: *RunnerScheduler, handle: types.RunnerHandle) types.Status {
    const idx: usize = @intCast(handle.index);
    if (idx >= self.contexts.len) return types.Status.err(.runner, .runner_error_threshold, handle.id);

    self.contexts[idx].kill_requested = true;

    if (self.threads[idx]) |thread| {
        thread.join();
        self.threads[idx] = null;
    }

    self.runners[idx].state = .stopped;
    return types.Status.ok();
}

pub fn recycle(self: *RunnerScheduler, handle: types.RunnerHandle) types.Status {
    const idx: usize = @intCast(handle.index);
    if (idx >= self.contexts.len) return types.Status.err(.runner, .runner_error_threshold, handle.id);

    // 1. Stop current iteration
    _ = self.stop(handle);

    // 2. Snapshot session
    // (delegated to session_mgr — snapshot not directly accessible here)

    // 3. Clone session
    const old_session = self.contexts[idx].session_handle;
    const clone_config = session_mod.CloneConfig{};
    const new_session = self.session_mgr.clone(old_session, &clone_config) orelse
        return types.Status.err(.runner, .runner_error_threshold, handle.id);

    // 4. Kill old session
    _ = self.session_mgr.kill(old_session);

    // 5. Point runner at new session
    self.contexts[idx].session_handle = new_session;
    self.runners[idx].recycle_count += 1;
    self.runners[idx].last_recycle_timestamp = currentTimestamp();

    // 6. Restart
    return self.start(handle);
}

// ============================================================
// Status
// ============================================================

pub fn getStatus(self: *RunnerScheduler, handle: types.RunnerHandle) RunnerStatus {
    const idx: usize = @intCast(handle.index);
    if (idx >= self.runners.len or !self.runner_active[idx]) {
        return std.mem.zeroes(RunnerStatus);
    }
    const r = &self.runners[idx];
    return .{
        .state = r.state,
        .runner_type = r.type,
        .iterations_completed = r.iterations_completed,
        .errors_consecutive = r.errors_consecutive,
        .errors_total = r.errors_total,
        .last_iteration_ms = r.last_iteration_ms,
        .last_iteration_timestamp = r.last_iteration_timestamp,
        .recycle_count = r.recycle_count,
    };
}

pub fn listAll(self: *RunnerScheduler, out: []RunnerStatus) i32 {
    var count: i32 = 0;
    for (self.runner_active, 0..) |active, i| {
        if (active and count < @as(i32, @intCast(out.len))) {
            out[@intCast(count)] = self.getStatus(.{ .id = self.runners[i].id, .index = @intCast(i) });
            count += 1;
        }
    }
    return count;
}

// ============================================================
// Runner thread entry points
// ============================================================

fn runnerThread(self: *RunnerScheduler, ctx: *RunnerContext, runner: *types.Runner) void {
    switch (ctx.config_type) {
        .poller => self.runPoller(ctx, runner),
        .processor => self.runProcessor(ctx, runner),
        .internal => self.runInternal(ctx, runner),
        .batch => self.runBatch(ctx, runner),
    }
}

fn runPoller(self: *RunnerScheduler, ctx: *RunnerContext, runner: *types.Runner) void {
    while (!ctx.stop_requested and !ctx.kill_requested) {
        const start_ns = std.time.nanoTimestamp();

        // Fire rules against KB (L3 execution)
        const status = self.inference.executeL3(ctx.session_handle, ctx.kb_id);

        const end_ns = std.time.nanoTimestamp();
        const elapsed_ms: i32 = @intCast(@divTrunc(end_ns - start_ns, 1_000_000));

        if (status.isOk()) {
            runner.iterations_completed += 1;
            runner.errors_consecutive = 0;
            runner.last_iteration_ms = elapsed_ms;
            runner.last_iteration_timestamp = currentTimestamp();
        } else {
            runner.errors_consecutive += 1;
            runner.errors_total += 1;
            if (runner.shouldStop()) {
                runner.state = .err;
                break;
            }
        }

        // Sleep for interval
        if (ctx.interval_ms > 0 and !ctx.stop_requested) {
            std.time.sleep(@intCast(@as(i64, ctx.interval_ms) * 1_000_000));
        }
    }
    runner.state = .stopped;
}

fn runProcessor(self: *RunnerScheduler, ctx: *RunnerContext, runner: *types.Runner) void {
    while (!ctx.stop_requested and !ctx.kill_requested) {
        // Process one turn
        var output = inference_mod.OutputBuffer.init(self.allocator, 4096);
        defer output.deinit(self.allocator);

        const start_ns = std.time.nanoTimestamp();
        const status = self.inference.cycle(ctx.session_handle, "", &output);
        const end_ns = std.time.nanoTimestamp();
        const elapsed_ms: i32 = @intCast(@divTrunc(end_ns - start_ns, 1_000_000));

        if (status.isOk()) {
            runner.iterations_completed += 1;
            runner.errors_consecutive = 0;
            runner.last_iteration_ms = elapsed_ms;
            runner.last_iteration_timestamp = currentTimestamp();
            ctx.current_backoff_ms = ctx.backoff_ms; // reset backoff
        } else {
            runner.errors_consecutive += 1;
            runner.errors_total += 1;

            if (runner.shouldStop()) {
                runner.state = .err;
                break;
            }

            // Exponential backoff on error
            std.time.sleep(@intCast(@as(i64, ctx.current_backoff_ms) * 1_000_000));
            ctx.current_backoff_ms = @min(ctx.current_backoff_ms * 2, ctx.backoff_max_ms);
        }

        // Check recycle threshold
        if (runner.shouldRecycle()) {
            runner.state = .recycling;
            break; // caller (or monitor) will recycle
        }
    }
    if (runner.state != .recycling) runner.state = .stopped;
}

fn runInternal(self: *RunnerScheduler, ctx: *RunnerContext, runner: *types.Runner) void {
    // Same as poller but for internal computations
    self.runPoller(ctx, runner);
}

fn runBatch(self: *RunnerScheduler, ctx: *RunnerContext, runner: *types.Runner) void {
    while (!ctx.stop_requested and !ctx.kill_requested) {
        // Check task queue for pending tasks
        const task = self.inference.kb_store.factRead(ctx.task_queue_kb_id, ctx.task_queue_slot);

        if (task == null or task.?.isEmpty()) {
            // No tasks — sleep and retry
            std.time.sleep(1_000_000_000); // 1 second
            continue;
        }

        // Clone session for isolation
        const clone_config = session_mod.CloneConfig{};
        const clone_handle = self.session_mgr.clone(ctx.session_handle, &clone_config) orelse continue;

        // Process task in clone
        var output = inference_mod.OutputBuffer.init(self.allocator, 4096);
        const status = self.inference.cycle(clone_handle, "", &output);
        output.deinit(self.allocator);

        if (status.isOk()) {
            // Merge clone results back
            _ = self.session_mgr.merge(ctx.session_handle, clone_handle, .theirs);
            runner.iterations_completed += 1;
            runner.errors_consecutive = 0;
        } else {
            runner.errors_consecutive += 1;
            runner.errors_total += 1;
        }

        // Kill clone
        _ = self.session_mgr.kill(clone_handle);

        // Retract processed task
        _ = self.inference.kb_store.factRetract(ctx.task_queue_kb_id, ctx.task_queue_slot);

        runner.last_iteration_timestamp = currentTimestamp();

        if (runner.shouldStop()) {
            runner.state = .err;
            break;
        }
    }
    runner.state = .stopped;
}

// ============================================================
// Helpers
// ============================================================

fn allocSlot(self: *RunnerScheduler) ?i32 {
    for (self.runner_active, 0..) |active, i| {
        if (!active) {
            self.runner_active[i] = true;
            self.runner_count += 1;
            return @intCast(i);
        }
    }
    return null;
}

fn buildRunner(id: i32, runner_type: types.RunnerType, session_id: i32, interval_ms: i32, max_errors: i32) types.Runner {
    var r = std.mem.zeroes(types.Runner);
    r.id = id;
    r.type = runner_type;
    r.state = .stopped;
    r.session_id = session_id;
    r.interval_ms = interval_ms;
    r.max_consecutive_errors = max_errors;
    return r;
}

fn currentTimestamp() i32 {
    const ts = std.time.timestamp();
    return @intCast(@min(ts, std.math.maxInt(i32)));
}
