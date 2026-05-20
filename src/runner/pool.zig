// ============================================================
// src/runner/pool.zig
// ============================================================

pub const MAX_RUNNERS: usize = 64;
pub const MAX_POOL_THREADS: usize = 32;
pub const TASK_QUEUE_CAPACITY: usize = 256;

pub const ThreadPool = struct {
    threads: [MAX_POOL_THREADS]std.Thread,
    n_threads: i32,
    task_queue: TaskQueue,
    shutdown_flag: std.atomic.Value(i32),
    active_count: std.atomic.Value(i32),
    initialized: bool,

    pub fn init(n_threads_requested: i32) ThreadPool {
        const cpu_count = std.Thread.getCpuCount() catch 4;
        const half_cpus: i32 = @intCast(@max(cpu_count / 2, 1));
        const n: i32 = if (n_threads_requested > 0)
            @min(n_threads_requested, @as(i32, MAX_POOL_THREADS))
        else
            @min(half_cpus, @as(i32, MAX_POOL_THREADS));

        return .{
            .threads = undefined,
            .n_threads = n,
            .task_queue = TaskQueue.init(),
            .shutdown_flag = std.atomic.Value(i32).init(0),
            .active_count = std.atomic.Value(i32).init(0),
            .initialized = false,
        };
    }

    pub fn start(self: *ThreadPool, runners: *RunnerTable) void {
        if (self.initialized) return;
        const nt: usize = @intCast(self.n_threads);
        for (0..nt) |i| {
            self.threads[i] = std.Thread.spawn(.{}, workerMain, .{ self, runners }) catch continue;
        }
        self.initialized = true;
    }

    pub fn submit(self: *ThreadPool, task: RunnerTask) bool {
        return self.task_queue.push(task);
    }

    pub fn shutdown(self: *ThreadPool) void {
        self.shutdown_flag.store(1, .seq_cst);
        const nt: usize = @intCast(self.n_threads);
        for (0..nt) |_| {
            _ = self.task_queue.push(.{ .runner_id = -1, .action = .stop });
        }
        if (self.initialized) {
            for (0..nt) |i| {
                self.threads[i].join();
            }
        }
        self.initialized = false;
    }

    pub fn activeCount(self: *const ThreadPool) i32 {
        return self.active_count.load(.seq_cst);
    }
};

fn workerMain(pool: *ThreadPool, runners: *RunnerTable) void {
    while (pool.shutdown_flag.load(.seq_cst) == 0) {
        const maybe_task = pool.task_queue.pop();
        if (maybe_task) |task| {
            if (task.runner_id < 0) continue;
            _ = pool.active_count.fetchAdd(1, .seq_cst);
            executeTask(runners, task);
            _ = pool.active_count.fetchSub(1, .seq_cst);
        } else {
            std.time.sleep(10 * std.time.ns_per_ms);
        }
    }
}

fn executeTask(runners: *RunnerTable, task: RunnerTask) void {
    const idx: usize = @intCast(task.runner_id);
    if (idx >= runners.runners.len) return;
    var runner = &runners.runners[idx];
    if (!runner.active) return;

    switch (task.action) {
        .run_cycle => {
            runner.data.state = .running;
            runSingleIteration(&runner.data, runners);
            if (runner.data.state == .running) {
                runner.data.iterations_completed += 1;
                runner.data.last_iteration_timestamp = timestampNow();
            }
        },
        .recycle => {
            runner.data.state = .recycling;
            runner.data.recycle_count += 1;
            runner.data.last_recycle_timestamp = timestampNow();
            runner.data.state = .running;
        },
        .stop => {
            runner.data.state = .stopped;
        },
        .kill => {
            runner.data.state = .stopped;
        },
    }
}

fn runSingleIteration(runner: *VlpRunner, runners: *RunnerTable) void {
    _ = runners;
    _ = runner;
}

fn timestampNow() i32 {
    return @intCast(@divTrunc(std.time.milliTimestamp(), 1000));
}

pub const TaskQueue = struct {
    buf: [TASK_QUEUE_CAPACITY]RunnerTask,
    head: usize,
    tail: usize,
    count: std.atomic.Value(i32),
    lock: std.Thread.Mutex,

    pub fn init() TaskQueue {
        return .{
            .buf = undefined,
            .head = 0,
            .tail = 0,
            .count = std.atomic.Value(i32).init(0),
            .lock = .{},
        };
    }

    pub fn push(self: *TaskQueue, task: RunnerTask) bool {
        self.lock.lock();
        defer self.lock.unlock();
        if (self.count.load(.seq_cst) >= @as(i32, TASK_QUEUE_CAPACITY)) return false;
        self.buf[self.tail] = task;
        self.tail = (self.tail + 1) % TASK_QUEUE_CAPACITY;
        _ = self.count.fetchAdd(1, .seq_cst);
        return true;
    }

    pub fn pop(self: *TaskQueue) ?RunnerTask {
        self.lock.lock();
        defer self.lock.unlock();
        if (self.count.load(.seq_cst) <= 0) return null;
        const task = self.buf[self.head];
        self.head = (self.head + 1) % TASK_QUEUE_CAPACITY;
        _ = self.count.fetchSub(1, .seq_cst);
        return task;
    }

    pub fn size(self: *const TaskQueue) i32 {
        return self.count.load(.seq_cst);
    }
};

pub const RunnerSlot = struct {
    data: VlpRunner,
    active: bool,
};

pub const RunnerTable = struct {
    runners: [MAX_RUNNERS]RunnerSlot,
    count: i32,

    pub fn init() RunnerTable {
        var table = RunnerTable{
            .runners = undefined,
            .count = 0,
        };
        for (&table.runners) |*slot| {
            slot.active = false;
            slot.data = defaultRunner();
        }
        return table;
    }

    pub fn allocate(self: *RunnerTable) ?i32 {
        for (&self.runners, 0..) |*slot, i| {
            if (!slot.active) {
                slot.active = true;
                slot.data = defaultRunner();
                slot.data.id = @intCast(i);
                self.count += 1;
                return @intCast(i);
            }
        }
        return null;
    }

    pub fn release(self: *RunnerTable, id: i32) void {
        const idx: usize = @intCast(id);
        if (idx >= self.runners.len) return;
        if (self.runners[idx].active) {
            self.runners[idx].active = false;
            self.count -= 1;
        }
    }

    pub fn get(self: *RunnerTable, id: i32) ?*VlpRunner {
        const idx: usize = @intCast(id);
        if (idx >= self.runners.len) return null;
        if (!self.runners[idx].active) return null;
        return &self.runners[idx].data;
    }

    pub fn getConst(self: *const RunnerTable, id: i32) ?*const VlpRunner {
        const idx: usize = @intCast(id);
        if (idx >= self.runners.len) return null;
        if (!self.runners[idx].active) return null;
        return &self.runners[idx].data;
    }
};
