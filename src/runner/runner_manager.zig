// ============================================================
// src/runner/runner_manager.zig
// ============================================================

pub const RunnerManager = struct {
    table: RunnerTable,
    pool: ThreadPool,

    pub fn init(n_threads: i32) RunnerManager {
        return .{
            .table = RunnerTable.init(),
            .pool = ThreadPool.init(n_threads),
        };
    }

    pub fn start(self: *RunnerManager) void {
        self.pool.start(&self.table);
    }

    pub fn createPoller(self: *RunnerManager, config: PollerConfig, store: *KBStore) ?i32 {
        return poller_mod.createPoller(config, store, &self.table);
    }

    pub fn startRunner(self: *RunnerManager, id: i32) VlpStatus {
        const runner = self.table.get(id) orelse return .err_kb_not_found;
        if (runner.state == .running) return .ok;
        runner.state = .running;
        _ = self.pool.submit(.{ .runner_id = id, .action = .run_cycle });
        return .ok;
    }

    pub fn stopRunner(self: *RunnerManager, id: i32) VlpStatus {
        const runner = self.table.get(id) orelse return .err_kb_not_found;
        runner.state = .stopped;
        _ = self.pool.submit(.{ .runner_id = id, .action = .stop });
        return .ok;
    }

    pub fn killRunner(self: *RunnerManager, id: i32) VlpStatus {
        const runner = self.table.get(id) orelse return .err_kb_not_found;
        runner.state = .stopped;
        return .ok;
    }

    pub fn recycleRunner(self: *RunnerManager, id: i32) VlpStatus {
        const runner = self.table.get(id) orelse return .err_kb_not_found;
        runner.state = .recycling;
        _ = self.pool.submit(.{ .runner_id = id, .action = .recycle });
        return .ok;
    }

    pub fn destroyRunner(self: *RunnerManager, id: i32) void {
        const runner = self.table.get(id);
        if (runner) |r| {
            if (r.state == .running) {
                r.state = .stopped;
            }
        }
        self.table.release(id);
    }

    pub fn getStatus(self: *const RunnerManager, id: i32) ?RunnerStatus {
        const runner = self.table.getConst(id) orelse return null;
        return types_mod.getStatus(runner);
    }

    pub fn shutdown(self: *RunnerManager) void {
        for (&self.table.runners) |*slot| {
            if (slot.active and slot.data.state == .running) {
                slot.data.state = .stopped;
            }
        }
        self.pool.shutdown();
    }

    pub fn activeRunnerCount(self: *const RunnerManager) i32 {
        return self.table.count;
    }
};

const poller_mod = @import("poller.zig");
const types_mod = @import("types.zig");
