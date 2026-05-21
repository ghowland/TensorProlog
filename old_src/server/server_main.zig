// ============================================================
// src/server/server_main.zig
// ============================================================

pub const ServerRuntime = struct {
    server: Server,
    reaper_config: ReaperConfig,
    reaper_thread: ?std.Thread,
    accept_thread: ?std.Thread,
    started: bool,

    pub fn init(config: server_types.ServerConfig, store: *KBStore) ServerRuntime {
        return .{
            .server = server_types.initServer(config, store),
            .reaper_config = defaultReaperConfig(),
            .reaper_thread = null,
            .accept_thread = null,
            .started = false,
        };
    }

    pub fn start(self: *ServerRuntime) VlpStatus {
        if (self.started) return .ok;

        const listen_result = listener.createListenSocket(
            self.server.config.port,
            self.server.config.backlog,
        );
        if (listen_result.status != .ok) return listen_result.status;
        self.server.listen_fd = listen_result.fd;

        self.reaper_config = .{
            .idle_timeout_seconds = self.server.config.idle_timeout_seconds,
            .max_session_turns = self.server.config.max_session_turns,
            .scan_interval_ms = 10000,
        };

        self.accept_thread = std.Thread.spawn(.{}, acceptThreadFn, .{&self.server}) catch return .err_snapshot_failed;
        self.reaper_thread = std.Thread.spawn(.{}, reaperThreadFn, .{ &self.server, self.reaper_config }) catch {
            self.server.shutdown_flag.store(1, .seq_cst);
            return .err_snapshot_failed;
        };

        self.started = true;
        return .ok;
    }

    pub fn stop(self: *ServerRuntime) ShutdownResult {
        const result = gracefulShutdown(&self.server);

        if (self.accept_thread) |t| {
            t.join();
            self.accept_thread = null;
        }

        if (self.reaper_thread) |t| {
            t.join();
            self.reaper_thread = null;
        }

        self.started = false;
        return result;
    }

    pub fn isRunning(self: *const ServerRuntime) bool {
        return self.started and self.server.shutdown_flag.load(.seq_cst) == 0;
    }

    pub fn getHealth(self: *const ServerRuntime) HealthReport {
        return collectHealth(&self.server);
    }

    pub fn getMetrics(self: *const ServerRuntime) server_types.ServerMetrics {
        return self.server.metrics;
    }
};

fn acceptThreadFn(server: *Server) void {
    listener.acceptLoop(server);
}

fn reaperThreadFn(server: *Server, config: ReaperConfig) void {
    reaper_mod.reaperLoop(server, config);
}

const reaper_mod = @import("reaper.zig");
