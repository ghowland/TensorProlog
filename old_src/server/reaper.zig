// ============================================================
// src/server/reaper.zig
// ============================================================

const listener = @import("listener.zig");
const auth_mod = @import("auth.zig");

pub const ReaperConfig = struct {
    idle_timeout_seconds: i32,
    max_session_turns: i32,
    scan_interval_ms: i32,
};

pub fn defaultReaperConfig() ReaperConfig {
    return .{
        .idle_timeout_seconds = 300,
        .max_session_turns = 1000,
        .scan_interval_ms = 10000,
    };
}

pub const ReaperStats = struct {
    idle_closed: i32,
    expired_closed: i32,
    recycled: i32,
};

pub fn reaperScan(server: *Server, config: ReaperConfig) ReaperStats {
    var stats = ReaperStats{
        .idle_closed = 0,
        .expired_closed = 0,
        .recycled = 0,
    };

    const now = timestampNow();

    for (&server.connections, 0..) |*conn, i| {
        if (conn.state != .active) continue;

        const idle_seconds = now - conn.last_active;
        if (idle_seconds >= config.idle_timeout_seconds) {
            sendTimeoutNotice(conn);
            listener.closeConnection(server, i, .idle_timeout);
            stats.idle_closed += 1;
            continue;
        }

        if (conn.credential.expires_at > 0 and now >= conn.credential.expires_at) {
            conn.credential.valid = false;
            sendExpiredNotice(conn);
            listener.closeConnection(server, i, .credential_expired);
            stats.expired_closed += 1;
            continue;
        }

        if (config.max_session_turns > 0 and conn.requests_served >= config.max_session_turns) {
            stats.recycled += 1;
        }
    }

    return stats;
}

fn sendTimeoutNotice(conn: *server_types.ServerConnection) void {
    const msg = "HTTP/1.1 408 Request Timeout\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    _ = listener.socketWrite(conn.fd, msg);
}

fn sendExpiredNotice(conn: *server_types.ServerConnection) void {
    const msg = "HTTP/1.1 401 Unauthorized\r\nContent-Length: 30\r\nConnection: close\r\n\r\n{\"error\":\"credential expired\"}";
    _ = listener.socketWrite(conn.fd, msg);
}

pub fn reaperLoop(server: *Server, config: ReaperConfig) void {
    while (server.shutdown_flag.load(.seq_cst) == 0) {
        _ = reaperScan(server, config);
        const sleep_ns: u64 = @intCast(@as(i64, config.scan_interval_ms) * std.time.ns_per_ms);
        std.time.sleep(sleep_ns);
    }
}

fn timestampNow() i32 {
    return @intCast(@divTrunc(std.time.milliTimestamp(), 1000));
}
