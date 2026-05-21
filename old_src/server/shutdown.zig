// ============================================================
// src/server/shutdown.zig
// ============================================================

const snapshot_mod = @import("../session/snapshot.zig");
const audit_mod = @import("../safety/audit.zig");
const kb_store_mod = @import("../kb/store.zig");
const KBStore = kb_store_mod.KBStore;

pub const ShutdownResult = struct {
    connections_drained: i32,
    connections_forced: i32,
    sessions_snapshotted: i32,
    status: VlpStatus,
};

pub fn gracefulShutdown(server: *Server) ShutdownResult {
    var result = ShutdownResult{
        .connections_drained = 0,
        .connections_forced = 0,
        .sessions_snapshotted = 0,
        .status = .ok,
    };

    server.shutdown_flag.store(1, .seq_cst);

    if (server.listen_fd >= 0) {
        listener.closeSocket(server.listen_fd);
        server.listen_fd = -1;
    }

    for (&server.connections) |*conn| {
        if (conn.state == .active) {
            conn.state = .draining;
            sendShutdownNotice(conn);
        }
    }

    const deadline = timestampNow() + server.config.shutdown_timeout_seconds;
    while (server.n_active.load(.seq_cst) > 0 and timestampNow() < deadline) {
        std.time.sleep(100 * std.time.ns_per_ms);

        for (&server.connections, 0..) |*conn, i| {
            if (conn.state == .draining) {
                listener.closeConnection(server, i, .shutdown);
                result.connections_drained += 1;
            }
        }
    }

    for (&server.connections, 0..) |*conn, i| {
        if (conn.state != .closed) {
            if (server.config.persistent_sessions and conn.session_id >= 0) {
                var snap_buf: [65536]u8 = undefined;
                var snap_len: i32 = 0;
                const snap_status = snapshot_mod.snapshotSave(
                    server.store,
                    conn.session_id,
                    &snap_buf,
                    @intCast(snap_buf.len),
                    &snap_len,
                );
                if (snap_status == .ok) {
                    result.sessions_snapshotted += 1;
                }
            }
            listener.closeConnection(server, i, .shutdown);
            result.connections_forced += 1;
        }
    }

    return result;
}

fn sendShutdownNotice(conn: *server_types.ServerConnection) void {
    const msg = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 28\r\nConnection: close\r\n\r\n{\"error\":\"server shutting down\"}";
    _ = listener.socketWrite(conn.fd, msg);
}
