// ============================================================
// src/server/listener.zig
// ============================================================

const kb_store_mod = @import("../kb/store.zig");
const KBStore = kb_store_mod.KBStore;

pub const ListenResult = struct {
    status: VlpStatus,
    fd: i32,
};

pub fn createListenSocket(port: i32, backlog: i32) ListenResult {
    const fd = std.posix.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0) catch {
        return .{ .status = .err_snapshot_failed, .fd = -1 };
    };

    const fd_i32: i32 = @intCast(fd);

    setReuseAddr(fd) catch {};

    const port_u16: u16 = @intCast(@as(u32, @bitCast(port)) & 0xFFFF);
    const addr = std.net.Address.initIp4(.{ 0, 0, 0, 0 }, port_u16);

    std.posix.bind(fd, &addr.any, addr.getOsSockLen()) catch {
        std.posix.close(fd);
        return .{ .status = .err_snapshot_failed, .fd = -1 };
    };

    const bl: u31 = @intCast(@as(u32, @bitCast(@max(backlog, 1))) & 0x7FFFFFFF);
    std.posix.listen(fd, bl) catch {
        std.posix.close(fd);
        return .{ .status = .err_snapshot_failed, .fd = -1 };
    };

    return .{ .status = .ok, .fd = fd_i32 };
}

fn setReuseAddr(fd: std.posix.socket_t) !void {
    const val: i32 = 1;
    const val_bytes = std.mem.asBytes(&val);
    try std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, val_bytes);
}

pub fn acceptConnection(listen_fd: i32) ?i32 {
    const fd: std.posix.socket_t = @intCast(@as(u32, @bitCast(listen_fd)));
    const result = std.posix.accept(fd, null, null) catch return null;
    return @intCast(result);
}

pub fn closeSocket(fd: i32) void {
    if (fd < 0) return;
    const sock: std.posix.socket_t = @intCast(@as(u32, @bitCast(fd)));
    std.posix.close(sock);
}

pub fn socketRead(fd: i32, buf: []u8, timeout_ms: i32) struct { n: i32, status: VlpStatus } {
    _ = timeout_ms;
    const sock: std.posix.socket_t = @intCast(@as(u32, @bitCast(fd)));
    const n = std.posix.read(sock, buf) catch return .{ .n = 0, .status = .err_snapshot_failed };
    if (n == 0) return .{ .n = 0, .status = .err_clone_failed };
    return .{ .n = @intCast(n), .status = .ok };
}

pub fn socketWrite(fd: i32, data: []const u8) struct { n: i32, status: VlpStatus } {
    const sock: std.posix.socket_t = @intCast(@as(u32, @bitCast(fd)));
    const n = std.posix.write(sock, data) catch return .{ .n = 0, .status = .err_snapshot_failed };
    return .{ .n = @intCast(n), .status = .ok };
}

pub fn findFreeSlot(server: *Server) ?usize {
    for (&server.connections, 0..) |*conn, i| {
        if (conn.state == .closed) return i;
    }
    return null;
}

pub fn acceptLoop(server: *Server) void {
    while (server.shutdown_flag.load(.seq_cst) == 0) {
        const maybe_fd = acceptConnection(server.listen_fd);
        if (maybe_fd == null) {
            std.time.sleep(10 * std.time.ns_per_ms);
            continue;
        }
        const client_fd = maybe_fd.?;

        if (server.n_active.load(.seq_cst) >= server.config.max_connections) {
            sendReject(client_fd, server.config.protocol);
            closeSocket(client_fd);
            server.metrics.total_connections_rejected += 1;
            continue;
        }

        const slot_idx = findFreeSlot(server);
        if (slot_idx == null) {
            sendReject(client_fd, server.config.protocol);
            closeSocket(client_fd);
            server.metrics.total_connections_rejected += 1;
            continue;
        }

        const idx = slot_idx.?;
        server.connections[idx] = defaultConnection();
        server.connections[idx].fd = client_fd;
        server.connections[idx].state = .handshake;
        server.connections[idx].created_at = timestampNow();
        server.connections[idx].last_active = server.connections[idx].created_at;

        _ = server.n_active.fetchAdd(1, .seq_cst);
        server.metrics.total_connections_accepted += 1;
    }
}

fn sendReject(fd: i32, protocol: VlpProtocolType) void {
    switch (protocol) {
        .http => {
            const resp = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
            _ = socketWrite(fd, resp);
        },
        .websocket => {
            const resp = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
            _ = socketWrite(fd, resp);
        },
        else => {},
    }
}

pub fn closeConnection(server: *Server, idx: usize, reason: VlpCloseReason) void {
    var conn = &server.connections[idx];
    if (conn.state == .closed) return;

    _ = reason;

    if (conn.fd >= 0) {
        closeSocket(conn.fd);
        conn.fd = -1;
    }

    conn.state = .closed;
    conn.session_id = -1;
    conn.credential.valid = false;

    _ = server.n_active.fetchSub(1, .seq_cst);
}

fn timestampNow() i32 {
    return @intCast(@divTrunc(std.time.milliTimestamp(), 1000));
}
