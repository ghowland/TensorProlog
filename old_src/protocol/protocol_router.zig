// ============================================================
// src/protocol/protocol_router.zig
// ============================================================

pub fn routeConnection(server: *server_types.Server, conn_idx: usize) void {
    var conn = &server.connections[conn_idx];

    var peek_buf: [4]u8 = undefined;
    const peek_result = listener.socketRead(conn.fd, &peek_buf, 5000);
    if (peek_result.status != .ok or peek_result.n == 0) {
        listener.closeConnection(server, conn_idx, .handshake_failed);
        return;
    }

    switch (server.config.protocol) {
        .http => {
            handler_mod.handleConnection(server, conn_idx);
        },
        .websocket => {
            var req = defaultHttpRequest();
            const parse_status = parseRequest(conn.read_buf[0..@intCast(conn.read_buf_len)], &req);
            if (parse_status == .ok and req.upgrade_websocket) {
                const upgrade_status = wsUpgrade(conn.fd, &req);
                if (upgrade_status == .ok) {
                    conn.state = .active;
                    wsHandleLoop(server, conn_idx);
                    return;
                }
            }
            handler_mod.handleConnection(server, conn_idx);
        },
        else => {
            handler_mod.handleConnection(server, conn_idx);
        },
    }
}
