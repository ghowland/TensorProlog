// ============================================================
// src/protocol/websocket.zig
// ============================================================

pub const WsOpcode = enum(u8) {
    continuation = 0x0,
    text = 0x1,
    binary = 0x2,
    close = 0x8,
    ping = 0x9,
    pong = 0xA,
    unknown = 0xFF,
};

pub const WsFrame = struct {
    fin: bool,
    opcode: WsOpcode,
    masked: bool,
    mask_key: [4]u8,
    payload: [8192]u8,
    payload_len: i32,
};

pub fn defaultWsFrame() WsFrame {
    return .{
        .fin = true,
        .opcode = .text,
        .masked = false,
        .mask_key = .{ 0, 0, 0, 0 },
        .payload = undefined,
        .payload_len = 0,
    };
}

pub fn wsUpgrade(fd: i32, request: *const HttpRequest) VlpStatus {
    if (!request.upgrade_websocket or request.ws_key_len == 0) return .err_command_parse;

    var accept_key: [64]u8 = undefined;
    const accept_len = computeAcceptKey(request.ws_key[0..@intCast(request.ws_key_len)], &accept_key);

    var resp_buf: [512]u8 = undefined;
    var pos: usize = 0;
    pos += copyStr(resp_buf[pos..], "HTTP/1.1 101 Switching Protocols\r\n");
    pos += copyStr(resp_buf[pos..], "Upgrade: websocket\r\n");
    pos += copyStr(resp_buf[pos..], "Connection: Upgrade\r\n");
    pos += copyStr(resp_buf[pos..], "Sec-WebSocket-Accept: ");
    @memcpy(resp_buf[pos .. pos + accept_len], accept_key[0..accept_len]);
    pos += accept_len;
    pos += copyStr(resp_buf[pos..], "\r\n\r\n");

    const write_result = listener.socketWrite(fd, resp_buf[0..pos]);
    if (write_result.status != .ok) return write_result.status;
    return .ok;
}

fn computeAcceptKey(ws_key: []const u8, output: []u8) usize {
    const magic = "258EAFA5-E914-47DA-95CA-5AB5DC76B45E";
    var combined: [128]u8 = undefined;
    const kl = @min(ws_key.len, 64);
    @memcpy(combined[0..kl], ws_key[0..kl]);
    @memcpy(combined[kl .. kl + magic.len], magic);
    const total = kl + magic.len;

    var hash: u32 = 0;
    for (combined[0..total]) |b| {
        hash = hash *% 31 +% @as(u32, b);
    }

    var hex_buf: [8]u8 = undefined;
    var h = hash;
    for (0..8) |i| {
        const digit: u8 = @intCast(h & 0xF);
        hex_buf[7 - i] = if (digit < 10) digit + '0' else digit - 10 + 'a';
        h >>= 4;
    }
    const out_len = @min(hex_buf.len, output.len);
    @memcpy(output[0..out_len], hex_buf[0..out_len]);
    return out_len;
}

pub fn wsReadFrame(fd: i32, frame: *WsFrame, timeout_ms: i32) VlpStatus {
    var header: [2]u8 = undefined;
    const h_read = listener.socketRead(fd, &header, timeout_ms);
    if (h_read.status != .ok) return h_read.status;
    if (h_read.n < 2) return .err_command_parse;

    frame.fin = (header[0] & 0x80) != 0;
    const opcode_raw = header[0] & 0x0F;
    frame.opcode = switch (opcode_raw) {
        0x0 => .continuation,
        0x1 => .text,
        0x2 => .binary,
        0x8 => .close,
        0x9 => .ping,
        0xA => .pong,
        else => .unknown,
    };

    frame.masked = (header[1] & 0x80) != 0;
    var payload_len: u64 = header[1] & 0x7F;

    if (payload_len == 126) {
        var ext: [2]u8 = undefined;
        const ext_read = listener.socketRead(fd, &ext, timeout_ms);
        if (ext_read.status != .ok) return ext_read.status;
        payload_len = (@as(u64, ext[0]) << 8) | @as(u64, ext[1]);
    } else if (payload_len == 127) {
        var ext: [8]u8 = undefined;
        const ext_read = listener.socketRead(fd, &ext, timeout_ms);
        if (ext_read.status != .ok) return ext_read.status;
        payload_len = 0;
        for (ext) |b| {
            payload_len = (payload_len << 8) | @as(u64, b);
        }
    }

    if (frame.masked) {
        var mask_buf: [4]u8 = undefined;
        const mask_read = listener.socketRead(fd, &mask_buf, timeout_ms);
        if (mask_read.status != .ok) return mask_read.status;
        frame.mask_key = mask_buf;
    }

    const read_len: usize = @intCast(@min(payload_len, frame.payload.len));
    if (read_len > 0) {
        const p_read = listener.socketRead(fd, frame.payload[0..read_len], timeout_ms);
        if (p_read.status != .ok) return p_read.status;
        frame.payload_len = p_read.n;
    } else {
        frame.payload_len = 0;
    }

    if (frame.masked and frame.payload_len > 0) {
        const pl: usize = @intCast(frame.payload_len);
        for (0..pl) |i| {
            frame.payload[i] ^= frame.mask_key[i % 4];
        }
    }

    return .ok;
}

pub fn wsSendText(fd: i32, data: []const u8) VlpStatus {
    return wsSendFrame(fd, .text, data);
}

pub fn wsSendClose(fd: i32, code: u16, reason: []const u8) VlpStatus {
    var payload: [128]u8 = undefined;
    payload[0] = @intCast((code >> 8) & 0xFF);
    payload[1] = @intCast(code & 0xFF);
    const rl = @min(reason.len, payload.len - 2);
    @memcpy(payload[2 .. 2 + rl], reason[0..rl]);
    return wsSendFrame(fd, .close, payload[0 .. 2 + rl]);
}

pub fn wsSendPong(fd: i32, data: []const u8) VlpStatus {
    return wsSendFrame(fd, .pong, data);
}

fn wsSendFrame(fd: i32, opcode: WsOpcode, data: []const u8) VlpStatus {
    var frame_buf: [16384]u8 = undefined;
    var pos: usize = 0;

    frame_buf[pos] = 0x80 | @as(u8, @intFromEnum(opcode));
    pos += 1;

    if (data.len < 126) {
        frame_buf[pos] = @intCast(data.len);
        pos += 1;
    } else if (data.len < 65536) {
        frame_buf[pos] = 126;
        pos += 1;
        frame_buf[pos] = @intCast((data.len >> 8) & 0xFF);
        pos += 1;
        frame_buf[pos] = @intCast(data.len & 0xFF);
        pos += 1;
    } else {
        frame_buf[pos] = 127;
        pos += 1;
        var len = data.len;
        var shift: u6 = 56;
        while (shift > 0) : (shift -= 8) {
            frame_buf[pos] = @intCast((len >> shift) & 0xFF);
            pos += 1;
        }
        frame_buf[pos] = @intCast(len & 0xFF);
        pos += 1;
    }

    const copy_len = @min(data.len, frame_buf.len - pos);
    @memcpy(frame_buf[pos .. pos + copy_len], data[0..copy_len]);
    pos += copy_len;

    const write_result = listener.socketWrite(fd, frame_buf[0..pos]);
    return write_result.status;
}

pub fn wsHandleLoop(server: *server_types.Server, conn_idx: usize) void {
    var conn = &server.connections[conn_idx];

    while (conn.state == .active and server.shutdown_flag.load(.seq_cst) == 0) {
        if (!@import("../server/auth.zig").credentialCheck(&conn.credential)) {
            _ = wsSendClose(conn.fd, 4001, "credential expired");
            conn.state = .draining;
            break;
        }

        var frame = defaultWsFrame();
        const read_status = wsReadFrame(conn.fd, &frame, 60000);

        if (read_status != .ok) {
            if (read_status == .err_clone_failed) {
                conn.state = .draining;
                break;
            }
            continue;
        }

        conn.last_active = timestampNow();

        switch (frame.opcode) {
            .text => {
                const payload = frame.payload[0..@intCast(frame.payload_len)];
                var response_buf: [8192]u8 = undefined;
                const resp_len = processWsMessage(server, conn, payload, &response_buf);
                if (resp_len > 0) {
                    _ = wsSendText(conn.fd, response_buf[0..@intCast(resp_len)]);
                }
                conn.requests_served += 1;
                server.metrics.total_requests_served += 1;
            },
            .binary => {
                _ = wsSendText(conn.fd, "{\"error\":\"binary not supported\"}");
            },
            .ping => {
                _ = wsSendPong(conn.fd, frame.payload[0..@intCast(frame.payload_len)]);
            },
            .close => {
                _ = wsSendClose(conn.fd, 1000, "normal closure");
                conn.state = .draining;
            },
            else => {},
        }
    }

    listener.closeConnection(server, conn_idx, .normal);
}

fn processWsMessage(server: *server_types.Server, conn: *ServerConnection, payload: []const u8, output: []u8) i32 {
    _ = server;
    _ = conn;

    var pos: usize = 0;
    pos += copyStr(output[pos..], "{\"received\":");
    pos += i32ToAscii(@intCast(payload.len), output[pos..]);
    pos += copyStr(output[pos..], ",\"status\":\"ok\"}");
    return @intCast(pos);
}

fn timestampNow() i32 {
    return @intCast(@divTrunc(std.time.milliTimestamp(), 1000));
}
