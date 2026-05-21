// ============================================================
// src/protocol/mqtt.zig
// ============================================================

pub const MqttPacketType = enum(u8) {
    connect = 1,
    connack = 2,
    publish = 3,
    puback = 4,
    subscribe = 8,
    suback = 9,
    pingreq = 12,
    pingresp = 13,
    disconnect = 14,
    unknown = 0,
};

pub const MqttConnect = struct {
    client_id: [128]u8,
    client_id_len: i32,
    username: [64]u8,
    username_len: i32,
    password: [64]u8,
    password_len: i32,
    clean_session: bool,
    keepalive: i32,
};

pub const MqttPublish = struct {
    topic: [256]u8,
    topic_len: i32,
    payload: [4096]u8,
    payload_len: i32,
    qos: i32,
    retain: bool,
};

pub fn readConnect(fd: i32, connect: *MqttConnect, timeout_ms: i32) VlpStatus {
    var header: [2]u8 = undefined;
    const h_read = listener.socketRead(fd, &header, timeout_ms);
    if (h_read.status != .ok) return h_read.status;

    const ptype: u8 = (header[0] >> 4) & 0x0F;
    if (ptype != @intFromEnum(MqttPacketType.connect)) return .err_command_parse;

    const remaining: usize = @intCast(header[1]);
    var body: [512]u8 = undefined;
    const body_read = listener.socketRead(fd, body[0..@min(remaining, body.len)], timeout_ms);
    if (body_read.status != .ok) return body_read.status;

    connect.clean_session = true;
    connect.keepalive = 60;
    connect.client_id_len = 0;
    connect.username_len = 0;
    connect.password_len = 0;

    return .ok;
}

pub fn sendConnack(fd: i32, session_present: bool, return_code: u8) VlpStatus {
    var packet: [4]u8 = undefined;
    packet[0] = 0x20;
    packet[1] = 0x02;
    packet[2] = if (session_present) 0x01 else 0x00;
    packet[3] = return_code;
    const r = listener.socketWrite(fd, &packet);
    return r.status;
}

pub fn readPublish(fd: i32, publish: *MqttPublish, timeout_ms: i32) VlpStatus {
    var header: [2]u8 = undefined;
    const h_read = listener.socketRead(fd, &header, timeout_ms);
    if (h_read.status != .ok) return h_read.status;

    const ptype: u8 = (header[0] >> 4) & 0x0F;
    if (ptype != @intFromEnum(MqttPacketType.publish)) return .err_command_parse;

    publish.qos = @intCast((header[0] >> 1) & 0x03);
    publish.retain = (header[0] & 0x01) != 0;

    const remaining: usize = @intCast(header[1]);
    var body: [4096]u8 = undefined;
    const body_read = listener.socketRead(fd, body[0..@min(remaining, body.len)], timeout_ms);
    if (body_read.status != .ok) return body_read.status;
    const bn: usize = @intCast(body_read.n);

    if (bn < 2) return .err_command_parse;
    const topic_len: usize = (@as(usize, body[0]) << 8) | @as(usize, body[1]);
    const tl = @min(topic_len, @min(bn - 2, publish.topic.len));
    if (tl > 0) @memcpy(publish.topic[0..tl], body[2 .. 2 + tl]);
    publish.topic_len = @intCast(tl);

    const payload_start = 2 + topic_len;
    if (payload_start < bn) {
        const pl = @min(bn - payload_start, publish.payload.len);
        @memcpy(publish.payload[0..pl], body[payload_start .. payload_start + pl]);
        publish.payload_len = @intCast(pl);
    } else {
        publish.payload_len = 0;
    }

    return .ok;
}

pub fn sendPingresp(fd: i32) VlpStatus {
    var packet: [2]u8 = .{ 0xD0, 0x00 };
    const r = listener.socketWrite(fd, &packet);
    return r.status;
}

pub fn mqttHandleLoop(server: *server_types.Server, conn_idx: usize) void {
    var conn = &server.connections[conn_idx];
    var connect: MqttConnect = undefined;
    const conn_status = readConnect(conn.fd, &connect, 30000);
    if (conn_status != .ok) {
        listener.closeConnection(server, conn_idx, .handshake_failed);
        return;
    }
    _ = sendConnack(conn.fd, false, 0);
    conn.state = .active;

    while (conn.state == .active and server.shutdown_flag.load(.seq_cst) == 0) {
        var peek: [1]u8 = undefined;
        const peek_read = listener.socketRead(conn.fd, &peek, 60000);
        if (peek_read.status != .ok) {
            conn.state = .draining;
            break;
        }

        const ptype: u8 = (peek[0] >> 4) & 0x0F;
        switch (ptype) {
            @intFromEnum(MqttPacketType.publish) => {
                var publish: MqttPublish = undefined;
                _ = readPublish(conn.fd, &publish, 30000);
                conn.requests_served += 1;
                server.metrics.total_requests_served += 1;
            },
            @intFromEnum(MqttPacketType.pingreq) => {
                var remaining: [1]u8 = undefined;
                _ = listener.socketRead(conn.fd, &remaining, 1000);
                _ = sendPingresp(conn.fd);
            },
            @intFromEnum(MqttPacketType.disconnect) => {
                conn.state = .draining;
            },
            else => {
                var discard: [256]u8 = undefined;
                _ = listener.socketRead(conn.fd, &discard, 1000);
            },
        }

        conn.last_active = timestampNow();
    }

    listener.closeConnection(server, conn_idx, .normal);
}
