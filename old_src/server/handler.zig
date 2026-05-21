// ============================================================
// src/server/handler.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");
const kb_types = @import("../kb/types.zig");
const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const server_types = @import("types.zig");
const listener = @import("listener.zig");
const auth_mod = @import("auth.zig");
const rate_limit_mod = @import("rate_limit.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;
const VlpFact = kb_types.VlpFact;
const KBStore = kb_store_mod.KBStore;
const Server = server_types.Server;
const ServerConnection = server_types.ServerConnection;
const ServerCredential = server_types.ServerCredential;
const VlpProtocolType = server_types.VlpProtocolType;
const VlpConnectionState = server_types.VlpConnectionState;
const VlpCloseReason = server_types.VlpCloseReason;

pub const Request = struct {
    method: HttpMethod,
    path: [512]u8,
    path_len: i32,
    body: [8192]u8,
    body_len: i32,
    content_length: i32,
    keepalive: bool,
    auth_token: [256]u8,
    auth_token_len: i32,
    content_type: ContentType,
    raw_bytes_len: i32,
};

pub const HttpMethod = enum(i8) {
    get = 0,
    post = 1,
    put = 2,
    delete = 3,
    unknown = -1,
};

pub const ContentType = enum(i8) {
    json = 0,
    form = 1,
    text = 2,
    unknown = -1,
};

pub const Response = struct {
    status_code: i32,
    reason: [32]u8,
    reason_len: i32,
    content_type: ContentType,
    body: [8192]u8,
    body_len: i32,
    buf: [16384]u8,
    buf_len: i32,
};

pub fn defaultRequest() Request {
    return .{
        .method = .unknown,
        .path = undefined,
        .path_len = 0,
        .body = undefined,
        .body_len = 0,
        .content_length = 0,
        .keepalive = false,
        .auth_token = undefined,
        .auth_token_len = 0,
        .content_type = .unknown,
        .raw_bytes_len = 0,
    };
}

pub fn defaultResponse() Response {
    return .{
        .status_code = 200,
        .reason = undefined,
        .reason_len = 0,
        .content_type = .json,
        .body = undefined,
        .body_len = 0,
        .buf = undefined,
        .buf_len = 0,
    };
}

pub fn handleConnection(server: *Server, conn_idx: usize) void {
    var conn = &server.connections[conn_idx];

    var read_result = listener.socketRead(conn.fd, &conn.read_buf, 30000);
    if (read_result.status != .ok) {
        listener.closeConnection(server, conn_idx, .handshake_failed);
        return;
    }
    conn.read_buf_len = read_result.n;
    conn.bytes_received += @intCast(read_result.n);
    conn.state = .authenticating;

    var request = defaultRequest();
    const parse_status = parseHttpRequest(conn.read_buf[0..@intCast(conn.read_buf_len)], &request);
    if (parse_status != .ok) {
        sendErrorResponse(conn, 400, "Bad Request");
        listener.closeConnection(server, conn_idx, .protocol_error);
        return;
    }

    var token_slice: []const u8 = &[_]u8{};
    if (request.auth_token_len > 0) {
        token_slice = request.auth_token[0..@intCast(request.auth_token_len)];
    }

    var auth_result = auth_mod.authenticate(
        server.store,
        server.auth_kb_id,
        token_slice,
        server.config.credential_ttl_seconds,
    );

    if (auth_result.status != .ok) {
        sendErrorResponse(conn, 401, "Unauthorized");
        listener.closeConnection(server, conn_idx, .auth_failed);
        return;
    }

    conn.credential = auth_result.credential;
    conn.state = .active;

    handleRequestLoop(server, conn_idx);
}

fn handleRequestLoop(server: *Server, conn_idx: usize) void {
    var conn = &server.connections[conn_idx];

    while (conn.state == .active and server.shutdown_flag.load(.seq_cst) == 0) {
        if (!auth_mod.credentialCheck(&conn.credential)) {
            sendErrorResponse(conn, 401, "Credential Expired");
            conn.state = .draining;
            break;
        }

        var request = defaultRequest();

        if (conn.requests_served == 0) {
            const parse_status = parseHttpRequest(conn.read_buf[0..@intCast(conn.read_buf_len)], &request);
            if (parse_status != .ok) {
                sendErrorResponse(conn, 400, "Bad Request");
                conn.state = .draining;
                break;
            }
        } else {
            var read_result = listener.socketRead(conn.fd, &conn.read_buf, 30000);
            if (read_result.status != .ok) {
                conn.state = .draining;
                break;
            }
            conn.read_buf_len = read_result.n;
            conn.bytes_received += @intCast(read_result.n);

            const parse_status = parseHttpRequest(conn.read_buf[0..@intCast(conn.read_buf_len)], &request);
            if (parse_status != .ok) {
                sendErrorResponse(conn, 400, "Bad Request");
                conn.state = .draining;
                break;
            }
        }

        const rate_result = rate_limit_mod.checkRateLimit(
            server.store,
            server.auth_kb_id,
            conn.credential.user_id,
        );
        if (!rate_result.allowed) {
            sendRateLimitResponse(conn, rate_result.retry_after_seconds);
            if (!request.keepalive) {
                conn.state = .draining;
                break;
            }
            continue;
        }

        var response = defaultResponse();
        processRequest(server, conn, &request, &response);
        sendResponse(conn, &response);

        conn.requests_served += 1;
        conn.last_active = timestampNow();
        server.metrics.total_requests_served += 1;

        if (!request.keepalive) {
            conn.state = .draining;
            break;
        }
    }

    listener.closeConnection(server, conn_idx, .normal);
}

fn processRequest(server: *Server, conn: *ServerConnection, request: *const Request, response: *Response) void {
    const path_slice = request.path[0..@intCast(request.path_len)];

    if (pathEquals(path_slice, "/health")) {
        processHealthRequest(server, response);
        return;
    }

    if (pathEquals(path_slice, "/metrics")) {
        processMetricsRequest(server, response);
        return;
    }

    if (pathStartsWith(path_slice, "/kb/")) {
        processKbRequest(server, conn, request, response);
        return;
    }

    if (pathEquals(path_slice, "/query")) {
        processQueryRequest(server, conn, request, response);
        return;
    }

    response.status_code = 404;
    setReason(response, "Not Found");
    setBody(response, "{\"error\": \"not found\"}");
}

fn processHealthRequest(server: *Server, response: *Response) void {
    response.status_code = 200;
    setReason(response, "OK");
    response.content_type = .json;

    var body_pos: usize = 0;
    const active = server.n_active.load(.seq_cst);
    body_pos += copyStr(response.body[body_pos..], "{\"status\":\"ok\",\"active_connections\":");
    body_pos += i32ToAscii(active, response.body[body_pos..]);
    body_pos += copyStr(response.body[body_pos..], ",\"total_requests\":");
    body_pos += i64ToAscii(server.metrics.total_requests_served, response.body[body_pos..]);
    body_pos += copyStr(response.body[body_pos..], "}");
    response.body_len = @intCast(body_pos);
}

fn processMetricsRequest(server: *Server, response: *Response) void {
    response.status_code = 200;
    setReason(response, "OK");
    response.content_type = .json;

    var body_pos: usize = 0;
    body_pos += copyStr(response.body[body_pos..], "{\"accepted\":");
    body_pos += i64ToAscii(server.metrics.total_connections_accepted, response.body[body_pos..]);
    body_pos += copyStr(response.body[body_pos..], ",\"rejected\":");
    body_pos += i64ToAscii(server.metrics.total_connections_rejected, response.body[body_pos..]);
    body_pos += copyStr(response.body[body_pos..], ",\"served\":");
    body_pos += i64ToAscii(server.metrics.total_requests_served, response.body[body_pos..]);
    body_pos += copyStr(response.body[body_pos..], ",\"active\":");
    body_pos += i32ToAscii(server.n_active.load(.seq_cst), response.body[body_pos..]);
    body_pos += copyStr(response.body[body_pos..], "}");
    response.body_len = @intCast(body_pos);
}

fn processKbRequest(server: *Server, conn: *ServerConnection, request: *const Request, response: *Response) void {
    const path_slice = request.path[0..@intCast(request.path_len)];
    const kb_path = path_slice[4..];

    var kb_id: i32 = -1;
    const resolve_status = server.store.resolvePath(kb_path, &kb_id);
    if (resolve_status != .ok or kb_id < 0) {
        response.status_code = 404;
        setReason(response, "Not Found");
        setBody(response, "{\"error\": \"kb not found\"}");
        return;
    }

    _ = conn;

    const kb = server.store.getKB(kb_id);
    if (kb == null) {
        response.status_code = 404;
        setReason(response, "Not Found");
        setBody(response, "{\"error\": \"kb not found\"}");
        return;
    }

    response.status_code = 200;
    setReason(response, "OK");
    response.content_type = .json;

    var body_pos: usize = 0;
    body_pos += copyStr(response.body[body_pos..], "{\"kb_id\":");
    body_pos += i32ToAscii(kb_id, response.body[body_pos..]);
    body_pos += copyStr(response.body[body_pos..], ",\"facts_count\":");
    body_pos += i32ToAscii(kb.?.facts_count, response.body[body_pos..]);
    body_pos += copyStr(response.body[body_pos..], ",\"facts_capacity\":");
    body_pos += i32ToAscii(kb.?.facts_capacity, response.body[body_pos..]);
    body_pos += copyStr(response.body[body_pos..], ",\"rules_count\":");
    body_pos += i32ToAscii(kb.?.rules_count, response.body[body_pos..]);
    body_pos += copyStr(response.body[body_pos..], "}");
    response.body_len = @intCast(body_pos);
}

fn processQueryRequest(server: *Server, conn: *ServerConnection, request: *const Request, response: *Response) void {
    _ = server;
    _ = conn;

    if (request.body_len <= 0) {
        response.status_code = 400;
        setReason(response, "Bad Request");
        setBody(response, "{\"error\": \"empty body\"}");
        return;
    }

    response.status_code = 200;
    setReason(response, "OK");
    response.content_type = .json;
    setBody(response, "{\"result\": \"processed\"}");
}

fn sendResponse(conn: *ServerConnection, response: *Response) void {
    var pos: usize = 0;

    pos += copyStr(conn.write_buf[pos..], "HTTP/1.1 ");
    pos += i32ToAscii(response.status_code, conn.write_buf[pos..]);
    pos += copyStr(conn.write_buf[pos..], " ");
    const rl: usize = @intCast(response.reason_len);
    @memcpy(conn.write_buf[pos .. pos + rl], response.reason[0..rl]);
    pos += rl;
    pos += copyStr(conn.write_buf[pos..], "\r\n");

    pos += copyStr(conn.write_buf[pos..], "Content-Type: ");
    switch (response.content_type) {
        .json => pos += copyStr(conn.write_buf[pos..], "application/json"),
        .text => pos += copyStr(conn.write_buf[pos..], "text/plain"),
        else => pos += copyStr(conn.write_buf[pos..], "application/octet-stream"),
    }
    pos += copyStr(conn.write_buf[pos..], "\r\n");

    pos += copyStr(conn.write_buf[pos..], "Content-Length: ");
    pos += i32ToAscii(response.body_len, conn.write_buf[pos..]);
    pos += copyStr(conn.write_buf[pos..], "\r\n");

    pos += copyStr(conn.write_buf[pos..], "Connection: keep-alive\r\n");
    pos += copyStr(conn.write_buf[pos..], "\r\n");

    if (response.body_len > 0) {
        const bl: usize = @intCast(response.body_len);
        @memcpy(conn.write_buf[pos .. pos + bl], response.body[0..bl]);
        pos += bl;
    }

    conn.write_buf_len = @intCast(pos);
    const write_result = listener.socketWrite(conn.fd, conn.write_buf[0..pos]);
    conn.bytes_sent += @intCast(write_result.n);
}

fn sendErrorResponse(conn: *ServerConnection, code: i32, message: []const u8) void {
    var response = defaultResponse();
    response.status_code = code;
    setReason(&response, message);

    var body_pos: usize = 0;
    body_pos += copyStr(response.body[body_pos..], "{\"error\":\"");
    const ml = @min(message.len, response.body.len - body_pos - 3);
    @memcpy(response.body[body_pos .. body_pos + ml], message[0..ml]);
    body_pos += ml;
    body_pos += copyStr(response.body[body_pos..], "\"}");
    response.body_len = @intCast(body_pos);

    sendResponse(conn, &response);
}

fn sendRateLimitResponse(conn: *ServerConnection, retry_after: i32) void {
    var response = defaultResponse();
    response.status_code = 429;
    setReason(&response, "Too Many Requests");

    var body_pos: usize = 0;
    body_pos += copyStr(response.body[body_pos..], "{\"error\":\"rate limited\",\"retry_after\":");
    body_pos += i32ToAscii(retry_after, response.body[body_pos..]);
    body_pos += copyStr(response.body[body_pos..], "}");
    response.body_len = @intCast(body_pos);

    sendResponse(conn, &response);
}

pub fn parseHttpRequest(data: []const u8, request: *Request) VlpStatus {
    if (data.len < 14) return .err_command_parse;

    var pos: usize = 0;

    const method_end = findByte(data, ' ', pos) orelse return .err_command_parse;
    const method_str = data[pos..method_end];
    request.method = parseMethod(method_str);
    if (request.method == .unknown) return .err_command_parse;
    pos = method_end + 1;

    const path_end = findByte(data, ' ', pos) orelse return .err_command_parse;
    const path_len = path_end - pos;
    const copy_path_len = @min(path_len, request.path.len);
    @memcpy(request.path[0..copy_path_len], data[pos .. pos + copy_path_len]);
    request.path_len = @intCast(copy_path_len);
    pos = path_end + 1;

    const line_end = findSequence(data, "\r\n", pos) orelse return .err_command_parse;
    pos = line_end + 2;

    request.keepalive = true;
    request.content_length = 0;
    request.content_type = .unknown;
    request.auth_token_len = 0;

    while (pos < data.len) {
        if (pos + 1 < data.len and data[pos] == '\r' and data[pos + 1] == '\n') {
            pos += 2;
            break;
        }

        const header_end = findSequence(data, "\r\n", pos) orelse break;
        const header_line = data[pos..header_end];
        pos = header_end + 2;

        const colon = findByte(header_line, ':', 0) orelse continue;
        const name = header_line[0..colon];
        var val_start = colon + 1;
        while (val_start < header_line.len and header_line[val_start] == ' ') val_start += 1;
        const value = header_line[val_start..];

        if (caseInsensitiveEql(name, "content-length")) {
            request.content_length = parseI32(value);
        } else if (caseInsensitiveEql(name, "connection")) {
            if (caseInsensitiveEql(value, "close")) {
                request.keepalive = false;
            }
        } else if (caseInsensitiveEql(name, "content-type")) {
            if (containsStr(value, "json")) {
                request.content_type = .json;
            } else if (containsStr(value, "form")) {
                request.content_type = .form;
            } else {
                request.content_type = .text;
            }
        } else if (caseInsensitiveEql(name, "authorization")) {
            var token_start: usize = 0;
            if (value.len > 7 and caseInsensitiveEql(value[0..7], "bearer ")) {
                token_start = 7;
            }
            const token = value[token_start..];
            const tl = @min(token.len, request.auth_token.len);
            @memcpy(request.auth_token[0..tl], token[0..tl]);
            request.auth_token_len = @intCast(tl);
        }
    }

    if (request.content_length > 0 and pos < data.len) {
        const remaining = data.len - pos;
        const copy_body = @min(remaining, @as(usize, @intCast(request.content_length)));
        const body_copy = @min(copy_body, request.body.len);
        @memcpy(request.body[0..body_copy], data[pos .. pos + body_copy]);
        request.body_len = @intCast(body_copy);
    }

    request.raw_bytes_len = @intCast(data.len);
    return .ok;
}

fn parseMethod(s: []const u8) HttpMethod {
    if (s.len == 3 and s[0] == 'G' and s[1] == 'E' and s[2] == 'T') return .get;
    if (s.len == 4 and s[0] == 'P' and s[1] == 'O' and s[2] == 'S' and s[3] == 'T') return .post;
    if (s.len == 3 and s[0] == 'P' and s[1] == 'U' and s[2] == 'T') return .put;
    if (s.len == 6 and s[0] == 'D') return .delete;
    return .unknown;
}

fn findByte(data: []const u8, byte: u8, start: usize) ?usize {
    var i = start;
    while (i < data.len) {
        if (data[i] == byte) return i;
        i += 1;
    }
    return null;
}

fn findSequence(data: []const u8, seq: []const u8, start: usize) ?usize {
    if (seq.len == 0) return start;
    if (data.len < seq.len + start) return null;
    var i = start;
    while (i <= data.len - seq.len) {
        if (std.mem.eql(u8, data[i .. i + seq.len], seq)) return i;
        i += 1;
    }
    return null;
}

fn caseInsensitiveEql(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |ca, cb| {
        const la = if (ca >= 'A' and ca <= 'Z') ca + 32 else ca;
        const lb = if (cb >= 'A' and cb <= 'Z') cb + 32 else cb;
        if (la != lb) return false;
    }
    return true;
}

fn containsStr(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i <= haystack.len - needle.len) {
        if (std.mem.eql(u8, haystack[i .. i + needle.len], needle)) return true;
        i += 1;
    }
    return false;
}

fn parseI32(s: []const u8) i32 {
    var result: i32 = 0;
    for (s) |c| {
        if (c >= '0' and c <= '9') {
            result = result * 10 + @as(i32, c - '0');
        }
    }
    return result;
}

fn pathEquals(path: []const u8, target: []const u8) bool {
    return std.mem.eql(u8, path, target);
}

fn pathStartsWith(path: []const u8, prefix: []const u8) bool {
    if (path.len < prefix.len) return false;
    return std.mem.eql(u8, path[0..prefix.len], prefix);
}

fn setReason(response: *Response, reason: []const u8) void {
    const rl = @min(reason.len, response.reason.len);
    @memcpy(response.reason[0..rl], reason[0..rl]);
    response.reason_len = @intCast(rl);
}

fn setBody(response: *Response, body: []const u8) void {
    const bl = @min(body.len, response.body.len);
    @memcpy(response.body[0..bl], body[0..bl]);
    response.body_len = @intCast(bl);
}

fn copyStr(dest: []u8, src: []const u8) usize {
    const n = @min(src.len, dest.len);
    @memcpy(dest[0..n], src[0..n]);
    return n;
}

fn i32ToAscii(val: i32, output: []u8) usize {
    if (output.len == 0) return 0;
    if (val == 0) {
        output[0] = '0';
        return 1;
    }
    var v: i64 = @intCast(val);
    var pos: usize = 0;
    if (v < 0) {
        output[pos] = '-';
        pos += 1;
        v = -v;
    }
    var buf: [12]u8 = undefined;
    var len: usize = 0;
    while (v > 0) {
        buf[len] = @intCast(@as(u8, @intCast(@mod(v, 10))) + '0');
        len += 1;
        v = @divTrunc(v, 10);
    }
    for (0..len) |i| {
        if (pos >= output.len) break;
        output[pos] = buf[len - 1 - i];
        pos += 1;
    }
    return pos;
}

fn i64ToAscii(val: i64, output: []u8) usize {
    if (output.len == 0) return 0;
    if (val == 0) {
        output[0] = '0';
        return 1;
    }
    var v = val;
    var pos: usize = 0;
    if (v < 0) {
        output[pos] = '-';
        pos += 1;
        v = -v;
    }
    var buf: [20]u8 = undefined;
    var len: usize = 0;
    while (v > 0) {
        buf[len] = @intCast(@as(u8, @intCast(@mod(v, 10))) + '0');
        len += 1;
        v = @divTrunc(v, 10);
    }
    for (0..len) |i| {
        if (pos >= output.len) break;
        output[pos] = buf[len - 1 - i];
        pos += 1;
    }
    return pos;
}

fn timestampNow() i32 {
    return @intCast(@divTrunc(std.time.milliTimestamp(), 1000));
}
