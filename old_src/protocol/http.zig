// ============================================================
// src/protocol/http.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");
const server_types = @import("../server/types.zig");
const listener = @import("../server/listener.zig");
const handler_mod = @import("../server/handler.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;
const ServerConnection = server_types.ServerConnection;

pub const HttpRequest = struct {
    method: handler_mod.HttpMethod,
    path: [512]u8,
    path_len: i32,
    version_major: i32,
    version_minor: i32,
    headers: [32]HttpHeader,
    header_count: i32,
    body: [8192]u8,
    body_len: i32,
    content_length: i32,
    keepalive: bool,
    auth_token: [256]u8,
    auth_token_len: i32,
    content_type: handler_mod.ContentType,
    host: [128]u8,
    host_len: i32,
    upgrade_websocket: bool,
    ws_key: [64]u8,
    ws_key_len: i32,
};

pub const HttpHeader = struct {
    name: [64]u8,
    name_len: i32,
    value: [256]u8,
    value_len: i32,
};

pub const HttpResponse = struct {
    status_code: i32,
    reason: [32]u8,
    reason_len: i32,
    headers: [16]HttpHeader,
    header_count: i32,
    body: [8192]u8,
    body_len: i32,
};

pub fn defaultHttpRequest() HttpRequest {
    return .{
        .method = .unknown,
        .path = undefined,
        .path_len = 0,
        .version_major = 1,
        .version_minor = 1,
        .headers = undefined,
        .header_count = 0,
        .body = undefined,
        .body_len = 0,
        .content_length = 0,
        .keepalive = true,
        .auth_token = undefined,
        .auth_token_len = 0,
        .content_type = .unknown,
        .host = undefined,
        .host_len = 0,
        .upgrade_websocket = false,
        .ws_key = undefined,
        .ws_key_len = 0,
    };
}

pub fn defaultHttpResponse() HttpResponse {
    return .{
        .status_code = 200,
        .reason = undefined,
        .reason_len = 0,
        .headers = undefined,
        .header_count = 0,
        .body = undefined,
        .body_len = 0,
    };
}

pub fn parseRequest(data: []const u8, request: *HttpRequest) VlpStatus {
    if (data.len < 14) return .err_command_parse;
    var pos: usize = 0;

    const method_end = findByte(data, ' ', pos) orelse return .err_command_parse;
    request.method = parseMethod(data[pos..method_end]);
    if (request.method == .unknown) return .err_command_parse;
    pos = method_end + 1;

    const path_end = findByte(data, ' ', pos) orelse return .err_command_parse;
    const pl = @min(path_end - pos, request.path.len);
    @memcpy(request.path[0..pl], data[pos .. pos + pl]);
    request.path_len = @intCast(pl);
    pos = path_end + 1;

    const version_end = findSeq(data, "\r\n", pos) orelse return .err_command_parse;
    const version_str = data[pos..version_end];
    if (version_str.len >= 8) {
        request.version_major = if (version_str[5] >= '0' and version_str[5] <= '9') @as(i32, version_str[5] - '0') else 1;
        request.version_minor = if (version_str[7] >= '0' and version_str[7] <= '9') @as(i32, version_str[7] - '0') else 1;
    }
    pos = version_end + 2;

    request.header_count = 0;
    request.keepalive = (request.version_minor >= 1);
    request.upgrade_websocket = false;

    while (pos < data.len) {
        if (pos + 1 < data.len and data[pos] == '\r' and data[pos + 1] == '\n') {
            pos += 2;
            break;
        }
        const hdr_end = findSeq(data, "\r\n", pos) orelse break;
        const hdr_line = data[pos..hdr_end];
        pos = hdr_end + 2;

        const colon = findByte(hdr_line, ':', 0) orelse continue;
        const name = hdr_line[0..colon];
        var vs: usize = colon + 1;
        while (vs < hdr_line.len and hdr_line[vs] == ' ') vs += 1;
        const value = hdr_line[vs..];

        if (request.header_count < 32) {
            const hi: usize = @intCast(request.header_count);
            const nl = @min(name.len, request.headers[hi].name.len);
            @memcpy(request.headers[hi].name[0..nl], name[0..nl]);
            request.headers[hi].name_len = @intCast(nl);
            const vl = @min(value.len, request.headers[hi].value.len);
            @memcpy(request.headers[hi].value[0..vl], value[0..vl]);
            request.headers[hi].value_len = @intCast(vl);
            request.header_count += 1;
        }

        if (ciEql(name, "content-length")) {
            request.content_length = parseI32(value);
        } else if (ciEql(name, "connection")) {
            if (ciEql(value, "close")) request.keepalive = false;
            if (containsCI(value, "upgrade")) request.upgrade_websocket = true;
        } else if (ciEql(name, "content-type")) {
            if (containsCI(value, "json")) {
                request.content_type = .json;
            } else if (containsCI(value, "form")) {
                request.content_type = .form;
            } else {
                request.content_type = .text;
            }
        } else if (ciEql(name, "authorization")) {
            var ts: usize = 0;
            if (value.len > 7 and ciEql(value[0..7], "bearer ")) ts = 7;
            const token = value[ts..];
            const tl = @min(token.len, request.auth_token.len);
            @memcpy(request.auth_token[0..tl], token[0..tl]);
            request.auth_token_len = @intCast(tl);
        } else if (ciEql(name, "host")) {
            const hl = @min(value.len, request.host.len);
            @memcpy(request.host[0..hl], value[0..hl]);
            request.host_len = @intCast(hl);
        } else if (ciEql(name, "upgrade")) {
            if (ciEql(value, "websocket")) request.upgrade_websocket = true;
        } else if (ciEql(name, "sec-websocket-key")) {
            const kl = @min(value.len, request.ws_key.len);
            @memcpy(request.ws_key[0..kl], value[0..kl]);
            request.ws_key_len = @intCast(kl);
        }
    }

    if (request.content_length > 0 and pos < data.len) {
        const remaining = data.len - pos;
        const cl: usize = @intCast(request.content_length);
        const copy_len = @min(@min(remaining, cl), request.body.len);
        @memcpy(request.body[0..copy_len], data[pos .. pos + copy_len]);
        request.body_len = @intCast(copy_len);
    }

    return .ok;
}

pub fn buildResponse(response: *const HttpResponse, output: []u8) i32 {
    var pos: usize = 0;

    pos += copyStr(output[pos..], "HTTP/1.1 ");
    pos += i32ToAscii(response.status_code, output[pos..]);
    pos += copyStr(output[pos..], " ");
    const rl: usize = @intCast(response.reason_len);
    if (rl > 0) {
        @memcpy(output[pos .. pos + rl], response.reason[0..rl]);
        pos += rl;
    }
    pos += copyStr(output[pos..], "\r\n");

    const hc: usize = @intCast(response.header_count);
    for (0..hc) |i| {
        const nl: usize = @intCast(response.headers[i].name_len);
        const vl: usize = @intCast(response.headers[i].value_len);
        @memcpy(output[pos .. pos + nl], response.headers[i].name[0..nl]);
        pos += nl;
        pos += copyStr(output[pos..], ": ");
        @memcpy(output[pos .. pos + vl], response.headers[i].value[0..vl]);
        pos += vl;
        pos += copyStr(output[pos..], "\r\n");
    }

    pos += copyStr(output[pos..], "Content-Length: ");
    pos += i32ToAscii(response.body_len, output[pos..]);
    pos += copyStr(output[pos..], "\r\n\r\n");

    if (response.body_len > 0) {
        const bl: usize = @intCast(response.body_len);
        const copy_bl = @min(bl, output.len - pos);
        @memcpy(output[pos .. pos + copy_bl], response.body[0..copy_bl]);
        pos += copy_bl;
    }

    return @intCast(pos);
}

pub fn sendHttpResponse(fd: i32, response: *const HttpResponse) VlpStatus {
    var buf: [16384]u8 = undefined;
    const len = buildResponse(response, &buf);
    const write_result = listener.socketWrite(fd, buf[0..@intCast(len)]);
    if (write_result.status != .ok) return write_result.status;
    return .ok;
}

pub fn sendHttpError(fd: i32, code: i32, reason: []const u8, body: []const u8) void {
    var resp = defaultHttpResponse();
    resp.status_code = code;
    const rn = @min(reason.len, resp.reason.len);
    @memcpy(resp.reason[0..rn], reason[0..rn]);
    resp.reason_len = @intCast(rn);
    const bl = @min(body.len, resp.body.len);
    @memcpy(resp.body[0..bl], body[0..bl]);
    resp.body_len = @intCast(bl);
    _ = sendHttpResponse(fd, &resp);
}

fn parseMethod(s: []const u8) handler_mod.HttpMethod {
    if (s.len == 3 and s[0] == 'G' and s[1] == 'E' and s[2] == 'T') return .get;
    if (s.len == 4 and s[0] == 'P' and s[1] == 'O' and s[2] == 'S' and s[3] == 'T') return .post;
    if (s.len == 3 and s[0] == 'P' and s[1] == 'U' and s[2] == 'T') return .put;
    if (s.len == 6 and s[0] == 'D') return .delete;
    return .unknown;
}

fn findByte(data: []const u8, byte: u8, start: usize) ?usize {
    var i = start;
    while (i < data.len) : (i += 1) {
        if (data[i] == byte) return i;
    }
    return null;
}

fn findSeq(data: []const u8, seq: []const u8, start: usize) ?usize {
    if (seq.len == 0) return start;
    if (data.len < seq.len + start) return null;
    var i = start;
    while (i <= data.len - seq.len) : (i += 1) {
        if (std.mem.eql(u8, data[i .. i + seq.len], seq)) return i;
    }
    return null;
}

fn ciEql(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |ca, cb| {
        const la = if (ca >= 'A' and ca <= 'Z') ca + 32 else ca;
        const lb = if (cb >= 'A' and cb <= 'Z') cb + 32 else cb;
        if (la != lb) return false;
    }
    return true;
}

fn containsCI(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i <= haystack.len - needle.len) : (i += 1) {
        if (ciEql(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

fn parseI32(s: []const u8) i32 {
    var result: i32 = 0;
    for (s) |c| {
        if (c >= '0' and c <= '9') result = result * 10 + @as(i32, c - '0');
    }
    return result;
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
