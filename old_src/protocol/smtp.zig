// ============================================================
// src/protocol/smtp.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const listener = @import("../server/listener.zig");
const server_types = @import("../server/types.zig");

const VlpStatus = types.VlpStatus;

pub const SmtpState = enum(i8) {
    greeting = 0,
    ehlo = 1,
    mail_from = 2,
    rcpt_to = 3,
    data = 4,
    quit = 5,
};

pub const SmtpCommand = struct {
    cmd: SmtpVerb,
    arg: [256]u8,
    arg_len: i32,
};

pub const SmtpVerb = enum(i8) {
    ehlo = 0,
    helo = 1,
    mail = 2,
    rcpt = 3,
    data = 4,
    quit = 5,
    rset = 6,
    noop = 7,
    auth = 8,
    unknown = -1,
};

pub fn sendGreeting(fd: i32, hostname: []const u8) VlpStatus {
    var buf: [256]u8 = undefined;
    var pos: usize = 0;
    pos += copyStr(buf[pos..], "220 ");
    pos += copyStr(buf[pos..], hostname);
    pos += copyStr(buf[pos..], " ESMTP VDR-LLM-Prolog\r\n");
    const r = listener.socketWrite(fd, buf[0..pos]);
    return r.status;
}

pub fn readCommand(fd: i32, cmd: *SmtpCommand, timeout_ms: i32) VlpStatus {
    var line: [512]u8 = undefined;
    const read_result = listener.socketRead(fd, &line, timeout_ms);
    if (read_result.status != .ok) return read_result.status;
    if (read_result.n < 4) return .err_command_parse;
    const n: usize = @intCast(read_result.n);
    cmd.cmd = parseSmtpVerb(line[0..@min(n, 4)]);
    var arg_start: usize = 4;
    while (arg_start < n and (line[arg_start] == ' ' or line[arg_start] == ':')) arg_start += 1;
    var arg_end = n;
    while (arg_end > arg_start and (line[arg_end - 1] == '\r' or line[arg_end - 1] == '\n')) arg_end -= 1;
    const al = @min(arg_end - arg_start, cmd.arg.len);
    if (al > 0) @memcpy(cmd.arg[0..al], line[arg_start .. arg_start + al]);
    cmd.arg_len = @intCast(al);
    return .ok;
}

pub fn sendResponse(fd: i32, code: i32, message: []const u8) VlpStatus {
    var buf: [512]u8 = undefined;
    var pos: usize = 0;
    pos += i32ToAscii(code, buf[pos..]);
    pos += copyStr(buf[pos..], " ");
    pos += copyStr(buf[pos..], message);
    pos += copyStr(buf[pos..], "\r\n");
    const r = listener.socketWrite(fd, buf[0..pos]);
    return r.status;
}

fn parseSmtpVerb(s: []const u8) SmtpVerb {
    if (s.len < 4) return .unknown;
    if (ciEql4(s, "EHLO")) return .ehlo;
    if (ciEql4(s, "HELO")) return .helo;
    if (ciEql4(s, "MAIL")) return .mail;
    if (ciEql4(s, "RCPT")) return .rcpt;
    if (ciEql4(s, "DATA")) return .data;
    if (ciEql4(s, "QUIT")) return .quit;
    if (ciEql4(s, "RSET")) return .rset;
    if (ciEql4(s, "NOOP")) return .noop;
    if (ciEql4(s, "AUTH")) return .auth;
    return .unknown;
}

fn ciEql4(a: []const u8, b: []const u8) bool {
    if (a.len < 4 or b.len < 4) return false;
    for (0..4) |i| {
        const ca = if (a[i] >= 'a' and a[i] <= 'z') a[i] - 32 else a[i];
        const cb = if (b[i] >= 'a' and b[i] <= 'z') b[i] - 32 else b[i];
        if (ca != cb) return false;
    }
    return true;
}

pub fn smtpHandleLoop(server: *server_types.Server, conn_idx: usize) void {
    var conn = &server.connections[conn_idx];
    _ = sendGreeting(conn.fd, "localhost");
    conn.state = .active;

    var smtp_state: SmtpState = .greeting;

    while (conn.state == .active and server.shutdown_flag.load(.seq_cst) == 0) {
        var cmd: SmtpCommand = undefined;
        const read_status = readCommand(conn.fd, &cmd, 60000);
        if (read_status != .ok) {
            conn.state = .draining;
            break;
        }

        switch (cmd.cmd) {
            .ehlo, .helo => {
                _ = sendResponse(conn.fd, 250, "OK");
                smtp_state = .ehlo;
            },
            .mail => {
                if (smtp_state == .ehlo or smtp_state == .mail_from) {
                    _ = sendResponse(conn.fd, 250, "OK");
                    smtp_state = .mail_from;
                } else {
                    _ = sendResponse(conn.fd, 503, "Bad sequence");
                }
            },
            .rcpt => {
                if (smtp_state == .mail_from or smtp_state == .rcpt_to) {
                    _ = sendResponse(conn.fd, 250, "OK");
                    smtp_state = .rcpt_to;
                } else {
                    _ = sendResponse(conn.fd, 503, "Bad sequence");
                }
            },
            .data => {
                if (smtp_state == .rcpt_to) {
                    _ = sendResponse(conn.fd, 354, "Start mail input");
                    smtp_state = .data;
                    var data_buf: [8192]u8 = undefined;
                    _ = listener.socketRead(conn.fd, &data_buf, 60000);
                    _ = sendResponse(conn.fd, 250, "OK");
                    smtp_state = .ehlo;
                    conn.requests_served += 1;
                    server.metrics.total_requests_served += 1;
                } else {
                    _ = sendResponse(conn.fd, 503, "Bad sequence");
                }
            },
            .quit => {
                _ = sendResponse(conn.fd, 221, "Bye");
                conn.state = .draining;
            },
            .rset => {
                _ = sendResponse(conn.fd, 250, "OK");
                smtp_state = .ehlo;
            },
            .noop => {
                _ = sendResponse(conn.fd, 250, "OK");
            },
            else => {
                _ = sendResponse(conn.fd, 502, "Command not implemented");
            },
        }

        conn.last_active = timestampNow();
    }

    listener.closeConnection(server, conn_idx, .normal);
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

fn timestampNow() i32 {
    return @intCast(@divTrunc(std.time.milliTimestamp(), 1000));
}
