// ============================================================
// src/server/types.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");
const kb_types = @import("../kb/types.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;

pub const VlpProtocolType = enum(i8) {
    http = 0,
    websocket = 1,
    smtp = 2,
    mqtt = 3,
    raw_tcp = 4,
};

pub const VlpConnectionState = enum(i8) {
    closed = 0,
    handshake = 1,
    authenticating = 2,
    active = 3,
    draining = 4,
};

pub const VlpCloseReason = enum(i8) {
    normal = 0,
    idle_timeout = 1,
    credential_expired = 2,
    auth_failed = 3,
    handshake_failed = 4,
    protocol_error = 5,
    shutdown = 6,
    capacity = 7,
};

pub const MAX_CONNECTIONS: usize = 256;

pub const ServerCredential = struct {
    user_id: i32,
    visibility_level: i8,
    grants: [16]CredentialGrant,
    n_grants: i32,
    issued_at: i32,
    expires_at: i32,
    valid: bool,
};

pub const CredentialGrant = struct {
    class: i8,
    target_hash: i32,
    remaining_uses: i32,
    expires_at: i32,
};

pub fn defaultCredential() ServerCredential {
    return .{
        .user_id = -1,
        .visibility_level = 0,
        .grants = undefined,
        .n_grants = 0,
        .issued_at = 0,
        .expires_at = 0,
        .valid = false,
    };
}

pub const ServerConnection = struct {
    fd: i32,
    session_id: i32,
    credential: ServerCredential,
    state: VlpConnectionState,
    created_at: i32,
    last_active: i32,
    requests_served: i32,
    bytes_received: i64,
    bytes_sent: i64,
    read_buf: [8192]u8,
    read_buf_len: i32,
    write_buf: [8192]u8,
    write_buf_len: i32,
};

pub fn defaultConnection() ServerConnection {
    return .{
        .fd = -1,
        .session_id = -1,
        .credential = defaultCredential(),
        .state = .closed,
        .created_at = 0,
        .last_active = 0,
        .requests_served = 0,
        .bytes_received = 0,
        .bytes_sent = 0,
        .read_buf = undefined,
        .read_buf_len = 0,
        .write_buf = undefined,
        .write_buf_len = 0,
    };
}

pub const ServerConfig = struct {
    port: i32,
    address: [64]u8,
    address_len: i32,
    protocol: VlpProtocolType,
    max_connections: i32,
    credential_ttl_seconds: i32,
    max_session_turns: i32,
    idle_timeout_seconds: i32,
    shutdown_timeout_seconds: i32,
    persistent_sessions: bool,
    template_snapshot_path: [256]u8,
    template_snapshot_path_len: i32,
    auth_kb_id: i32,
    protocol_grammar_kb_id: i32,
    domain_kb_id: i32,
    backlog: i32,
};

pub fn defaultServerConfig() ServerConfig {
    var cfg = ServerConfig{
        .port = 8080,
        .address = undefined,
        .address_len = 0,
        .protocol = .http,
        .max_connections = 64,
        .credential_ttl_seconds = 3600,
        .max_session_turns = 1000,
        .idle_timeout_seconds = 300,
        .shutdown_timeout_seconds = 30,
        .persistent_sessions = false,
        .template_snapshot_path = undefined,
        .template_snapshot_path_len = 0,
        .auth_kb_id = -1,
        .protocol_grammar_kb_id = -1,
        .domain_kb_id = -1,
        .backlog = 128,
    };
    const addr = "0.0.0.0";
    @memcpy(cfg.address[0..addr.len], addr);
    cfg.address_len = @intCast(addr.len);
    return cfg;
}

pub const ServerMetrics = struct {
    total_connections_accepted: i64,
    total_connections_rejected: i64,
    total_requests_served: i64,
    active_connections: i32,
    active_sessions: i32,
};

pub const Server = struct {
    config: ServerConfig,
    connections: [MAX_CONNECTIONS]ServerConnection,
    n_active: std.atomic.Value(i32),
    shutdown_flag: std.atomic.Value(i32),
    metrics: ServerMetrics,
    listen_fd: i32,
    auth_kb_id: i32,
    template_session_id: i32,
    store: *@import("../kb/store.zig").KBStore,

    pub fn activeCount(self: *const Server) i32 {
        return self.n_active.load(.seq_cst);
    }

    pub fn isShutdown(self: *const Server) bool {
        return self.shutdown_flag.load(.seq_cst) != 0;
    }
};

pub fn initServer(config: ServerConfig, store: *@import("../kb/store.zig").KBStore) Server {
    var server = Server{
        .config = config,
        .connections = undefined,
        .n_active = std.atomic.Value(i32).init(0),
        .shutdown_flag = std.atomic.Value(i32).init(0),
        .metrics = .{
            .total_connections_accepted = 0,
            .total_connections_rejected = 0,
            .total_requests_served = 0,
            .active_connections = 0,
            .active_sessions = 0,
        },
        .listen_fd = -1,
        .auth_kb_id = config.auth_kb_id,
        .template_session_id = -1,
        .store = store,
    };
    for (&server.connections) |*conn| {
        conn.* = defaultConnection();
    }
    return server;
}
