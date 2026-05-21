// ============================================================
// src/config/system_config.zig
// ============================================================

const std = @import("std");
const types = @import("../vdr/types.zig");
const q16 = @import("../vdr/q16.zig");

const Q16 = q16.Q16;
const VlpStatus = types.VlpStatus;

pub const SystemConfig = struct {
    device_id: i32,
    n_devices: i32,

    model_checkpoint_path: [256]u8,
    model_checkpoint_path_len: i32,
    model_n_layers: i32,
    model_d_model: i32,
    model_n_heads: i32,
    model_vocab_size: i32,
    model_mlp_dim: i32,
    model_qbasis: i32,

    max_total_kbs: i32,
    max_total_facts: i64,
    max_total_rules: i32,
    max_total_terms: i64,
    text_store_bytes: i64,
    scratch_per_stream_bytes: i64,

    max_concurrent_sessions: i32,
    default_max_kb_per_session: i32,
    default_max_turns: i32,
    auto_snapshot_interval: i32,

    max_runners: i32,
    runner_thread_pool_size: i32,

    audit_ring_capacity: i32,
    default_visibility: i8,

    seed_snapshot_path: [256]u8,
    seed_snapshot_path_len: i32,

    default_temperature_v: i32,
    default_top_k: i32,
    default_top_p_v: i32,

    server_port: i32,
    server_max_connections: i32,
    server_credential_ttl: i32,
    server_idle_timeout: i32,
    server_shutdown_timeout: i32,
    server_persistent_sessions: bool,

    rate_limit_window: i32,
    rate_limit_max_requests: i32,
};

pub fn defaults() SystemConfig {
    var cfg = SystemConfig{
        .device_id = -1,
        .n_devices = 1,

        .model_checkpoint_path = undefined,
        .model_checkpoint_path_len = 0,
        .model_n_layers = 1,
        .model_d_model = 64,
        .model_n_heads = 1,
        .model_vocab_size = 256,
        .model_mlp_dim = 128,
        .model_qbasis = 16,

        .max_total_kbs = 100000,
        .max_total_facts = 10000000,
        .max_total_rules = 100000,
        .max_total_terms = 1000000,
        .text_store_bytes = 104857600,
        .scratch_per_stream_bytes = 10485760,

        .max_concurrent_sessions = 10000,
        .default_max_kb_per_session = 100,
        .default_max_turns = 0,
        .auto_snapshot_interval = 100,

        .max_runners = 64,
        .runner_thread_pool_size = 0,

        .audit_ring_capacity = 1000000,
        .default_visibility = 1,

        .seed_snapshot_path = undefined,
        .seed_snapshot_path_len = 0,

        .default_temperature_v = Q16.D,
        .default_top_k = 50,
        .default_top_p_v = 58982,

        .server_port = 8080,
        .server_max_connections = 64,
        .server_credential_ttl = 3600,
        .server_idle_timeout = 300,
        .server_shutdown_timeout = 30,
        .server_persistent_sessions = false,

        .rate_limit_window = 60,
        .rate_limit_max_requests = 100,
    };

    const default_seed = "seed.vlps";
    @memcpy(cfg.seed_snapshot_path[0..default_seed.len], default_seed);
    cfg.seed_snapshot_path_len = @intCast(default_seed.len);

    return cfg;
}

pub fn setPath(dest: *[256]u8, dest_len: *i32, src: []const u8) void {
    const n = @min(src.len, dest.len);
    @memcpy(dest[0..n], src[0..n]);
    dest_len.* = @intCast(n);
}
