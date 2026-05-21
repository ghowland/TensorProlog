// ============================================================
// src/config/config_file.zig
// ============================================================

pub fn parseConfigFile(path: []const u8, config: *SystemConfig) VlpStatus {
    const file = std.fs.cwd().openFile(path, .{}) catch return .err_snapshot_failed;
    defer file.close();
    var buf: [8192]u8 = undefined;
    const n = file.read(&buf) catch return .err_snapshot_failed;
    const data = buf[0..n];
    var pos: usize = 0;

    while (pos < data.len) {
        while (pos < data.len and (data[pos] == ' ' or data[pos] == '\t' or data[pos] == '\n' or data[pos] == '\r')) pos += 1;
        if (pos >= data.len) break;
        if (data[pos] == '#') {
            while (pos < data.len and data[pos] != '\n') pos += 1;
            continue;
        }

        const key_start = pos;
        while (pos < data.len and data[pos] != '=' and data[pos] != '\n') pos += 1;
        if (pos >= data.len or data[pos] != '=') continue;
        const key = trimSlice(data[key_start..pos]);
        pos += 1;

        while (pos < data.len and (data[pos] == ' ' or data[pos] == '\t')) pos += 1;
        const val_start = pos;
        while (pos < data.len and data[pos] != '\n' and data[pos] != '\r') pos += 1;
        const val = trimSlice(data[val_start..pos]);

        applyConfigValue(config, key, val);
    }

    return .ok;
}

fn applyConfigValue(config: *SystemConfig, key: []const u8, val: []const u8) void {
    if (eql(key, "port")) {
        config.server_port = parseI32(val);
    } else if (eql(key, "max_connections")) {
        config.server_max_connections = parseI32(val);
    } else if (eql(key, "max_runners")) {
        config.max_runners = parseI32(val);
    } else if (eql(key, "pool_threads")) {
        config.runner_thread_pool_size = parseI32(val);
    } else if (eql(key, "max_kbs")) {
        config.max_total_kbs = parseI32(val);
    } else if (eql(key, "max_facts")) {
        config.max_total_facts = @intCast(parseI32(val));
    } else if (eql(key, "max_rules")) {
        config.max_total_rules = parseI32(val);
    } else if (eql(key, "credential_ttl")) {
        config.server_credential_ttl = parseI32(val);
    } else if (eql(key, "idle_timeout")) {
        config.server_idle_timeout = parseI32(val);
    } else if (eql(key, "snapshot_interval")) {
        config.auto_snapshot_interval = parseI32(val);
    } else if (eql(key, "layers")) {
        config.model_n_layers = parseI32(val);
    } else if (eql(key, "d_model")) {
        config.model_d_model = parseI32(val);
    } else if (eql(key, "n_heads")) {
        config.model_n_heads = parseI32(val);
    } else if (eql(key, "vocab_size")) {
        config.model_vocab_size = parseI32(val);
    } else if (eql(key, "rate_limit_window")) {
        config.rate_limit_window = parseI32(val);
    } else if (eql(key, "rate_limit_max")) {
        config.rate_limit_max_requests = parseI32(val);
    } else if (eql(key, "seed_path")) {
        config_defaults.setPath(&config.seed_snapshot_path, &config.seed_snapshot_path_len, val);
    } else if (eql(key, "checkpoint_path")) {
        config_defaults.setPath(&config.model_checkpoint_path, &config.model_checkpoint_path_len, val);
    } else if (eql(key, "persistent_sessions")) {
        config.server_persistent_sessions = eql(val, "true");
    }
}

fn trimSlice(s: []const u8) []const u8 {
    var start: usize = 0;
    while (start < s.len and (s[start] == ' ' or s[start] == '\t')) start += 1;
    var end = s.len;
    while (end > start and (s[end - 1] == ' ' or s[end - 1] == '\t' or s[end - 1] == '\r')) end -= 1;
    return s[start..end];
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn parseI32(s: []const u8) i32 {
    var result: i32 = 0;
    var negative = false;
    var start: usize = 0;
    if (s.len > 0 and s[0] == '-') {
        negative = true;
        start = 1;
    }
    for (s[start..]) |c| {
        if (c >= '0' and c <= '9') result = result * 10 + @as(i32, c - '0');
    }
    return if (negative) -result else result;
}
