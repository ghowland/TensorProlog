// ============================================================
// src/server/rate_limit.zig
// ============================================================

pub const RateLimitConfig = struct {
    window_seconds: i32,
    max_requests: i32,
    counter_kb_id: i32,
};

pub const RateLimitResult = struct {
    allowed: bool,
    remaining: i32,
    retry_after_seconds: i32,
};

var global_rate_config = RateLimitConfig{
    .window_seconds = 60,
    .max_requests = 100,
    .counter_kb_id = -1,
};

pub fn configureRateLimit(config: RateLimitConfig) void {
    global_rate_config = config;
}

pub fn checkRateLimit(store: *KBStore, auth_kb_id: i32, user_id: i32) RateLimitResult {
    if (global_rate_config.counter_kb_id < 0) {
        return .{ .allowed = true, .remaining = global_rate_config.max_requests, .retry_after_seconds = 0 };
    }

    const now = timestampNow();
    const counter_slot = user_id * 2;
    const window_slot = user_id * 2 + 1;

    _ = auth_kb_id;
    const rl_kb = global_rate_config.counter_kb_id;

    var counter_value: i32 = 0;
    var window_start: i32 = 0;

    const counter_fact = fact_mod.factQuery(store, rl_kb, counter_slot);
    if (counter_fact) |cf| {
        counter_value = cf.value.v;
    }

    const window_fact = fact_mod.factQuery(store, rl_kb, window_slot);
    if (window_fact) |wf| {
        window_start = wf.value.v;
    }

    if (now - window_start >= global_rate_config.window_seconds) {
        counter_value = 0;
        window_start = now;

        const wf = VlpFact{
            .tag = .counter,
            .value = .{ .v = now, .r0 = 0 },
            .provenance = rlProvenance(rl_kb, window_slot),
        };
        _ = fact_mod.factAssert(store, rl_kb, window_slot, &wf);
    }

    if (counter_value >= global_rate_config.max_requests) {
        const remaining_window = global_rate_config.window_seconds - (now - window_start);
        return .{
            .allowed = false,
            .remaining = 0,
            .retry_after_seconds = @max(remaining_window, 1),
        };
    }

    counter_value += 1;
    const cf = VlpFact{
        .tag = .counter,
        .value = .{ .v = counter_value, .r0 = 0 },
        .provenance = rlProvenance(rl_kb, counter_slot),
    };
    _ = fact_mod.factAssert(store, rl_kb, counter_slot, &cf);

    return .{
        .allowed = true,
        .remaining = global_rate_config.max_requests - counter_value,
        .retry_after_seconds = 0,
    };
}

fn rlProvenance(kb_id: i32, slot_id: i32) kb_types.VlpProvenance {
    return .{
        .source_type = .vdr_computation,
        .source_kb_id = kb_id,
        .source_slot_id = slot_id,
        .confidence = .{ .v = Q16.D, .r0 = 0 },
        .timestamp = timestampNow(),
        .derivation_rule_id = -1,
    };
}

pub fn createRateLimitKB(store: *KBStore, parent_id: i32, max_users: i32) i32 {
    return store.createKB(.{
        .name = "rate_limits",
        .parent_id = parent_id,
        .visibility = .internal,
        .owner = "system",
        .max_facts = max_users * 2,
        .max_rules = 0,
        .max_children = 0,
    });
}

pub fn resetRateLimit(store: *KBStore, user_id: i32) void {
    if (global_rate_config.counter_kb_id < 0) return;
    const counter_slot = user_id * 2;
    _ = fact_mod.factRetract(store, global_rate_config.counter_kb_id, counter_slot);
}

pub fn getRateLimitStatus(store: *KBStore, user_id: i32) struct { count: i32, window_start: i32 } {
    if (global_rate_config.counter_kb_id < 0) return .{ .count = 0, .window_start = 0 };
    const counter_slot = user_id * 2;
    const window_slot = user_id * 2 + 1;

    var count: i32 = 0;
    var ws: i32 = 0;

    const cf = fact_mod.factQuery(store, global_rate_config.counter_kb_id, counter_slot);
    if (cf) |f| count = f.value.v;

    const wf = fact_mod.factQuery(store, global_rate_config.counter_kb_id, window_slot);
    if (wf) |f| ws = f.value.v;

    return .{ .count = count, .window_start = ws };
}
