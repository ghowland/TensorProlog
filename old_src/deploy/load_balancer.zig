// ============================================================
// src/deploy/load_balancer.zig
// ============================================================

pub const MAX_BACKENDS: usize = 16;

pub const Backend = struct {
    id: i32,
    address: [64]u8,
    address_len: i32,
    port: i32,
    healthy: bool,
    active_connections: i32,
    total_requests: i64,
    last_health_check: i32,
};

pub const LoadBalancer = struct {
    backends: [MAX_BACKENDS]Backend,
    n_backends: i32,
    next_index: i32,
    strategy: BalanceStrategy,
};

pub const BalanceStrategy = enum(i8) {
    round_robin = 0,
    least_connections = 1,
};

pub fn initLoadBalancer(strategy: BalanceStrategy) LoadBalancer {
    var lb = LoadBalancer{
        .backends = undefined,
        .n_backends = 0,
        .next_index = 0,
        .strategy = strategy,
    };
    for (&lb.backends) |*b| {
        b.healthy = false;
        b.active_connections = 0;
        b.total_requests = 0;
        b.id = -1;
    }
    return lb;
}

pub fn addBackend(lb: *LoadBalancer, address: []const u8, port: i32) ?i32 {
    if (lb.n_backends >= MAX_BACKENDS) return null;
    const idx: usize = @intCast(lb.n_backends);
    lb.backends[idx].id = lb.n_backends;
    const al = @min(address.len, lb.backends[idx].address.len);
    @memcpy(lb.backends[idx].address[0..al], address[0..al]);
    lb.backends[idx].address_len = @intCast(al);
    lb.backends[idx].port = port;
    lb.backends[idx].healthy = true;
    lb.backends[idx].active_connections = 0;
    lb.backends[idx].total_requests = 0;
    lb.backends[idx].last_health_check = timestampNow();
    lb.n_backends += 1;
    return lb.n_backends - 1;
}

pub fn route(lb: *LoadBalancer) ?i32 {
    if (lb.n_backends <= 0) return null;
    const nb: usize = @intCast(lb.n_backends);

    switch (lb.strategy) {
        .round_robin => {
            var attempts: usize = 0;
            while (attempts < nb) {
                const idx: usize = @intCast(@mod(lb.next_index, lb.n_backends));
                lb.next_index = @mod(lb.next_index + 1, lb.n_backends);
                if (lb.backends[idx].healthy) {
                    lb.backends[idx].active_connections += 1;
                    lb.backends[idx].total_requests += 1;
                    return lb.backends[idx].id;
                }
                attempts += 1;
            }
            return null;
        },
        .least_connections => {
            var best: ?usize = null;
            var best_conns: i32 = std.math.maxInt(i32);
            for (0..nb) |i| {
                if (lb.backends[i].healthy and lb.backends[i].active_connections < best_conns) {
                    best_conns = lb.backends[i].active_connections;
                    best = i;
                }
            }
            if (best) |idx| {
                lb.backends[idx].active_connections += 1;
                lb.backends[idx].total_requests += 1;
                return lb.backends[idx].id;
            }
            return null;
        },
    }
}

pub fn releaseBackend(lb: *LoadBalancer, backend_id: i32) void {
    const idx: usize = @intCast(backend_id);
    if (idx >= @as(usize, @intCast(lb.n_backends))) return;
    if (lb.backends[idx].active_connections > 0) lb.backends[idx].active_connections -= 1;
}

pub fn markUnhealthy(lb: *LoadBalancer, backend_id: i32) void {
    const idx: usize = @intCast(backend_id);
    if (idx >= @as(usize, @intCast(lb.n_backends))) return;
    lb.backends[idx].healthy = false;
}

pub fn markHealthy(lb: *LoadBalancer, backend_id: i32) void {
    const idx: usize = @intCast(backend_id);
    if (idx >= @as(usize, @intCast(lb.n_backends))) return;
    lb.backends[idx].healthy = true;
    lb.backends[idx].last_health_check = timestampNow();
}

pub fn removeUnhealthy(lb: *LoadBalancer) i32 {
    var removed: i32 = 0;
    var i: usize = 0;
    while (i < @as(usize, @intCast(lb.n_backends))) {
        if (!lb.backends[i].healthy) {
            const nb: usize = @intCast(lb.n_backends);
            lb.backends[i] = lb.backends[nb - 1];
            lb.n_backends -= 1;
            removed += 1;
        } else {
            i += 1;
        }
    }
    return removed;
}
