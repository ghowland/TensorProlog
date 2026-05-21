// ============================================================
// src/runner/sre_deployment.zig
// ============================================================

pub const SreDeploymentConfig = struct {
    prometheus_session_id: i32,
    deploy_session_id: i32,
    triage_session_id: i32,
    hygiene_session_id: i32,
    prometheus_interval_ms: i32,
    triage_interval_ms: i32,
    hygiene_interval_ms: i32,
    max_turns_before_recycle: i32,
    ops_kb_id: i32,
    rules_kb_id: i32,
    incidents_kb_id: i32,
    notification_kb_id: i32,
    log_kb_id: i32,
};

pub const SreDeployment = struct {
    prometheus_runner_id: i32,
    deploy_runner_id: i32,
    triage_runner_id: i32,
    hygiene_runner_id: i32,
};

pub fn createSreDeployment(mgr: *RunnerManager, config: SreDeploymentConfig, store: *KBStore) SreDeployment {
    const prom_id = createProcessor(.{
        .session_id = config.prometheus_session_id,
        .max_turns_before_recycle = config.max_turns_before_recycle,
        .max_consecutive_errors = 10,
        .compact_rules_kb_id = config.rules_kb_id,
        .log_kb_id = config.log_kb_id,
    }, store, &mgr.table) orelse -1;

    const deploy_id = createProcessor(.{
        .session_id = config.deploy_session_id,
        .max_turns_before_recycle = config.max_turns_before_recycle,
        .max_consecutive_errors = 10,
        .compact_rules_kb_id = config.rules_kb_id,
        .log_kb_id = config.log_kb_id,
    }, store, &mgr.table) orelse -1;

    const triage_id = pool_mod_poller.createPoller(.{
        .session_id = config.triage_session_id,
        .interval_ms = config.triage_interval_ms,
        .max_consecutive_errors = 5,
        .notification_kb_id = config.notification_kb_id,
        .log_kb_id = config.log_kb_id,
    }, store, &mgr.table) orelse -1;

    const hygiene_id = internal_mod.createInternal(.{
        .session_id = config.hygiene_session_id,
        .interval_ms = config.hygiene_interval_ms,
        .max_consecutive_errors = 3,
        .log_kb_id = config.log_kb_id,
    }, store, &mgr.table) orelse -1;

    return .{
        .prometheus_runner_id = prom_id,
        .deploy_runner_id = deploy_id,
        .triage_runner_id = triage_id,
        .hygiene_runner_id = hygiene_id,
    };
}

pub fn startSreDeployment(mgr: *RunnerManager, deployment: SreDeployment) void {
    if (deployment.prometheus_runner_id >= 0) _ = mgr.startRunner(deployment.prometheus_runner_id);
    if (deployment.deploy_runner_id >= 0) _ = mgr.startRunner(deployment.deploy_runner_id);
    if (deployment.triage_runner_id >= 0) _ = mgr.startRunner(deployment.triage_runner_id);
    if (deployment.hygiene_runner_id >= 0) _ = mgr.startRunner(deployment.hygiene_runner_id);
}

pub fn stopSreDeployment(mgr: *RunnerManager, deployment: SreDeployment) void {
    if (deployment.prometheus_runner_id >= 0) _ = mgr.stopRunner(deployment.prometheus_runner_id);
    if (deployment.deploy_runner_id >= 0) _ = mgr.stopRunner(deployment.deploy_runner_id);
    if (deployment.triage_runner_id >= 0) _ = mgr.stopRunner(deployment.triage_runner_id);
    if (deployment.hygiene_runner_id >= 0) _ = mgr.stopRunner(deployment.hygiene_runner_id);
}

pub fn getSreStatus(mgr: *const RunnerManager, deployment: SreDeployment) SreDeploymentStatus {
    return .{
        .prometheus = mgr.getStatus(deployment.prometheus_runner_id),
        .deploy = mgr.getStatus(deployment.deploy_runner_id),
        .triage = mgr.getStatus(deployment.triage_runner_id),
        .hygiene = mgr.getStatus(deployment.hygiene_runner_id),
    };
}

pub const SreDeploymentStatus = struct {
    prometheus: ?runner_types.RunnerStatus,
    deploy: ?runner_types.RunnerStatus,
    triage: ?runner_types.RunnerStatus,
    hygiene: ?runner_types.RunnerStatus,
};

const pool_mod_poller = @import("poller.zig");
