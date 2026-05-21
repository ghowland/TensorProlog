// ============================================================
// src/config/integration_test.zig
// ============================================================

const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const kb_types = @import("../kb/types.zig");
const seed_mod = @import("../seed/seed_init.zig");
const server_types = @import("../server/types.zig");
const auth_mod = @import("../server/auth.zig");
const rate_limit_mod = @import("../server/rate_limit.zig");
const health_mod = @import("../server/health.zig");
const runner_manager_mod = @import("../runner/runner_manager.zig");
const runner_types = @import("../runner/types.zig");
const sre_mod = @import("../test_scenarios/sre_scenario.zig");
const det_mod = @import("../test_scenarios/determinism_tests.zig");
const prolog_rule = @import("../prolog/rule.zig");
const grammar_compile = @import("../grammar/compile.zig");
const grammar_render = @import("../grammar/render.zig");
const confidence_mod = @import("../confidence/propagate.zig");
const dispatch_mod = @import("../builtins/dispatch.zig");
const register_arith = @import("../builtins/arithmetic.zig");
const q16_mod = @import("../vdr/q16.zig");

const Q16 = q16_mod.Q16;
const KBStore = kb_store_mod.KBStore;
const VlpFact = kb_types.VlpFact;

pub const IntegrationResult = struct {
    seed_init_ok: bool,
    kb_operations_ok: bool,
    fact_roundtrip_ok: bool,
    prolog_ok: bool,
    grammar_ok: bool,
    confidence_ok: bool,
    auth_ok: bool,
    rate_limit_ok: bool,
    health_ok: bool,
    runner_create_ok: bool,
    sre_scenario_ok: bool,
    determinism_ok: bool,
    builtin_dispatch_ok: bool,
    total_checks: i32,
    total_passed: i32,
};

pub fn runIntegrationTest(config: *const SystemConfig) IntegrationResult {
    _ = config;

    var result = IntegrationResult{
        .seed_init_ok = false,
        .kb_operations_ok = false,
        .fact_roundtrip_ok = false,
        .prolog_ok = false,
        .grammar_ok = false,
        .confidence_ok = false,
        .auth_ok = false,
        .rate_limit_ok = false,
        .health_ok = false,
        .runner_create_ok = false,
        .sre_scenario_ok = false,
        .determinism_ok = false,
        .builtin_dispatch_ok = false,
        .total_checks = 13,
        .total_passed = 0,
    };

    var kb_buf: [4096]kb_types.VlpKB = undefined;
    var fact_buf: [65536]VlpFact = undefined;
    var text_buf: [262144]u8 = undefined;
    var path_entries: [8192]kb_store_mod.PathEntry = undefined;

    var store = KBStore.init(&kb_buf, &fact_buf, &text_buf, &path_entries);

    const seeds = seed_mod.seedInit(&store);
    result.seed_init_ok = (seeds.root >= 0 and seeds.system >= 0 and seeds.oso >= 0);
    if (result.seed_init_ok) result.total_passed += 1;

    const test_kb = store.createKB(.{
        .name = "integration_test",
        .parent_id = seeds.root,
        .visibility = .public,
        .owner = "test",
        .max_facts = 64,
        .max_rules = 8,
        .max_children = 4,
    });
    const child_kb = store.createKB(.{
        .name = "child",
        .parent_id = test_kb,
        .visibility = .internal,
        .owner = "test",
        .max_facts = 32,
        .max_rules = 0,
        .max_children = 0,
    });
    result.kb_operations_ok = (test_kb >= 0 and child_kb >= 0);
    if (result.kb_operations_ok) result.total_passed += 1;

    const test_val = Q16.fromFraction(355, 113);
    const test_fact = VlpFact{
        .tag = .value,
        .value = test_val,
        .provenance = .{
            .source_type = .vdr_computation,
            .source_kb_id = test_kb,
            .source_slot_id = 0,
            .confidence = .{ .v = Q16.D, .r0 = 0 },
            .timestamp = 1700000000,
            .derivation_rule_id = -1,
        },
    };
    _ = fact_mod.factAssert(&store, test_kb, 0, &test_fact);
    const read_back = fact_mod.factQuery(&store, test_kb, 0);
    result.fact_roundtrip_ok = false;
    if (read_back) |f| {
        result.fact_roundtrip_ok = Q16.eql(f.value, test_val) and f.tag == .value;
    }
    if (result.fact_roundtrip_ok) result.total_passed += 1;

    var fired_ids: [16]i32 = undefined;
    var n_fired: i32 = 0;
    const fire_status = prolog_rule.fireAll(&store, test_kb, &fired_ids, 16, &n_fired);
    result.prolog_ok = (fire_status == .ok);
    if (result.prolog_ok) result.total_passed += 1;

    var grammar: grammar_compile.VlpGrammar = undefined;
    const template = "Service {name:text} status: {status:text}";
    const compile_status = grammar_compile.compile(template, &grammar);
    result.grammar_ok = (compile_status == .ok and grammar.validated);
    if (result.grammar_ok) result.total_passed += 1;

    var sources = [_]Q16{ .{ .v = 62259, .r0 = 0 }, .{ .v = 62259, .r0 = 0 } };
    var combined: Q16 = undefined;
    _ = confidence_mod.combineAgreeing(&sources, 2, &combined);
    result.confidence_ok = (combined.v > 62259 and combined.v <= Q16.D);
    if (result.confidence_ok) result.total_passed += 1;

    const auth_kb = auth_mod.createAuthKB(&store, seeds.root);
    _ = auth_mod.registerUser(&store, auth_kb, 1, "test_token_abc", 1);
    const auth_result = auth_mod.authenticate(&store, auth_kb, "test_token_abc", 3600);
    result.auth_ok = (auth_result.status == .ok and auth_result.credential.valid and auth_result.credential.user_id == 1);
    if (result.auth_ok) result.total_passed += 1;

    const rl_kb = rate_limit_mod.createRateLimitKB(&store, seeds.root, 100);
    rate_limit_mod.configureRateLimit(.{ .window_seconds = 60, .max_requests = 5, .counter_kb_id = rl_kb });
    var all_allowed = true;
    for (0..5) |_| {
        const rl_result = rate_limit_mod.checkRateLimit(&store, auth_kb, 1);
        if (!rl_result.allowed) all_allowed = false;
    }
    const rl_over = rate_limit_mod.checkRateLimit(&store, auth_kb, 1);
    result.rate_limit_ok = (all_allowed and !rl_over.allowed);
    if (result.rate_limit_ok) result.total_passed += 1;

    var server = server_types.initServer(server_types.defaultServerConfig(), &store);
    const health = health_mod.collectHealth(&server);
    result.health_ok = (health.active_connections == 0 and health.total_requests == 0);
    if (result.health_ok) result.total_passed += 1;

    var mgr = runner_manager_mod.RunnerManager.init(2);
    const poller_id = mgr.createPoller(.{
        .session_id = 0,
        .interval_ms = 60000,
        .max_consecutive_errors = 5,
        .notification_kb_id = test_kb,
        .log_kb_id = test_kb,
    }, &store);
    result.runner_create_ok = (poller_id != null);
    if (result.runner_create_ok) result.total_passed += 1;

    const sre_result = sre_mod.runSreScenario(&store);
    result.sre_scenario_ok = (sre_result.kb_tree_ok and sre_result.facts_asserted_ok and sre_result.confidence_ok and sre_result.grammar_render_ok and sre_result.l3_resolution_ok);
    if (result.sre_scenario_ok) result.total_passed += 1;

    const det_result = det_mod.runDeterminismTests(&store);
    result.determinism_ok = (det_result.total_mismatches == 0 and det_result.q16_arithmetic_ok and det_result.softmax_ok and det_result.collections_ok and det_result.sets_ok and det_result.linalg_ok and det_result.stats_ok and det_result.kb_fact_roundtrip_ok and det_result.confidence_ok);
    if (result.determinism_ok) result.total_passed += 1;

    var builtin_table = dispatch_mod.BuiltinTable.init();
    dispatch_mod.registerAllBuiltins(&builtin_table);
    result.builtin_dispatch_ok = (builtin_table.count > 0 and builtin_table.isRegistered(0) and builtin_table.isRegistered(100));
    if (result.builtin_dispatch_ok) result.total_passed += 1;

    return result;
}

pub fn printIntegrationResult(result: *const IntegrationResult) void {
    std.debug.print("\n=== TensorProlog Integration Test ===\n", .{});
    printCheck("Seed init", result.seed_init_ok);
    printCheck("KB operations", result.kb_operations_ok);
    printCheck("Fact roundtrip", result.fact_roundtrip_ok);
    printCheck("Prolog engine", result.prolog_ok);
    printCheck("Grammar engine", result.grammar_ok);
    printCheck("Confidence propagation", result.confidence_ok);
    printCheck("Authentication", result.auth_ok);
    printCheck("Rate limiting", result.rate_limit_ok);
    printCheck("Health check", result.health_ok);
    printCheck("Runner creation", result.runner_create_ok);
    printCheck("SRE scenario", result.sre_scenario_ok);
    printCheck("Determinism", result.determinism_ok);
    printCheck("Builtin dispatch", result.builtin_dispatch_ok);
    std.debug.print("\nResult: {d}/{d} passed\n", .{ result.total_passed, result.total_checks });
}

fn printCheck(name: []const u8, ok: bool) void {
    const status = if (ok) "PASS" else "FAIL";
    std.debug.print("  [{s}] {s}\n", .{ status, name });
}
