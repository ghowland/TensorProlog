// ============================================================
// src/config/cli.zig
// ============================================================

const SystemConfig = @import("system_config.zig").SystemConfig;
const config_defaults = @import("system_config.zig");

pub const CliArgs = struct {
    config: SystemConfig,
    config_file_path: [256]u8,
    config_file_path_len: i32,
    show_help: bool,
    show_version: bool,
    run_tests: bool,
    verbose: bool,
};

pub fn defaultCliArgs() CliArgs {
    return .{
        .config = config_defaults.defaults(),
        .config_file_path = undefined,
        .config_file_path_len = 0,
        .show_help = false,
        .show_version = false,
        .run_tests = false,
        .verbose = false,
    };
}

pub fn parseCli(argv: []const []const u8) CliArgs {
    var args = defaultCliArgs();
    var i: usize = 1;
    while (i < argv.len) {
        const arg = argv[i];

        if (eql(arg, "--help") or eql(arg, "-h")) {
            args.show_help = true;
        } else if (eql(arg, "--version") or eql(arg, "-v")) {
            args.show_version = true;
        } else if (eql(arg, "--test")) {
            args.run_tests = true;
        } else if (eql(arg, "--verbose")) {
            args.verbose = true;
        } else if (eql(arg, "--port") and i + 1 < argv.len) {
            i += 1;
            args.config.server_port = parseI32(argv[i]);
        } else if (eql(arg, "--max-connections") and i + 1 < argv.len) {
            i += 1;
            args.config.server_max_connections = parseI32(argv[i]);
        } else if (eql(arg, "--max-runners") and i + 1 < argv.len) {
            i += 1;
            args.config.max_runners = parseI32(argv[i]);
        } else if (eql(arg, "--pool-threads") and i + 1 < argv.len) {
            i += 1;
            args.config.runner_thread_pool_size = parseI32(argv[i]);
        } else if (eql(arg, "--max-kbs") and i + 1 < argv.len) {
            i += 1;
            args.config.max_total_kbs = parseI32(argv[i]);
        } else if (eql(arg, "--max-facts") and i + 1 < argv.len) {
            i += 1;
            args.config.max_total_facts = @intCast(parseI32(argv[i]));
        } else if (eql(arg, "--max-rules") and i + 1 < argv.len) {
            i += 1;
            args.config.max_total_rules = parseI32(argv[i]);
        } else if (eql(arg, "--credential-ttl") and i + 1 < argv.len) {
            i += 1;
            args.config.server_credential_ttl = parseI32(argv[i]);
        } else if (eql(arg, "--idle-timeout") and i + 1 < argv.len) {
            i += 1;
            args.config.server_idle_timeout = parseI32(argv[i]);
        } else if (eql(arg, "--snapshot-interval") and i + 1 < argv.len) {
            i += 1;
            args.config.auto_snapshot_interval = parseI32(argv[i]);
        } else if (eql(arg, "--seed") and i + 1 < argv.len) {
            i += 1;
            config_defaults.setPath(&args.config.seed_snapshot_path, &args.config.seed_snapshot_path_len, argv[i]);
        } else if (eql(arg, "--checkpoint") and i + 1 < argv.len) {
            i += 1;
            config_defaults.setPath(&args.config.model_checkpoint_path, &args.config.model_checkpoint_path_len, argv[i]);
        } else if (eql(arg, "--config") and i + 1 < argv.len) {
            i += 1;
            const cl = @min(argv[i].len, args.config_file_path.len);
            @memcpy(args.config_file_path[0..cl], argv[i][0..cl]);
            args.config_file_path_len = @intCast(cl);
        } else if (eql(arg, "--layers") and i + 1 < argv.len) {
            i += 1;
            args.config.model_n_layers = parseI32(argv[i]);
        } else if (eql(arg, "--d-model") and i + 1 < argv.len) {
            i += 1;
            args.config.model_d_model = parseI32(argv[i]);
        } else if (eql(arg, "--n-heads") and i + 1 < argv.len) {
            i += 1;
            args.config.model_n_heads = parseI32(argv[i]);
        } else if (eql(arg, "--vocab-size") and i + 1 < argv.len) {
            i += 1;
            args.config.model_vocab_size = parseI32(argv[i]);
        } else if (eql(arg, "--persistent-sessions")) {
            args.config.server_persistent_sessions = true;
        } else if (eql(arg, "--rate-limit-window") and i + 1 < argv.len) {
            i += 1;
            args.config.rate_limit_window = parseI32(argv[i]);
        } else if (eql(arg, "--rate-limit-max") and i + 1 < argv.len) {
            i += 1;
            args.config.rate_limit_max_requests = parseI32(argv[i]);
        }

        i += 1;
    }
    return args;
}

pub fn printHelp() void {
    const help =
        \\TensorProlog - Integer GPU Compute for Exact LLM Inference
        \\
        \\Usage: tensorprolog [options]
        \\
        \\Options:
        \\  --help, -h              Show this help
        \\  --version, -v           Show version
        \\  --test                  Run test suite
        \\  --verbose               Verbose output
        \\  --config <path>         Config file path
        \\  --port <n>              Server port (default: 8080)
        \\  --max-connections <n>   Max concurrent connections (default: 64)
        \\  --max-runners <n>       Max runners (default: 64)
        \\  --pool-threads <n>      Thread pool size (default: auto)
        \\  --max-kbs <n>           Max KBs (default: 100000)
        \\  --max-facts <n>         Max facts (default: 10000000)
        \\  --max-rules <n>         Max rules (default: 100000)
        \\  --credential-ttl <n>    Credential TTL seconds (default: 3600)
        \\  --idle-timeout <n>      Idle timeout seconds (default: 300)
        \\  --snapshot-interval <n> Auto snapshot interval (default: 100)
        \\  --seed <path>           Seed snapshot path
        \\  --checkpoint <path>     Model checkpoint path
        \\  --layers <n>            Model layers
        \\  --d-model <n>           Model dimension
        \\  --n-heads <n>           Number of attention heads
        \\  --vocab-size <n>        Vocabulary size
        \\  --persistent-sessions   Enable persistent sessions
        \\  --rate-limit-window <n> Rate limit window seconds (default: 60)
        \\  --rate-limit-max <n>    Rate limit max requests (default: 100)
        \\
    ;
    std.debug.print("{s}", .{help});
}

pub fn printVersion() void {
    std.debug.print("TensorProlog v0.1.0 (Zig 0.15.1, Q16 integer arithmetic)\n", .{});
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
        if (c >= '0' and c <= '9') {
            result = result * 10 + @as(i32, c - '0');
        }
    }
    return if (negative) -result else result;
}
