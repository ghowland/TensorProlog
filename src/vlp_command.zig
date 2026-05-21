// ============================================================
// vlp_command.zig
// Command processor — LLM→system interface.
// Parses command tokens, enforces access+grants, dispatches.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const kb_mod = @import("vlp_kb_store.zig");
const prolog_mod = @import("vlp_prolog.zig");
const grammar_mod = @import("vlp_grammar.zig");
const builtin_mod = @import("vlp_builtin.zig");
const grant_mod = @import("vlp_grant.zig");
const access_mod = @import("vlp_access.zig");
const audit_mod = @import("vlp_audit.zig");
const session_mod = @import("vlp_session.zig");

// ============================================================
// Command result
// ============================================================

pub const CommandResult = struct {
    status: types.Status,
    output_kb_id: i32,
    output_slot_id: i32,
    output_bytes: i32,
    output_text: ?[]const u8,

    pub fn ok(kb_id: i32, slot_id: i32) CommandResult {
        return .{
            .status = types.Status.ok(),
            .output_kb_id = kb_id,
            .output_slot_id = slot_id,
            .output_bytes = 0,
            .output_text = null,
        };
    }

    pub fn okWithBytes(kb_id: i32, slot_id: i32, bytes: i32) CommandResult {
        return .{
            .status = types.Status.ok(),
            .output_kb_id = kb_id,
            .output_slot_id = slot_id,
            .output_bytes = bytes,
            .output_text = null,
        };
    }

    pub fn denied(code: types.ErrorCode, detail: i32) CommandResult {
        return .{
            .status = types.Status.err(.grant, code, detail),
            .output_kb_id = -1,
            .output_slot_id = -1,
            .output_bytes = 0,
            .output_text = null,
        };
    }

    pub fn failed(status: types.Status) CommandResult {
        return .{
            .status = status,
            .output_kb_id = -1,
            .output_slot_id = -1,
            .output_bytes = 0,
            .output_text = null,
        };
    }
};

// ============================================================
// KB URL — parsed from kb://root.ops.data.field references
// ============================================================

pub const KbUrl = struct {
    kb_id: i32,
    slot_id: i32,

    pub fn invalid() KbUrl {
        return .{ .kb_id = -1, .slot_id = -1 };
    }
};

// ============================================================
// Command Processor
// ============================================================

pub const CommandProcessor = struct {
    kb_store: *kb_mod.KbStore,
    prolog: *prolog_mod.PrologEngine,
    grammar: *grammar_mod.GrammarEngine,
    builtins: *builtin_mod.BuiltinExecutor,
    grants: *grant_mod.GrantEnforcer,
    access: *access_mod.AccessChecker,
    audit: *audit_mod.AuditLog,
    session_mgr: *session_mod.SessionManager,
    allocator: std.mem.Allocator,

    // Scratch for command argument parsing
    arg_buf: []i32,
    arg_capacity: i32,

    // Render output buffer
    render_buf: []u8,
    render_capacity: i32,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(
    kb_store: *kb_mod.KbStore,
    prolog: *prolog_mod.PrologEngine,
    grammar: *grammar_mod.GrammarEngine,
    builtins: *builtin_mod.BuiltinExecutor,
    grants: *grant_mod.GrantEnforcer,
    access: *access_mod.AccessChecker,
    audit: *audit_mod.AuditLog,
    session_mgr: *session_mod.SessionManager,
    allocator: std.mem.Allocator,
) CommandProcessor {
    const args = allocator.alloc(i32, 256) catch &.{};
    const render = allocator.alloc(u8, 16384) catch &.{};

    return .{
        .kb_store = kb_store,
        .prolog = prolog,
        .grammar = grammar,
        .builtins = builtins,
        .grants = grants,
        .access = access,
        .audit = audit,
        .session_mgr = session_mgr,
        .allocator = allocator,
        .arg_buf = args,
        .arg_capacity = @intCast(args.len),
        .render_buf = render,
        .render_capacity = @intCast(render.len),
    };
}

pub fn deinit(self: *CommandProcessor) void {
    if (self.arg_buf.len > 0) self.allocator.free(self.arg_buf);
    if (self.render_buf.len > 0) self.allocator.free(self.render_buf);
}

// ============================================================
// Parse — token stream → Command struct
// ============================================================

pub fn parse(self: *CommandProcessor, tokens: []const i32) ?types.Command {
    if (tokens.len == 0) return null;

    // First token → command type (~15 options, ~4 bits entropy)
    const cmd_type = tokenToCommandType(tokens[0]) orelse return null;

    var cmd = std.mem.zeroes(types.Command);
    cmd.type = cmd_type;
    cmd.grant_required = -1;

    // Second token → target path (resolve to kb_id)
    if (tokens.len > 1) {
        // Token is an index into command vocabulary → maps to path string
        // Simplified: treat token value as direct kb_id for now
        // Real implementation: look up token in command vocab KB,
        // extract dotted path, resolve via path index
        cmd.target_kb_id = tokens[1];
    }

    // Third token → target slot (for assert/query/retract)
    if (tokens.len > 2) {
        cmd.target_slot_id = tokens[2];
    }

    // For builtin calls, token 3 → builtin_id
    if (cmd_type == .builtin_call and tokens.len > 3) {
        cmd.builtin_id = tokens[3];
    }

    // Remaining tokens → args
    if (tokens.len > 3) {
        const arg_start: usize = 3;
        const n_args = @min(tokens.len - arg_start, @as(usize, @intCast(self.arg_capacity)));
        for (tokens[arg_start .. arg_start + n_args], 0..) |t, i| {
            self.arg_buf[i] = t;
        }
        cmd.args_count = @intCast(n_args);
        cmd.args_offset = 0; // args in self.arg_buf
    }

    // Set grant requirement for operational commands
    if (cmd.isOperational()) {
        cmd.grant_required = switch (cmd.type) {
            .op_filesystem => @intFromEnum(types.GrantClass.filesystem),
            .op_compile => @intFromEnum(types.GrantClass.compile),
            .op_execute => @intFromEnum(types.GrantClass.execute),
            .op_network => @intFromEnum(types.GrantClass.network),
            .op_process => @intFromEnum(types.GrantClass.process),
            else => -1,
        };
    }

    return cmd;
}

pub fn parseKbUrl(self: *CommandProcessor, text: []const u8) KbUrl {
    // Format: kb://root.ops.incidents.inc_001.field_name
    // or: root.ops.incidents.inc_001.field_name
    var path = text;

    // Strip kb:// prefix
    if (path.len > 5 and std.mem.eql(u8, path[0..5], "kb://")) {
        path = path[5..];
    }

    // Split on last '.' to separate path from field
    var last_dot: ?usize = null;
    for (path, 0..) |c, i| {
        if (c == '.') last_dot = i;
    }

    if (last_dot) |dot| {
        const kb_path = path[0..dot];
        const field = path[dot + 1 ..];

        const kb_id = self.kb_store.pathResolve(kb_path) orelse return KbUrl.invalid();
        // Field → slot_id: parse as integer or look up by name
        const slot_id = parseSlotId(field);
        return .{ .kb_id = kb_id, .slot_id = slot_id };
    }

    // No dot — entire text is the path, slot 0
    const kb_id = self.kb_store.pathResolve(path) orelse return KbUrl.invalid();
    return .{ .kb_id = kb_id, .slot_id = 0 };
}

// ============================================================
// Execute — the main dispatch point
// ============================================================

pub fn execute(self: *CommandProcessor, handle: types.SessionHandle, command: *const types.Command) CommandResult {
    const session = self.session_mgr.get(handle) orelse
        return CommandResult.failed(types.Status.err(.session, .session_limit, handle.id));

    const now = currentTimestamp();

    // Step 1: Access check
    if (command.target_kb_id >= 0) {
        if (!self.access.check(session, command.target_kb_id)) {
            self.audit.writeDenied(now, session.id, session.user_id, .access_denied, command.target_kb_id, command.target_slot_id);
            return CommandResult.denied(.kb_access_denied, command.target_kb_id);
        }
    }

    // Step 2: Grant check (if operational)
    if (command.requiresGrant()) {
        const grant_class = command.grantClass() orelse
            return CommandResult.denied(.grant_denied, -1);

        // Build target string for pattern matching
        var target_buf: [256]u8 = undefined;
        const target_len = self.kb_store.buildPathForKb(command.target_kb_id, &target_buf);
        const target = target_buf[0..@intCast(target_len)];

        const grant_result = self.grants.check(session, grant_class, target);

        self.audit.writeGrantCheck(now, session.id, session.user_id, command.target_kb_id, grant_result.grant_id, grant_result.granted);

        if (!grant_result.granted) {
            return CommandResult.denied(.grant_denied, command.target_kb_id);
        }
    }

    // Step 3: Dispatch by command type
    const result = switch (command.type) {
        .kb_assert => self.executeKbAssert(session, command, now),
        .kb_query => self.executeKbQuery(session, command, now),
        .kb_retract => self.executeKbRetract(session, command, now),
        .prolog_query => self.executePrologQuery(session, command, now),
        .prolog_assert_rule => self.executePrologAssertRule(session, command, now),
        .builtin_call => self.executeBuiltinCall(session, command, now),
        .grammar_render => self.executeGrammarRender(session, command, now),
        .direct_output => self.executeDirectOutput(session, command, now),
        .session_snapshot => self.executeSessionSnapshot(handle, now),
        .session_clone => self.executeSessionClone(handle, now),
        .op_filesystem, .op_compile, .op_execute, .op_network, .op_process => self.executeOperational(session, command, now),
    };

    // Step 4: Update session counters
    session.command_tokens_consumed += @as(i64, command.args_count) + 3; // type + target + slot + args

    return result;
}

pub fn executeBatch(self: *CommandProcessor, handle: types.SessionHandle, commands: []const types.Command, results: []CommandResult) types.Status {
    if (commands.len != results.len) return types.Status.err(.system, .init_failed, 0);

    for (commands, 0..) |*cmd, i| {
        results[i] = self.execute(handle, cmd);
        // Abort on first failure unless command is a query (reads don't fail hard)
        if (results[i].status.isErr() and cmd.type != .kb_query) {
            return results[i].status;
        }
    }

    return types.Status.ok();
}

// ============================================================
// Individual command executors
// ============================================================

fn executeKbAssert(self: *CommandProcessor, session: *types.Session, cmd: *const types.Command, now: i32) CommandResult {
    // Build fact from command args
    var fact = types.Fact{
        .tag = .value,
        .value = types.Q16.fromParts(if (cmd.args_count > 0) self.arg_buf[0] else 0, 0),
        .provenance = types.Provenance.direct(.llm_generated, cmd.target_kb_id, cmd.target_slot_id, now),
    };

    // Tag from arg if provided
    if (cmd.args_count > 1) {
        fact.tag = @enumFromInt(self.arg_buf[1]);
    }

    const status = self.kb_store.factWrite(cmd.target_kb_id, cmd.target_slot_id, &fact);
    if (status.isErr()) return CommandResult.failed(status);

    session.facts_asserted += 1;
    self.audit.writeAllowed(now, session.id, session.user_id, .fact_assert, cmd.target_kb_id, cmd.target_slot_id);

    return CommandResult.ok(cmd.target_kb_id, cmd.target_slot_id);
}

fn executeKbQuery(self: *CommandProcessor, session: *types.Session, cmd: *const types.Command, now: i32) CommandResult {
    _ = now;

    if (cmd.target_slot_id >= 0) {
        // Direct slot read
        const fact = self.kb_store.factRead(cmd.target_kb_id, cmd.target_slot_id) orelse
            return CommandResult.failed(types.Status.err(.kb, .slot_empty, cmd.target_slot_id));
        _ = fact;
        _ = session;
        return CommandResult.ok(cmd.target_kb_id, cmd.target_slot_id);
    }

    // Tag-based scan
    const tag: types.FactTag = if (cmd.args_count > 0) @enumFromInt(self.arg_buf[0]) else .value;
    const results = self.kb_store.factScanByTag(cmd.target_kb_id, tag, 100);
    if (results.count == 0) {
        return CommandResult.failed(types.Status.err(.kb, .slot_empty, cmd.target_kb_id));
    }

    return CommandResult.okWithBytes(cmd.target_kb_id, -1, results.count);
}

fn executeKbRetract(self: *CommandProcessor, session: *types.Session, cmd: *const types.Command, now: i32) CommandResult {
    const status = self.kb_store.factRetract(cmd.target_kb_id, cmd.target_slot_id);
    if (status.isErr()) return CommandResult.failed(status);

    session.facts_retracted += 1;
    self.audit.writeAllowed(now, session.id, session.user_id, .fact_retract, cmd.target_kb_id, cmd.target_slot_id);

    return CommandResult.ok(cmd.target_kb_id, cmd.target_slot_id);
}

fn executePrologQuery(self: *CommandProcessor, session: *types.Session, cmd: *const types.Command, now: i32) CommandResult {
    _ = now;

    // Build query term from args
    var query_term = types.Term.atom(if (cmd.args_count > 0) self.arg_buf[0] else 0);
    if (cmd.args_count > 1) {
        query_term = types.Term.compound(self.arg_buf[0], self.arg_buf[1], if (cmd.args_count > 2) self.arg_buf[2] else 0);
    }

    const result = self.prolog.query(cmd.target_kb_id, &query_term);
    session.prolog_queries += 1;

    if (result.result_count == 0) {
        return CommandResult.failed(types.Status.err(.prolog, .no_matching_rule, cmd.target_kb_id));
    }

    return CommandResult.okWithBytes(cmd.target_kb_id, -1, result.result_count);
}

fn executePrologAssertRule(self: *CommandProcessor, session: *types.Session, cmd: *const types.Command, now: i32) CommandResult {
    _ = now;

    // Parse args into head + body terms
    // Convention: arg[0] = head atom, arg[1] = body count, arg[2..] = body atoms
    if (cmd.args_count < 1) return CommandResult.failed(types.Status.err(.prolog, .unification_failed, 0));

    const head = types.Term.atom(self.arg_buf[0]);
    const body_count: usize = if (cmd.args_count > 1) @intCast(self.arg_buf[1]) else 0;

    var body_terms: [16]types.Term = undefined;
    const actual_body = @min(body_count, 16);
    for (0..actual_body) |i| {
        if (i + 2 < @as(usize, @intCast(cmd.args_count))) {
            body_terms[i] = types.Term.atom(self.arg_buf[i + 2]);
        }
    }

    const rule_id = self.prolog.ruleAssert(cmd.target_kb_id, &head, body_terms[0..actual_body], &.{});
    if (rule_id < 0) return CommandResult.failed(types.Status.err(.prolog, .no_matching_rule, cmd.target_kb_id));

    session.rules_fired += 1; // counted as rule-related activity
    self.audit.writeAllowed(currentTimestamp(), session.id, session.user_id, .rule_assert, cmd.target_kb_id, rule_id);

    return CommandResult.ok(cmd.target_kb_id, rule_id);
}

fn executeBuiltinCall(self: *CommandProcessor, session: *types.Session, cmd: *const types.Command, now: i32) CommandResult {
    _ = now;

    const args = builtin_mod.BuiltinArgs{
        .input_kb_id = cmd.target_kb_id,
        .input_slot_ids = if (cmd.args_count > 0) self.arg_buf[0..@intCast(@min(cmd.args_count, self.arg_capacity))] else &.{},
        .output_kb_id = cmd.target_kb_id,
        .output_slot_id = cmd.target_slot_id,
        .extra_params = &.{},
        .input_array_length = cmd.args_count,
    };

    const result = self.builtins.dispatch(cmd.builtin_id, &args);
    session.primitive_calls += 1;

    if (result.status.isErr()) return CommandResult.failed(result.status);
    return CommandResult.ok(result.output_kb_id, result.output_slot_id);
}

fn executeGrammarRender(self: *CommandProcessor, session: *types.Session, cmd: *const types.Command, now: i32) CommandResult {
    _ = now;

    // Load grammar through inheritance
    const grammar_slot: i32 = if (cmd.args_count > 0) self.arg_buf[0] else 0;
    const grammar = self.grammar.inherit(cmd.target_kb_id, grammar_slot) orelse
        return CommandResult.failed(types.Status.err(.grammar, .invalid_template, cmd.target_kb_id));

    // Compile grammar (or use cached compiled form)
    var template_buf: [16384]u8 = undefined;
    const tlen: usize = @intCast(@min(grammar.template_length, 16384));
    _ = self.kb_store.textRead(grammar.template_offset, @intCast(tlen), template_buf[0..tlen]);

    const compiled = self.grammar.compile(template_buf[0..tlen], grammar.id, session.id);
    if (compiled.status.isErr()) return CommandResult.failed(compiled.status);

    // Build mappings from command args: pairs of (kb_id, slot_id)
    var mappings: [32]types.GrammarKbMapping = undefined;
    const n_mappings = @min(@divTrunc(cmd.args_count, 2), 32);
    for (0..@intCast(n_mappings)) |i| {
        const base = @as(usize, i) * 2 + 1; // skip grammar_slot arg
        mappings[i] = .{
            .slot_index = @intCast(i),
            .kb_id = if (base < @as(usize, @intCast(cmd.args_count))) self.arg_buf[base] else cmd.target_kb_id,
            .slot_id = if (base + 1 < @as(usize, @intCast(cmd.args_count))) self.arg_buf[base + 1] else 0,
        };
    }

    const config = grammar_mod.RenderConfig{};
    const rendered = self.grammar.renderFromKb(&grammar, &compiled, mappings[0..@intCast(n_mappings)], &config, self.render_buf);

    session.grammar_renders += 1;

    if (rendered <= 0) return CommandResult.failed(types.Status.err(.grammar, .render_capacity_exceeded, 0));

    return .{
        .status = types.Status.ok(),
        .output_kb_id = cmd.target_kb_id,
        .output_slot_id = -1,
        .output_bytes = rendered,
        .output_text = self.render_buf[0..@intCast(rendered)],
    };
}

fn executeDirectOutput(self: *CommandProcessor, session: *types.Session, cmd: *const types.Command, now: i32) CommandResult {
    _ = now;
    _ = session;

    // Load fact from KB
    const fact = self.kb_store.factRead(cmd.target_kb_id, cmd.target_slot_id) orelse
        return CommandResult.failed(types.Status.err(.kb, .slot_empty, cmd.target_slot_id));

    // Try to find associated grammar
    if (self.grammar.inherit(cmd.target_kb_id, 0)) |grammar| {
        var template_buf: [16384]u8 = undefined;
        const tlen: usize = @intCast(@min(grammar.template_length, 16384));
        _ = self.kb_store.textRead(grammar.template_offset, @intCast(tlen), template_buf[0..tlen]);
        const compiled = self.grammar.compile(template_buf[0..tlen], grammar.id, 0);
        if (compiled.status.isOk()) {
            const mapping = [_]types.GrammarKbMapping{.{
                .slot_index = 0,
                .kb_id = cmd.target_kb_id,
                .slot_id = cmd.target_slot_id,
            }};
            const config = grammar_mod.RenderConfig{};
            const rendered = self.grammar.renderFromKb(&grammar, &compiled, &mapping, &config, self.render_buf);
            if (rendered > 0) {
                return .{
                    .status = types.Status.ok(),
                    .output_kb_id = cmd.target_kb_id,
                    .output_slot_id = cmd.target_slot_id,
                    .output_bytes = rendered,
                    .output_text = self.render_buf[0..@intCast(rendered)],
                };
            }
        }
    }

    // No grammar — render fact value as text
    const rendered = grammar_mod.q16ToString(fact.value, self.render_buf);
    return .{
        .status = types.Status.ok(),
        .output_kb_id = cmd.target_kb_id,
        .output_slot_id = cmd.target_slot_id,
        .output_bytes = rendered,
        .output_text = self.render_buf[0..@intCast(rendered)],
    };
}

fn executeSessionSnapshot(self: *CommandProcessor, handle: types.SessionHandle, now: i32) CommandResult {
    _ = now;
    _ = self;
    // Delegate to session manager — actual snapshot logic in vlp_session + vlp_snapshot
    return CommandResult.ok(-1, handle.id);
}

fn executeSessionClone(self: *CommandProcessor, handle: types.SessionHandle, now: i32) CommandResult {
    _ = now;
    const config = session_mod.CloneConfig{};
    const child = self.session_mgr.clone(handle, &config) orelse
        return CommandResult.failed(types.Status.err(.session, .clone_failed, handle.id));
    return CommandResult.ok(-1, child.id);
}

fn executeOperational(self: *CommandProcessor, session: *types.Session, cmd: *const types.Command, now: i32) CommandResult {
    _ = now;
    // Grant already checked. Dispatch to builtin executor for operational builtins.
    const args = builtin_mod.BuiltinArgs{
        .input_kb_id = cmd.target_kb_id,
        .input_slot_ids = if (cmd.args_count > 0) self.arg_buf[0..@intCast(@min(cmd.args_count, self.arg_capacity))] else &.{},
        .output_kb_id = cmd.target_kb_id,
        .output_slot_id = cmd.target_slot_id,
        .extra_params = &.{},
        .input_array_length = cmd.args_count,
    };

    const result = self.builtins.dispatch(cmd.builtin_id, &args);
    session.primitive_calls += 1;

    self.audit.writeAllowed(currentTimestamp(), session.id, session.user_id, .op_execute, cmd.target_kb_id, cmd.target_slot_id);

    if (result.status.isErr()) return CommandResult.failed(result.status);
    return CommandResult.ok(result.output_kb_id, result.output_slot_id);
}

// ============================================================
// Helpers
// ============================================================

fn tokenToCommandType(token: i32) ?types.CommandType {
    if (token < 0 or token > 14) return null;
    return @enumFromInt(@as(i8, @intCast(token)));
}

fn parseSlotId(field: []const u8) i32 {
    var result: i32 = 0;
    for (field) |c| {
        if (c >= '0' and c <= '9') {
            result = result * 10 + @as(i32, c - '0');
        } else break;
    }
    return result;
}

fn currentTimestamp() i32 {
    const ts = std.time.timestamp();
    return @intCast(@min(ts, std.math.maxInt(i32)));
}
