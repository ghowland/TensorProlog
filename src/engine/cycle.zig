// ============================================================
// src/engine/cycle.zig
// ============================================================

const vdr_types = @import("../vdr/types.zig");
const q16_mod = @import("../vdr/q16.zig");
const session_types = @import("../session/types.zig");
const session_life = @import("../session/lifecycle.zig");
const store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const rule_mod = @import("../prolog/rule.zig");
const prolog_types = @import("../prolog/types.zig");
const grant_mod = @import("../safety/grant.zig");
const audit_mod = @import("../safety/audit.zig");
const grammar_inherit = @import("../grammar/inherit.zig");
const grammar_render = @import("../grammar/render.zig");
const grammar_types = @import("../grammar/types.zig");
const context_mod = @import("context.zig");
const scratchpad_mod = @import("scratchpad.zig");
const token_classify = @import("token_classify.zig");
const level_stats_mod = @import("level_stats.zig");
const auto_resolve = @import("auto_resolve.zig");
const cmd_parse = @import("command_parse.zig");
const cmd_exec = @import("command_exec.zig");

const VlpStatus = vdr_types.VlpStatus;
const VlpExecutionLevel = vdr_types.VlpExecutionLevel;
const VlpTokenClass = vdr_types.VlpTokenClass;
const Q16 = q16_mod.Q16;
const VlpSession = session_types.VlpSession;
const SessionStore = session_life.SessionStore;
const KBStore = store_mod.KBStore;
const RuleStore = rule_mod.RuleStore;
const PrologFired = prolog_types.PrologFired;
const GrantStore = grant_mod.GrantStore;
const AuditRing = audit_mod.AuditRing;
const GrammarStore = grammar_inherit.GrammarStore;
const VlpContext = context_mod.VlpContext;
const VlpInput = context_mod.VlpInput;
const Scratchpad = scratchpad_mod.Scratchpad;
const TokenClassConfig = token_classify.TokenClassConfig;
const LevelStats = level_stats_mod.LevelStats;
const VlpCommand = cmd_parse.VlpCommand;
const ExecContext = cmd_exec.ExecContext;

pub const CycleResult = struct {
    status: VlpStatus = .ok,
    tokens_consumed: i32 = 0,
    commands_executed: i32 = 0,
    level: VlpExecutionLevel = .l1,
    should_recycle: bool = false,
    output_len: i32 = 0,
};

pub const CycleConfig = struct {
    max_tokens_per_turn: i32 = 512,
    auto_resolve_threshold: Q16 = Q16{ .v = 58982, .r0 = 0 },
    token_config: TokenClassConfig = .{},
    system_prompt: []const i32 = &[_]i32{},
};

pub const CycleState = struct {
    session_store: *SessionStore,
    kb_store: *KBStore,
    rule_store: *RuleStore,
    grant_store: *GrantStore,
    grammar_store: *GrammarStore,
    audit: *AuditRing,
    level_stats: *LevelStats,
    config: CycleConfig,
};

pub const LLMInterface = struct {
    generateFn: *const fn (ctx: *LLMGenContext) i32,
    context: *anyopaque,
};

pub const LLMGenContext = struct {
    llm: *LLMInterface,
    step: i32,
};

pub const VlpOutputBuffer = struct {
    buf: []u8,
    len: i32,

    pub fn init(buf: []u8) VlpOutputBuffer {
        return .{ .buf = buf, .len = 0 };
    }

    pub fn writeBytes(self: *VlpOutputBuffer, data: []const u8) void {
        for (data) |b| {
            if (self.len >= @as(i32, @intCast(self.buf.len))) return;
            self.buf[@intCast(self.len)] = b;
            self.len += 1;
        }
    }

    pub fn writeByte(self: *VlpOutputBuffer, b: u8) void {
        if (self.len >= @as(i32, @intCast(self.buf.len))) return;
        self.buf[@intCast(self.len)] = b;
        self.len += 1;
    }

    pub fn remaining(self: *const VlpOutputBuffer) i32 {
        return @as(i32, @intCast(self.buf.len)) - self.len;
    }
};

pub fn vlpCycle(
    state: *CycleState,
    session_id: i32,
    input: *const VlpInput,
    output: *VlpOutputBuffer,
    scratchpad: *Scratchpad,
    llm: *LLMInterface,
    now: i32,
) CycleResult {
    var result = CycleResult{};
    scratchpad.clear();

    const session = state.session_store.get(session_id) orelse {
        result.status = .err_kb_not_found;
        return result;
    };

    // phase 0: pre-LLM rule evaluation
    var fired_buf: [64]PrologFired = undefined;
    const n_fired = state.rule_store.fireAll(state.kb_store, session.kb_root_id, now, &fired_buf);

    if (n_fired > 0) {
        var output_phase0: [4096]u8 = undefined;
        const resolution = auto_resolve.check(
            &fired_buf,
            n_fired,
            state.config.auto_resolve_threshold,
            state.grammar_store,
            state.kb_store,
            &output_phase0,
        );

        if (auto_resolve.isFullyResolved(&resolution)) {
            state.rule_store.fireAndCommit(state.kb_store, session.kb_root_id, now, &fired_buf);
            output.writeBytes(output_phase0[0..@intCast(resolution.output_len)]);
            result.level = .l3;
            result.output_len = resolution.output_len;
            result.commands_executed = resolution.n_actions;
            state.level_stats.update(.l3, 0);
            state.session_store.updateLevel(session_id, .l3, 0);
            state.session_store.incrementTurn(session_id);
            result.should_recycle = state.session_store.shouldRecycle(session_id);
            return result;
        }

        for (fired_buf[0..@intCast(n_fired)]) |*f| {
            var ai: i32 = 0;
            while (ai < f.action_count) : (ai += 1) {
                scratchpad.writeInt(f.rule_id);
            }
        }
    }

    // phase 1: context assembly
    var ctx: VlpContext = .{};
    context_mod.build(session, input, state.kb_store, state.config.system_prompt, scratchpad.getTokens(), &ctx);

    // phase 2: LLM generation + command dispatch
    var tokens_generated: i32 = 0;
    var commands_executed: i32 = 0;
    var level: VlpExecutionLevel = .l1;

    var gen_ctx = LLMGenContext{ .llm = llm, .step = 0 };

    while (tokens_generated < state.config.max_tokens_per_turn) {
        const next_token = llm.generateFn(&gen_ctx);
        gen_ctx.step += 1;
        tokens_generated += 1;

        const tc = token_classify.classify(next_token, &state.config.token_config);

        switch (tc) {
            .end_of_turn => break,

            .command_start => {
                var cmd_tokens: [32]i32 = undefined;
                cmd_tokens[0] = next_token;
                var cmd_len: i32 = 1;
                while (cmd_len < 32) {
                    const ct = llm.generateFn(&gen_ctx);
                    gen_ctx.step += 1;
                    tokens_generated += 1;
                    cmd_tokens[@intCast(cmd_len)] = ct;
                    cmd_len += 1;
                    if (ct == cmd_parse.COMMAND_TOKENS.END_COMMAND) break;
                }

                var cmd: VlpCommand = .{};
                const parse_status = cmd_parse.parse(cmd_tokens[0..@intCast(cmd_len)], state.kb_store, &cmd);
                if (parse_status != .ok) {
                    scratchpad.writeError(parse_status);
                    continue;
                }

                var exec_ctx = ExecContext{
                    .session_id = session_id,
                    .user_id = session.user_id,
                    .user_vis = session.visibility_level,
                    .kb_store = state.kb_store,
                    .rule_store = state.rule_store,
                    .grant_store = state.grant_store,
                    .grammar_store = state.grammar_store,
                    .audit = state.audit,
                    .scratchpad = scratchpad,
                    .output_buf = output.buf[@intCast(output.len)..],
                    .now = now,
                };

                const cmd_result = cmd_exec.execute(&exec_ctx, &cmd);
                commands_executed += 1;

                if (cmd_result.output_len > 0) {
                    output.len += cmd_result.output_len;
                }

                if (cmd_result.rule_fired) level = .l2;
            },

            .direct_output => {
                var ref_tokens: [4]i32 = undefined;
                var ref_len: i32 = 0;
                while (ref_len < 4) {
                    const rt = llm.generateFn(&gen_ctx);
                    gen_ctx.step += 1;
                    tokens_generated += 1;
                    ref_tokens[@intCast(ref_len)] = rt;
                    ref_len += 1;
                    if (rt == cmd_parse.COMMAND_TOKENS.END_COMMAND) break;
                }

                if (ref_len >= 2) {
                    const ref_kb = ref_tokens[0];
                    const ref_slot = ref_tokens[1];
                    var direct_cmd = VlpCommand{
                        .cmd_type = .direct_output,
                        .target_kb_id = ref_kb,
                        .target_slot_id = ref_slot,
                    };

                    var exec_ctx = ExecContext{
                        .session_id = session_id,
                        .user_id = session.user_id,
                        .user_vis = session.visibility_level,
                        .kb_store = state.kb_store,
                        .rule_store = state.rule_store,
                        .grant_store = state.grant_store,
                        .grammar_store = state.grammar_store,
                        .audit = state.audit,
                        .scratchpad = scratchpad,
                        .output_buf = output.buf[@intCast(output.len)..],
                        .now = now,
                    };

                    const dr = cmd_exec.execute(&exec_ctx, &direct_cmd);
                    if (dr.output_len > 0) output.len += dr.output_len;
                }
            },

            .prose => {
                if (next_token > 0 and next_token < 128) {
                    output.writeByte(@intCast(next_token));
                }
            },
        }
    }

    // phase 3: post-cycle
    result.tokens_consumed = tokens_generated;
    result.commands_executed = commands_executed;
    result.level = level;
    result.output_len = output.len;

    state.level_stats.update(level, tokens_generated);
    state.session_store.updateLevel(session_id, level, tokens_generated);
    state.session_store.incrementTurn(session_id);
    result.should_recycle = state.session_store.shouldRecycle(session_id);

    return result;
}
