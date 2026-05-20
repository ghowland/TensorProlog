// ============================================================
// src/engine/command_exec.zig
// ============================================================

const vdr_types = @import("../vdr/types.zig");
const q16_mod = @import("../vdr/q16.zig");
const kb_types = @import("../kb/types.zig");
const store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const vis_mod = @import("../kb/visibility.zig");
const grant_mod = @import("../safety/grant.zig");
const audit_mod = @import("../safety/audit.zig");
const safety_types = @import("../safety/types.zig");
const rule_mod = @import("../prolog/rule.zig");
const prolog_types = @import("../prolog/types.zig");
const grammar_render = @import("../grammar/render.zig");
const grammar_inherit = @import("../grammar/inherit.zig");
const grammar_types = @import("../grammar/types.zig");
const scratchpad_mod = @import("scratchpad.zig");
const cmd_parse = @import("command_parse.zig");
const conf_mod = @import("../confidence/propagate.zig");

const VlpStatus = vdr_types.VlpStatus;
const VlpCommandType = vdr_types.VlpCommandType;
const VlpGrantClass = vdr_types.VlpGrantClass;
const VlpAuditAction = vdr_types.VlpAuditAction;
const VlpVisibility = vdr_types.VlpVisibility;
const Q16 = q16_mod.Q16;
const VlpFact = kb_types.VlpFact;
const VlpProvenance = kb_types.VlpProvenance;
const KBStore = store_mod.KBStore;
const GrantStore = grant_mod.GrantStore;
const AuditRing = audit_mod.AuditRing;
const RuleStore = rule_mod.RuleStore;
const GrammarStore = grammar_inherit.GrammarStore;
const Scratchpad = scratchpad_mod.Scratchpad;
const VlpCommand = cmd_parse.VlpCommand;

pub const CommandResult = struct {
    status: VlpStatus = .ok,
    result_kb_id: i32 = -1,
    result_slot_id: i32 = -1,
    rule_fired: bool = false,
    output_len: i32 = 0,
};

pub const ExecContext = struct {
    session_id: i32,
    user_id: i32,
    user_vis: VlpVisibility,
    kb_store: *KBStore,
    rule_store: *RuleStore,
    grant_store: *GrantStore,
    grammar_store: *GrammarStore,
    audit: *AuditRing,
    scratchpad: *Scratchpad,
    output_buf: []u8,
    now: i32,
};

pub fn execute(ctx: *ExecContext, cmd: *const VlpCommand) CommandResult {
    var result = CommandResult{};

    if (cmd.target_kb_id >= 0) {
        if (!vis_mod.checkAccess(ctx.kb_store, ctx.user_id, ctx.user_vis, cmd.target_kb_id)) {
            audit_mod.writeAudit(ctx.audit, ctx.now, ctx.session_id, ctx.user_id, .access_denied, cmd.target_kb_id, cmd.target_slot_id, -1, 0);
            ctx.scratchpad.writeDenied(cmd.target_kb_id);
            result.status = .err_kb_access_denied;
            return result;
        }
    }

    if (cmd.grant_required >= 0) {
        const gc: VlpGrantClass = @enumFromInt(cmd.grant_required);
        const check = ctx.grant_store.check(ctx.user_id, gc, "", ctx.now);
        if (!check.granted) {
            audit_mod.writeAudit(ctx.audit, ctx.now, ctx.session_id, ctx.user_id, .grant_check, cmd.target_kb_id, -1, -1, 0);
            ctx.scratchpad.writeGrantDenied(gc);
            result.status = .err_grant_denied;
            return result;
        }
        audit_mod.writeAudit(ctx.audit, ctx.now, ctx.session_id, ctx.user_id, .grant_check, cmd.target_kb_id, -1, check.grant_id, 1);
    }

    switch (cmd.cmd_type) {
        .kb_assert => return execKBAssert(ctx, cmd),
        .kb_query => return execKBQuery(ctx, cmd),
        .kb_retract => return execKBRetract(ctx, cmd),
        .prolog_query => return execPrologQuery(ctx, cmd),
        .prolog_assert_rule => return execPrologAssertRule(ctx, cmd),
        .builtin_call => return execBuiltinCall(ctx, cmd),
        .grammar_render => return execGrammarRender(ctx, cmd),
        .direct_output => return execDirectOutput(ctx, cmd),
        .session_snapshot, .session_clone => {
            result.status = .ok;
            return result;
        },
        .op_filesystem, .op_compile, .op_execute, .op_network, .op_process => {
            result.status = .ok;
            return result;
        },
    }
}

fn execKBAssert(ctx: *ExecContext, cmd: *const VlpCommand) CommandResult {
    var result = CommandResult{};
    var slot = cmd.target_slot_id;
    if (slot < 0) {
        slot = fact_mod.firstEmpty(ctx.kb_store, cmd.target_kb_id) orelse {
            result.status = .err_kb_full;
            return result;
        };
    }

    const prov = VlpProvenance{
        .source_type = .llm_generated,
        .confidence = conf_mod.assignFromSource(.llm_generated),
        .timestamp = ctx.now,
        .source_kb_id = cmd.target_kb_id,
        .source_slot_id = slot,
    };

    const fact = VlpFact{
        .tag = .value,
        .value = cmd.args_q[0],
        .provenance = prov,
    };

    result.status = fact_mod.assert(ctx.kb_store, cmd.target_kb_id, slot, &fact);
    result.result_kb_id = cmd.target_kb_id;
    result.result_slot_id = slot;

    audit_mod.writeAudit(ctx.audit, ctx.now, ctx.session_id, ctx.user_id, .fact_assert, cmd.target_kb_id, slot, -1, 1);
    ctx.scratchpad.writeInt(slot);
    return result;
}

fn execKBQuery(ctx: *ExecContext, cmd: *const VlpCommand) CommandResult {
    var result = CommandResult{};

    if (cmd.target_slot_id >= 0) {
        const fact = fact_mod.query(ctx.kb_store, cmd.target_kb_id, cmd.target_slot_id);
        if (fact) |f| {
            ctx.scratchpad.writeValue(f.value);
            result.result_kb_id = cmd.target_kb_id;
            result.result_slot_id = cmd.target_slot_id;
        }
    } else {
        var results: [16]VlpFact = undefined;
        const tag_arg = cmd.args_i[0];
        const tag: kb_types.VlpFactTag = @enumFromInt(@as(i8, @intCast(tag_arg)));
        const found = fact_mod.search(ctx.kb_store, cmd.target_kb_id, tag, &results);
        var i: i32 = 0;
        while (i < found) : (i += 1) {
            ctx.scratchpad.writeValue(results[@intCast(i)].value);
        }
    }

    audit_mod.writeAudit(ctx.audit, ctx.now, ctx.session_id, ctx.user_id, .fact_query, cmd.target_kb_id, cmd.target_slot_id, -1, 1);
    return result;
}

fn execKBRetract(ctx: *ExecContext, cmd: *const VlpCommand) CommandResult {
    var result = CommandResult{};
    result.status = fact_mod.retract(ctx.kb_store, cmd.target_kb_id, cmd.target_slot_id);
    audit_mod.writeAudit(ctx.audit, ctx.now, ctx.session_id, ctx.user_id, .fact_retract, cmd.target_kb_id, cmd.target_slot_id, -1, 1);
    return result;
}

fn execPrologQuery(ctx: *ExecContext, cmd: *const VlpCommand) CommandResult {
    var result = CommandResult{};
    _ = cmd;

    var fired_buf: [16]prolog_types.PrologFired = undefined;
    const n = ctx.rule_store.fireAll(ctx.kb_store, ctx.kb_store.kbs[0].id, ctx.now, &fired_buf);
    if (n > 0) result.rule_fired = true;

    audit_mod.writeAudit(ctx.audit, ctx.now, ctx.session_id, ctx.user_id, .fact_query, -1, -1, -1, 1);
    ctx.scratchpad.writeInt(n);
    return result;
}

fn execPrologAssertRule(ctx: *ExecContext, cmd: *const VlpCommand) CommandResult {
    var result = CommandResult{};

    const head_idx = cmd.args_i[0];
    const rule_id = ctx.rule_store.assertRule(head_idx, &[_]i32{}, &[_]prolog_types.PrologAction{}, ctx.now, ctx.session_id) orelse {
        result.status = .err_out_of_memory;
        return result;
    };

    result.result_slot_id = rule_id;
    audit_mod.writeAudit(ctx.audit, ctx.now, ctx.session_id, ctx.user_id, .rule_assert, cmd.target_kb_id, rule_id, -1, 1);
    ctx.scratchpad.writeInt(rule_id);
    return result;
}

fn execBuiltinCall(ctx: *ExecContext, cmd: *const VlpCommand) CommandResult {
    var result = CommandResult{};
    result.result_kb_id = cmd.target_kb_id;
    _ = ctx;
    return result;
}

fn execGrammarRender(ctx: *ExecContext, cmd: *const VlpCommand) CommandResult {
    var result = CommandResult{};

    const grammar = grammar_inherit.inherit(ctx.grammar_store, ctx.kb_store, cmd.target_kb_id);
    if (grammar) |g| {
        const fills = [_]grammar_types.GrammarFill{};
        const rr = grammar_render.render(g, &ctx.kb_store.text, &fills, ctx.output_buf);
        result.output_len = rr.len;
        result.status = rr.status;
    }

    audit_mod.writeAudit(ctx.audit, ctx.now, ctx.session_id, ctx.user_id, .fact_query, cmd.target_kb_id, -1, -1, 1);
    return result;
}

fn execDirectOutput(ctx: *ExecContext, cmd: *const VlpCommand) CommandResult {
    var result = CommandResult{};

    if (cmd.target_slot_id >= 0) {
        const fact = fact_mod.query(ctx.kb_store, cmd.target_kb_id, cmd.target_slot_id);
        if (fact) |f| {
            const grammar = grammar_inherit.inherit(ctx.grammar_store, ctx.kb_store, cmd.target_kb_id);
            if (grammar) |g| {
                const fills = [_]grammar_types.GrammarFill{
                    .{ .slot_index = 0, .fill_type = .vdr_value, .vdr_value = f.value },
                };
                const rr = grammar_render.render(g, &ctx.kb_store.text, &fills, ctx.output_buf);
                result.output_len = rr.len;
                result.status = rr.status;
            } else {
                const wrote = renderFactDirect(f, ctx.output_buf);
                result.output_len = wrote;
            }
            result.result_kb_id = cmd.target_kb_id;
            result.result_slot_id = cmd.target_slot_id;
        }
    }

    return result;
}

fn renderFactDirect(fact: VlpFact, output: []u8) i32 {
    var buf: [12]u8 = undefined;
    var v = fact.value.v;
    var negative = false;
    if (v < 0) {
        negative = true;
        v = -v;
    }
    if (v == 0) {
        if (output.len > 0) {
            output[0] = '0';
            return 1;
        }
        return 0;
    }
    var len: i32 = 0;
    while (v > 0) : (len += 1) {
        buf[@intCast(len)] = @intCast(@as(u32, @intCast(@mod(v, 10))) + '0');
        v = @divTrunc(v, 10);
    }
    if (negative) {
        buf[@intCast(len)] = '-';
        len += 1;
    }
    if (len > @as(i32, @intCast(output.len))) return 0;
    var i: i32 = 0;
    while (i < len) : (i += 1) {
        output[@intCast(i)] = buf[@intCast(len - 1 - i)];
    }
    return len;
}
