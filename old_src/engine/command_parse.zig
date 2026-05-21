// ============================================================
// src/engine/command_parse.zig
// ============================================================

const vdr_types = @import("../vdr/types.zig");
const q16_mod = @import("../vdr/q16.zig");
const store_mod = @import("../kb/store.zig");

const VlpCommandType = vdr_types.VlpCommandType;
const VlpStatus = vdr_types.VlpStatus;
const VlpGrantClass = vdr_types.VlpGrantClass;
const Q16 = q16_mod.Q16;
const KBStore = store_mod.KBStore;

pub const VlpCommand = struct {
    cmd_type: VlpCommandType = .kb_assert,
    target_kb_id: i32 = -1,
    target_slot_id: i32 = -1,
    builtin_id: i32 = -1,
    args_i: [8]i32 = .{0} ** 8,
    args_q: [4]Q16 = .{Q16.zero()} ** 4,
    args_text: [2][]const u8 = .{ &[_]u8{}, &[_]u8{} },
    arg_count: i32 = 0,
    grant_required: i8 = -1,
};

pub const COMMAND_TOKENS = struct {
    pub const KB_ASSERT: i32 = -950;
    pub const KB_QUERY: i32 = -949;
    pub const KB_RETRACT: i32 = -948;
    pub const PROLOG_QUERY: i32 = -947;
    pub const PROLOG_ASSERT_RULE: i32 = -946;
    pub const BUILTIN_CALL: i32 = -945;
    pub const GRAMMAR_RENDER: i32 = -944;
    pub const DIRECT_OUTPUT: i32 = -943;
    pub const OP_FILESYSTEM: i32 = -942;
    pub const OP_COMPILE: i32 = -941;
    pub const OP_EXECUTE: i32 = -940;
    pub const OP_NETWORK: i32 = -939;
    pub const OP_PROCESS: i32 = -938;
    pub const SESSION_SNAPSHOT: i32 = -937;
    pub const SESSION_CLONE: i32 = -936;
    pub const PATH_MARKER: i32 = -800;
    pub const SLOT_MARKER: i32 = -801;
    pub const ARG_INT: i32 = -802;
    pub const ARG_Q16: i32 = -803;
    pub const END_COMMAND: i32 = -899;
};

pub fn parse(tokens: []const i32, kb_store: *const KBStore, cmd: *VlpCommand) VlpStatus {
    cmd.* = VlpCommand{};
    if (tokens.len == 0) return .err_command_parse;

    const type_status = parseCommandType(tokens[0], cmd);
    if (type_status != .ok) return type_status;

    var pos: i32 = 1;
    const tlen: i32 = @intCast(tokens.len);

    if (pos < tlen and tokens[@intCast(pos)] == COMMAND_TOKENS.PATH_MARKER) {
        pos += 1;
        if (pos >= tlen) return .err_command_parse;
        cmd.target_kb_id = tokens[@intCast(pos)];
        pos += 1;
    }

    if (pos < tlen and tokens[@intCast(pos)] == COMMAND_TOKENS.SLOT_MARKER) {
        pos += 1;
        if (pos >= tlen) return .err_command_parse;
        cmd.target_slot_id = tokens[@intCast(pos)];
        pos += 1;
    }

    _ = kb_store;

    var arg_idx: i32 = 0;
    while (pos < tlen) {
        const tok = tokens[@intCast(pos)];
        if (tok == COMMAND_TOKENS.END_COMMAND) break;

        if (tok == COMMAND_TOKENS.ARG_INT) {
            pos += 1;
            if (pos >= tlen) return .err_command_parse;
            if (arg_idx < 8) {
                cmd.args_i[@intCast(arg_idx)] = tokens[@intCast(pos)];
                arg_idx += 1;
            }
        } else if (tok == COMMAND_TOKENS.ARG_Q16) {
            pos += 1;
            if (pos + 1 >= tlen) return .err_command_parse;
            const qi = @divTrunc(arg_idx, 2);
            if (qi < 4) {
                cmd.args_q[@intCast(qi)] = Q16{ .v = tokens[@intCast(pos)], .r0 = @intCast(tokens[@intCast(pos + 1)]) };
            }
            pos += 1;
        } else {
            if (arg_idx < 8) {
                cmd.args_i[@intCast(arg_idx)] = tok;
                arg_idx += 1;
            }
        }
        pos += 1;
    }
    cmd.arg_count = arg_idx;

    setGrantRequired(cmd);
    return .ok;
}

fn parseCommandType(token: i32, cmd: *VlpCommand) VlpStatus {
    if (token == COMMAND_TOKENS.KB_ASSERT) {
        cmd.cmd_type = .kb_assert;
        return .ok;
    }
    if (token == COMMAND_TOKENS.KB_QUERY) {
        cmd.cmd_type = .kb_query;
        return .ok;
    }
    if (token == COMMAND_TOKENS.KB_RETRACT) {
        cmd.cmd_type = .kb_retract;
        return .ok;
    }
    if (token == COMMAND_TOKENS.PROLOG_QUERY) {
        cmd.cmd_type = .prolog_query;
        return .ok;
    }
    if (token == COMMAND_TOKENS.PROLOG_ASSERT_RULE) {
        cmd.cmd_type = .prolog_assert_rule;
        return .ok;
    }
    if (token == COMMAND_TOKENS.BUILTIN_CALL) {
        cmd.cmd_type = .builtin_call;
        return .ok;
    }
    if (token == COMMAND_TOKENS.GRAMMAR_RENDER) {
        cmd.cmd_type = .grammar_render;
        return .ok;
    }
    if (token == COMMAND_TOKENS.DIRECT_OUTPUT) {
        cmd.cmd_type = .direct_output;
        return .ok;
    }
    if (token == COMMAND_TOKENS.OP_FILESYSTEM) {
        cmd.cmd_type = .op_filesystem;
        return .ok;
    }
    if (token == COMMAND_TOKENS.OP_COMPILE) {
        cmd.cmd_type = .op_compile;
        return .ok;
    }
    if (token == COMMAND_TOKENS.OP_EXECUTE) {
        cmd.cmd_type = .op_execute;
        return .ok;
    }
    if (token == COMMAND_TOKENS.OP_NETWORK) {
        cmd.cmd_type = .op_network;
        return .ok;
    }
    if (token == COMMAND_TOKENS.OP_PROCESS) {
        cmd.cmd_type = .op_process;
        return .ok;
    }
    if (token == COMMAND_TOKENS.SESSION_SNAPSHOT) {
        cmd.cmd_type = .session_snapshot;
        return .ok;
    }
    if (token == COMMAND_TOKENS.SESSION_CLONE) {
        cmd.cmd_type = .session_clone;
        return .ok;
    }
    return .err_command_parse;
}

fn setGrantRequired(cmd: *VlpCommand) void {
    cmd.grant_required = switch (cmd.cmd_type) {
        .op_filesystem => @intFromEnum(VlpGrantClass.filesystem),
        .op_compile => @intFromEnum(VlpGrantClass.compile),
        .op_execute => @intFromEnum(VlpGrantClass.execute),
        .op_network => @intFromEnum(VlpGrantClass.network),
        .op_process => @intFromEnum(VlpGrantClass.process),
        else => -1,
    };
}
