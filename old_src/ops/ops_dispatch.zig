// ============================================================
// src/ops/ops_dispatch.zig
// ============================================================

const dispatch_mod = @import("../builtins/dispatch.zig");
const BuiltinArgs = dispatch_mod.BuiltinArgs;
const BuiltinResult = dispatch_mod.BuiltinResult;
const BuiltinTable = dispatch_mod.BuiltinTable;

const filesystem = @import("filesystem.zig");
const network = @import("network.zig");
const execute = @import("execute.zig");
const compile_check = @import("compile_check.zig");
const process = @import("process.zig");

pub fn builtinFsRead(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const result = filesystem.fsRead(args.text_in[0], args.text_out);
    if (result.status != .ok) return dispatch_mod.errorResult(result.status);
    args.text_out_len.* = result.len;
    return dispatch_mod.emptyResult();
}

pub fn builtinFsWrite(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    const status = filesystem.fsWrite(args.text_in[0], args.text_in[1]);
    if (status != .ok) return dispatch_mod.errorResult(status);
    return dispatch_mod.emptyResult();
}

pub fn builtinFsAppend(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    const status = filesystem.fsAppend(args.text_in[0], args.text_in[1]);
    if (status != .ok) return dispatch_mod.errorResult(status);
    return dispatch_mod.emptyResult();
}

pub fn builtinFsDelete(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const status = filesystem.fsDelete(args.text_in[0]);
    if (status != .ok) return dispatch_mod.errorResult(status);
    return dispatch_mod.emptyResult();
}

pub fn builtinFsStat(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const result = filesystem.fsStat(args.text_in[0]);
    return dispatch_mod.intResult(@intCast(result.size));
}

pub fn builtinNetFetch(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const result = network.netFetch(args.text_in[0], args.text_out);
    if (result.status != .ok) return dispatch_mod.errorResult(result.status);
    args.text_out_len.* = result.len;
    return dispatch_mod.emptyResult();
}

pub fn builtinExecRun(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const empty_args = [_][]const u8{};
    const result = execute.execRun(args.text_in[0], &empty_args, args.text_out);
    if (result.status != .ok) return dispatch_mod.errorResult(result.status);
    args.text_out_len.* = result.len;
    return dispatch_mod.intResult(result.exit_code);
}

pub fn builtinCompileCheck(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const result = compile_check.compileCheck(args.text_in[0]);
    return dispatch_mod.boolResult(result.valid);
}

pub fn builtinProcStart(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const handle = process.procStart(args.text_in[0]);
    return dispatch_mod.intResult(handle.pid);
}

pub fn builtinProcKill(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.emptyResult();
}

pub fn builtinProcStatus(args: *BuiltinArgs) BuiltinResult {
    _ = args;
    return dispatch_mod.boolResult(false);
}

pub fn registerOpsBuiltins(table: *BuiltinTable) void {
    table.register(500, "fs_read", builtinFsRead, false, 1, .text);
    table.register(501, "fs_write", builtinFsWrite, false, 2, .empty);
    table.register(502, "fs_append", builtinFsAppend, false, 2, .empty);
    table.register(503, "fs_delete", builtinFsDelete, false, 1, .empty);
    table.register(504, "fs_stat", builtinFsStat, false, 1, .value);
    table.register(505, "net_fetch", builtinNetFetch, false, 1, .text);
    table.register(506, "exec_run", builtinExecRun, false, 1, .text);
    table.register(507, "compile_check", builtinCompileCheck, false, 1, .boolean);
    table.register(508, "proc_start", builtinProcStart, false, 1, .value);
    table.register(509, "proc_kill", builtinProcKill, false, 1, .empty);
    table.register(510, "proc_status", builtinProcStatus, false, 1, .boolean);
}
