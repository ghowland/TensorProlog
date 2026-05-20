// ============================================================
// src/ops/compile_op.zig
// ============================================================

const types_op = @import("../vdr/types.zig");
const VlpStatusOp = types_op.VlpStatus;

pub const CompileResult = struct {
    valid: bool,
    error_msg: [256]u8,
    error_len: i32,
    brace_depth: i32,
    paren_depth: i32,
    bracket_depth: i32,
};

pub fn compileCheck(source: []const u8) CompileResult {
    var result = CompileResult{
        .valid = true,
        .error_msg = undefined,
        .error_len = 0,
        .brace_depth = 0,
        .paren_depth = 0,
        .bracket_depth = 0,
    };

    var in_string = false;
    var in_line_comment = false;
    var prev_char: u8 = 0;

    for (source) |c| {
        if (in_line_comment) {
            if (c == '\n') in_line_comment = false;
            prev_char = c;
            continue;
        }

        if (c == '/' and prev_char == '/') {
            in_line_comment = true;
            prev_char = c;
            continue;
        }

        if (c == '"' and prev_char != '\\') {
            in_string = !in_string;
            prev_char = c;
            continue;
        }

        if (in_string) {
            prev_char = c;
            continue;
        }

        switch (c) {
            '{' => result.brace_depth += 1,
            '}' => {
                result.brace_depth -= 1;
                if (result.brace_depth < 0) {
                    result.valid = false;
                    setError(&result, "unmatched closing brace");
                    return result;
                }
            },
            '(' => result.paren_depth += 1,
            ')' => {
                result.paren_depth -= 1;
                if (result.paren_depth < 0) {
                    result.valid = false;
                    setError(&result, "unmatched closing paren");
                    return result;
                }
            },
            '[' => result.bracket_depth += 1,
            ']' => {
                result.bracket_depth -= 1;
                if (result.bracket_depth < 0) {
                    result.valid = false;
                    setError(&result, "unmatched closing bracket");
                    return result;
                }
            },
            else => {},
        }

        prev_char = c;
    }

    if (in_string) {
        result.valid = false;
        setError(&result, "unterminated string");
        return result;
    }

    if (result.brace_depth != 0) {
        result.valid = false;
        setError(&result, "unclosed brace");
        return result;
    }

    if (result.paren_depth != 0) {
        result.valid = false;
        setError(&result, "unclosed paren");
        return result;
    }

    if (result.bracket_depth != 0) {
        result.valid = false;
        setError(&result, "unclosed bracket");
        return result;
    }

    return result;
}

pub fn compileCheckWithLanguage(source: []const u8, language: []const u8) CompileResult {
    _ = language;
    return compileCheck(source);
}

fn setError(result: *CompileResult, msg: []const u8) void {
    const n = @min(msg.len, result.error_msg.len);
    @memcpy(result.error_msg[0..n], msg[0..n]);
    result.error_len = @intCast(n);
}
