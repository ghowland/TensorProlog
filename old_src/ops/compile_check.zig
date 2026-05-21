// ============================================================
// src/ops/compile_check.zig
// ============================================================

pub fn compileCheck(source: []const u8) struct { valid: bool, error_msg: [256]u8, error_len: i32 } {
    var result: struct { valid: bool, error_msg: [256]u8, error_len: i32 } = .{
        .valid = true,
        .error_msg = undefined,
        .error_len = 0,
    };

    var brace_depth: i32 = 0;
    var paren_depth: i32 = 0;
    var bracket_depth: i32 = 0;

    for (source) |c| {
        switch (c) {
            '{' => brace_depth += 1,
            '}' => {
                brace_depth -= 1;
                if (brace_depth < 0) {
                    result.valid = false;
                    const msg = "unmatched }";
                    @memcpy(result.error_msg[0..msg.len], msg);
                    result.error_len = @intCast(msg.len);
                    return result;
                }
            },
            '(' => paren_depth += 1,
            ')' => {
                paren_depth -= 1;
                if (paren_depth < 0) {
                    result.valid = false;
                    const msg = "unmatched )";
                    @memcpy(result.error_msg[0..msg.len], msg);
                    result.error_len = @intCast(msg.len);
                    return result;
                }
            },
            '[' => bracket_depth += 1,
            ']' => {
                bracket_depth -= 1;
                if (bracket_depth < 0) {
                    result.valid = false;
                    const msg = "unmatched ]";
                    @memcpy(result.error_msg[0..msg.len], msg);
                    result.error_len = @intCast(msg.len);
                    return result;
                }
            },
            else => {},
        }
    }

    if (brace_depth != 0 or paren_depth != 0 or bracket_depth != 0) {
        result.valid = false;
        const msg = "unclosed delimiter";
        @memcpy(result.error_msg[0..msg.len], msg);
        result.error_len = @intCast(msg.len);
    }

    return result;
}
