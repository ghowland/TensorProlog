// ============================================================
// src/engine/token_classify.zig
// ============================================================

const vdr_types = @import("../vdr/types.zig");
const VlpTokenClass = vdr_types.VlpTokenClass;

pub const TokenClassConfig = struct {
    command_range_start: i32 = -1000,
    command_range_end: i32 = -900,
    direct_output_token: i32 = -800,
    eos_token: i32 = 0,
};

pub fn classify(token_id: i32, cfg: *const TokenClassConfig) VlpTokenClass {
    if (token_id == cfg.eos_token) return .end_of_turn;
    if (token_id == cfg.direct_output_token) return .direct_output;
    if (token_id >= cfg.command_range_start and token_id <= cfg.command_range_end) return .command_start;
    return .prose;
}

pub fn isCommand(token_id: i32, cfg: *const TokenClassConfig) bool {
    return classify(token_id, cfg) == .command_start;
}

pub fn isEndOfTurn(token_id: i32, cfg: *const TokenClassConfig) bool {
    return classify(token_id, cfg) == .end_of_turn;
}
