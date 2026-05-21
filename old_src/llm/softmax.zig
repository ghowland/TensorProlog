// ============================================================
// src/llm/softmax.zig
// ============================================================

const q16_mod = @import("../vdr/q16.zig");
const Q16 = q16_mod.Q16;

pub fn softmaxSurrogate(logits: []const Q16, probs: []Q16) void {
    Q16.softmax(logits, probs);
}

pub fn verifySoftmaxSum(probs: []const Q16) bool {
    var sum: i32 = 0;
    for (probs) |p| sum += p.v;
    return sum == Q16.D;
}
