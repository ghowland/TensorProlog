// ============================================================
// src/engine/context.zig
// ============================================================

const session_types = @import("../session/types.zig");
const store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const kb_types = @import("../kb/types.zig");

const VlpSession = session_types.VlpSession;
const VlpStatus = session_types.VlpStatus;
const KBStore = store_mod.KBStore;

pub const MAX_CONTEXT_TOKENS: i32 = 4096;

pub const VlpContext = struct {
    token_ids: [4096]i32 = .{0} ** 4096,
    n_tokens: i32 = 0,
};

pub const VlpInput = struct {
    token_ids: []const i32,
    n_tokens: i32,
    raw_bytes: ?[]const u8 = null,
};

pub fn build(
    session: *const VlpSession,
    input: *const VlpInput,
    kb_store: *const KBStore,
    system_prompt: []const i32,
    scratchpad_tokens: []const i32,
    ctx: *VlpContext,
) void {
    ctx.n_tokens = 0;

    appendTokens(ctx, system_prompt);

    var scope_buf: [16]i32 = undefined;
    const scope_len = encodeScopeRef(kb_store, session.kb_root_id, &scope_buf);
    appendTokens(ctx, scope_buf[0..@intCast(scope_len)]);

    appendTokens(ctx, scratchpad_tokens);

    if (input.n_tokens > 0) {
        appendTokens(ctx, input.token_ids[0..@intCast(input.n_tokens)]);
    }
}

fn appendTokens(ctx: *VlpContext, tokens: []const i32) void {
    for (tokens) |t| {
        if (ctx.n_tokens >= MAX_CONTEXT_TOKENS) return;
        ctx.token_ids[@intCast(ctx.n_tokens)] = t;
        ctx.n_tokens += 1;
    }
}

fn encodeScopeRef(kb_store: *const KBStore, kb_id: i32, buf: []i32) i32 {
    _ = kb_store;
    if (kb_id < 0) return 0;
    buf[0] = -100;
    buf[1] = kb_id;
    buf[2] = -101;
    return 3;
}
