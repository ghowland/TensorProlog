// ============================================================
// vlp_inference.zig
// Inference loop — top-level host orchestration.
// Drives the LLM generation cycle, command dispatch, output assembly.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const session_mod = @import("vlp_session.zig");
const llm_mod = @import("vlp_llm.zig");
const command_mod = @import("vlp_command.zig");
const kb_mod = @import("vlp_kb_store.zig");
const grammar_mod = @import("vlp_grammar.zig");

// ============================================================
// Output buffer
// ============================================================

pub const OutputBuffer = struct {
    data: []u8,
    length: i32,
    capacity: i32,

    pub fn init(allocator: std.mem.Allocator, capacity: i32) OutputBuffer {
        const data = allocator.alloc(u8, @intCast(capacity)) catch &.{};
        return .{ .data = data, .length = 0, .capacity = capacity };
    }

    pub fn deinit(self: *OutputBuffer, allocator: std.mem.Allocator) void {
        if (self.data.len > 0) allocator.free(self.data);
    }

    pub fn append(self: *OutputBuffer, bytes: []const u8) void {
        const avail = @as(usize, @intCast(self.capacity - self.length));
        const n = @min(bytes.len, avail);
        if (n > 0) {
            @memcpy(self.data[self.length .. self.length + n], bytes[0..n]);
            self.length += @intCast(n);
        }
    }

    pub fn appendByte(self: *OutputBuffer, b: u8) void {
        if (self.length < self.capacity) {
            self.data[@intCast(self.length)] = b;
            self.length += 1;
        }
    }

    pub fn reset(self: *OutputBuffer) void {
        self.length = 0;
    }

    pub fn contents(self: *OutputBuffer) []const u8 {
        return self.data[0..@intCast(self.length)];
    }
};

// ============================================================
// Scratchpad — command results visible to LLM on next step
// ============================================================

pub const ScratchpadEntry = struct {
    command_index: i32,
    result: command_mod.CommandResult,
};

pub const Scratchpad = struct {
    entries: []ScratchpadEntry,
    count: i32,
    capacity: i32,

    pub fn init(allocator: std.mem.Allocator, capacity: i32) Scratchpad {
        const entries = allocator.alloc(ScratchpadEntry, @intCast(capacity)) catch &.{};
        return .{ .entries = entries, .count = 0, .capacity = capacity };
    }

    pub fn deinit(self: *Scratchpad, allocator: std.mem.Allocator) void {
        if (self.entries.len > 0) allocator.free(self.entries);
    }

    pub fn write(self: *Scratchpad, cmd_idx: i32, result: *const command_mod.CommandResult) void {
        if (self.count >= self.capacity) return;
        self.entries[@intCast(self.count)] = .{ .command_index = cmd_idx, .result = result.* };
        self.count += 1;
    }

    pub fn clear(self: *Scratchpad) void {
        self.count = 0;
    }
};

// ============================================================
// Token classification
// ============================================================

pub const TokenClass = enum(i32) {
    prose = 0,
    command_start = 1,
    direct_output = 2,
    end_of_turn = 3,
};

// Token ID conventions — these would come from the tokenizer config
const COMMAND_START_TOKEN: i32 = 32000; // special token marking command start
const DIRECT_OUTPUT_TOKEN: i32 = 32001; // special token marking kb:// reference
const END_OF_TURN_TOKEN: i32 = 2; // standard EOS

// ============================================================
// Context configuration
// ============================================================

pub const ContextConfig = struct {
    system_prompt_kb_id: i32,
    scope_kb_id: i32,
    max_scratchpad_tokens: i32 = 50,
    max_context_tokens: i32 = 4096,
};

// ============================================================
// Inference Engine
// ============================================================

pub const InferenceEngine = struct {
    session_mgr: *session_mod.SessionManager,
    llm: *llm_mod.LlmEngine,
    commands: *command_mod.CommandProcessor,
    kb_store: *kb_mod.KbStore,
    allocator: std.mem.Allocator,
    context_config: ContextConfig,

    // Per-cycle state
    scratchpad: Scratchpad,
    context_buf: []i32,
    context_capacity: i32,

    // Command vocab for constrained generation
    command_vocab: []i32,
    command_vocab_size: i32,

    // Token output buffer for generation
    token_buf: []i32,
    token_capacity: i32,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(
    session_mgr: *session_mod.SessionManager,
    llm: *llm_mod.LlmEngine,
    commands: *command_mod.CommandProcessor,
    kb_store: *kb_mod.KbStore,
    allocator: std.mem.Allocator,
    context_config: *const ContextConfig,
) InferenceEngine {
    const ctx_cap = context_config.max_context_tokens;
    const ctx_buf = allocator.alloc(i32, @intCast(ctx_cap)) catch &.{};
    const cmd_vocab = allocator.alloc(i32, 300) catch &.{}; // ~300 command tokens
    const tok_buf = allocator.alloc(i32, 512) catch &.{};

    return .{
        .session_mgr = session_mgr,
        .llm = llm,
        .commands = commands,
        .kb_store = kb_store,
        .allocator = allocator,
        .context_config = context_config.*,
        .scratchpad = Scratchpad.init(allocator, 32),
        .context_buf = ctx_buf,
        .context_capacity = ctx_cap,
        .command_vocab = cmd_vocab,
        .command_vocab_size = 0,
        .token_buf = tok_buf,
        .token_capacity = @intCast(tok_buf.len),
    };
}

pub fn deinit(self: *InferenceEngine) void {
    self.scratchpad.deinit(self.allocator);
    if (self.context_buf.len > 0) self.allocator.free(self.context_buf);
    if (self.command_vocab.len > 0) self.allocator.free(self.command_vocab);
    if (self.token_buf.len > 0) self.allocator.free(self.token_buf);
}

// ============================================================
// Full inference cycle — Section 7 of the spec
// ============================================================

pub fn cycle(self: *InferenceEngine, handle: types.SessionHandle, input: []const u8, output: *OutputBuffer) types.Status {
    output.reset();
    self.scratchpad.clear();

    const session = self.session_mgr.get(handle) orelse
        return types.Status.err(.session, .session_limit, handle.id);

    // Phase 1: Tokenize input
    const input_tokens = tokenize(input, self.context_buf);

    // Phase 2: Build context
    const context_len = self.buildContext(handle, input_tokens);
    if (context_len <= 0) return types.Status.err(.system, .init_failed, 0);

    // Feed context to LLM
    const fwd = self.llm.forward(self.context_buf[0..@intCast(context_len)]);
    if (fwd.status.isErr()) return fwd.status;

    // Phase 3: Generation loop
    var tokens_generated: i32 = 0;
    var commands_generated: i32 = 0;
    const max_tokens: i32 = 2048;
    const sampling = llm_mod.SamplingConfig{};

    while (tokens_generated < max_tokens) {
        const token = self.llm.generateToken(&sampling);
        if (token < 0) break;
        tokens_generated += 1;

        const class = classifyToken(token);

        switch (class) {
            .command_start => {
                // Generate constrained command tokens
                const cmd_len = self.llm.generateCommandTokens(
                    self.command_vocab[0..@intCast(self.command_vocab_size)],
                    32,
                    self.token_buf,
                );
                tokens_generated += cmd_len;
                commands_generated += 1;

                // Parse and execute
                if (self.commands.parse(self.token_buf[0..@intCast(cmd_len)])) |cmd| {
                    const result = self.commands.execute(handle, &cmd);
                    self.scratchpad.write(commands_generated, &result);

                    // If command produced output text, append to output
                    if (result.output_text) |text| {
                        output.append(text);
                    }
                }
            },
            .direct_output => {
                // Next tokens form a kb:// URL
                const url_len = self.llm.generateProse(&sampling, 32, self.token_buf);
                const url_text = tokensToText(self.token_buf[0..@intCast(url_len)]);

                const kb_url = self.commands.parseKbUrl(url_text);
                if (kb_url.kb_id >= 0) {
                    // Render data from KB through grammar
                    var render_cmd = std.mem.zeroes(types.Command);
                    render_cmd.type = .direct_output;
                    render_cmd.target_kb_id = kb_url.kb_id;
                    render_cmd.target_slot_id = kb_url.slot_id;

                    const result = self.commands.execute(handle, &render_cmd);
                    if (result.output_text) |text| {
                        output.append(text);
                    }
                }
                tokens_generated += url_len;
            },
            .end_of_turn => break,
            .prose => {
                // LLM judgment and framing — pass through to output
                const text = tokenToText(token);
                output.append(text);
            },
        }
    }

    // Phase 5: Post-processing
    _ = self.session_mgr.incrementTurn(handle, tokens_generated, commands_generated * 8);

    // Determine execution level for stats
    const level: i8 = if (commands_generated == 0 and tokens_generated > 50) 1 // L1: full judgment
        else if (commands_generated > 0 and tokens_generated < 30) 2 // L2: rule invocation
        else 1;
    _ = self.session_mgr.updateLevelStats(handle, level, tokens_generated);

    // Phase 6: Auto-snapshot
    if (self.session_mgr.shouldAutoSnapshot(handle)) {
        // Trigger snapshot — actual implementation in session_mgr + snapshot_mgr
        _ = session;
    }

    return types.Status.ok();
}

// ============================================================
// Execution levels — direct entry points
// ============================================================

pub fn executeL1(self: *InferenceEngine, handle: types.SessionHandle, input: []const u8, output: *OutputBuffer) types.Status {
    // Full LLM judgment. 50-500 tokens.
    return self.cycle(handle, input, output);
}

pub fn executeL2(self: *InferenceEngine, handle: types.SessionHandle, pattern: *const types.Term) types.Status {
    // LLM invokes stored rule. ~8 command tokens + ~10 prose tokens.
    const result = self.commands.prolog.query(0, pattern);
    _ = result;
    _ = self.session_mgr.updateLevelStats(handle, 2, 18);
    return types.Status.ok();
}

pub fn executeL3(self: *InferenceEngine, handle: types.SessionHandle, kb_id: i32) types.Status {
    // Automatic rule firing. 0 LLM tokens.
    const fired = self.commands.prolog.fireAndCommit(kb_id);
    _ = fired;
    _ = self.session_mgr.updateLevelStats(handle, 3, 0);
    return types.Status.ok();
}

// ============================================================
// Context building
// ============================================================

fn buildContext(self: *InferenceEngine, handle: types.SessionHandle, input_tokens: []const i32) i32 {
    var pos: i32 = 0;
    const cap = self.context_capacity;

    // System prompt from seed KB
    const sys_prompt_len = loadSystemPrompt(self, self.context_buf[@intCast(pos)..@intCast(cap)]);
    pos += sys_prompt_len;

    // Scope reference (~5 tokens)
    const session = self.session_mgr.get(handle);
    if (session) |s| {
        if (pos + 5 < cap) {
            self.context_buf[@intCast(pos)] = s.kb_root_id;
            pos += 1;
        }
    }

    // Input tokens
    for (input_tokens) |t| {
        if (pos >= cap) break;
        self.context_buf[@intCast(pos)] = t;
        pos += 1;
    }

    // Scratchpad tokens (previous command results this turn)
    for (self.scratchpad.entries[0..@intCast(self.scratchpad.count)]) |entry| {
        if (pos + 3 >= cap) break;
        self.context_buf[@intCast(pos)] = entry.result.output_kb_id;
        pos += 1;
        self.context_buf[@intCast(pos)] = entry.result.output_slot_id;
        pos += 1;
        self.context_buf[@intCast(pos)] = entry.result.output_bytes;
        pos += 1;
    }

    return pos;
}

// ============================================================
// Helpers
// ============================================================

fn classifyToken(token: i32) TokenClass {
    if (token == COMMAND_START_TOKEN) return .command_start;
    if (token == DIRECT_OUTPUT_TOKEN) return .direct_output;
    if (token == END_OF_TURN_TOKEN) return .end_of_turn;
    return .prose;
}

fn tokenize(input: []const u8, output: []i32) []const i32 {
    // Simplified tokenizer — real implementation uses compiled tokenizer
    // For now: one token per byte
    const n = @min(input.len, output.len);
    for (input[0..n], 0..) |b, i| {
        output[i] = @as(i32, b);
    }
    return output[0..n];
}

fn tokenToText(token: i32) []const u8 {
    // Simplified: return single byte for token
    // Real implementation: look up in tokenizer vocabulary
    _ = token;
    return " ";
}

fn tokensToText(tokens: []const i32) []const u8 {
    // Simplified
    _ = tokens;
    return "";
}

fn loadSystemPrompt(self: *InferenceEngine, buf: []i32) i32 {
    // Load system prompt tokens from seed KB
    _ = self;
    _ = buf;
    return 0; // stub — real implementation reads from system.prompt KB
}
