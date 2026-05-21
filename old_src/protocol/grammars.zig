// ============================================================
// src/protocol/grammars.zig
// ============================================================

const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const grammar_compile = @import("../grammar/compile.zig");
const grammar_render = @import("../grammar/render.zig");
const kb_types = @import("../kb/types.zig");
const KBStore = kb_store_mod.KBStore;
const VlpFact = kb_types.VlpFact;

pub const GRAMMAR_SLOT_HTTP_STATUS: i32 = 0;
pub const GRAMMAR_SLOT_HTTP_HEADER: i32 = 1;
pub const GRAMMAR_SLOT_HTTP_CONTENT_TYPE: i32 = 2;
pub const GRAMMAR_SLOT_JSON_BODY: i32 = 3;
pub const GRAMMAR_SLOT_JSON_ERROR: i32 = 4;
pub const GRAMMAR_SLOT_WS_CLOSE: i32 = 5;
pub const GRAMMAR_SLOT_HEALTH: i32 = 6;
pub const GRAMMAR_SLOT_SMTP_GREETING: i32 = 7;
pub const GRAMMAR_SLOT_SMTP_OK: i32 = 8;
pub const GRAMMAR_SLOT_MQTT_CONNACK: i32 = 9;

pub fn initProtocolGrammars(store: *KBStore, grammar_kb_id: i32) VlpStatus {
    const templates = [_]struct { slot: i32, template: []const u8 }{
        .{ .slot = GRAMMAR_SLOT_HTTP_STATUS, .template = "HTTP/1.1 {code:integer} {reason:text}\r\n" },
        .{ .slot = GRAMMAR_SLOT_HTTP_HEADER, .template = "{name:text}: {value:text}\r\n" },
        .{ .slot = GRAMMAR_SLOT_HTTP_CONTENT_TYPE, .template = "Content-Type: {mime:text}\r\n" },
        .{ .slot = GRAMMAR_SLOT_JSON_BODY, .template = "{{\"result\":{data:text},\"confidence\":{confidence:vdr_value}}}" },
        .{ .slot = GRAMMAR_SLOT_JSON_ERROR, .template = "{{\"error\":\"{message:text}\",\"code\":{code:integer}}}" },
        .{ .slot = GRAMMAR_SLOT_WS_CLOSE, .template = "{{\"code\":{code:integer},\"reason\":\"{reason:text}\"}}" },
        .{ .slot = GRAMMAR_SLOT_HEALTH, .template = "{{\"active_connections\":{conns:integer},\"total_requests\":{reqs:integer},\"l3_auto_percent\":\"{l3_num:integer}/{l3_den:integer}\",\"sessions\":{sessions:integer},\"rules\":{rules:integer},\"facts\":{facts:integer}}}" },
        .{ .slot = GRAMMAR_SLOT_SMTP_GREETING, .template = "220 {hostname:text} ESMTP VDR-LLM-Prolog\r\n" },
        .{ .slot = GRAMMAR_SLOT_SMTP_OK, .template = "250 {message:text}\r\n" },
        .{ .slot = GRAMMAR_SLOT_MQTT_CONNACK, .template = "CONNACK:{session_present:integer}:{return_code:integer}" },
    };

    for (templates) |t| {
        const ref = store.text.append(t.template);
        const fact = VlpFact{
            .tag = .text,
            .value = .{ .v = ref.offset, .r0 = @intCast(ref.length) },
            .provenance = .{
                .source_type = .vdr_computation,
                .source_kb_id = grammar_kb_id,
                .source_slot_id = t.slot,
                .confidence = .{ .v = Q16.D, .r0 = 0 },
                .timestamp = 0,
                .derivation_rule_id = -1,
            },
        };
        _ = fact_mod.factAssert(store, grammar_kb_id, t.slot, &fact);
    }

    return .ok;
}

pub fn renderHttpResponse(
    store: *KBStore,
    grammar_kb_id: i32,
    status_code: i32,
    reason: []const u8,
    content_type: []const u8,
    body: []const u8,
    output: []u8,
) i32 {
    var pos: usize = 0;

    pos += copyStr(output[pos..], "HTTP/1.1 ");
    pos += i32ToAscii(status_code, output[pos..]);
    pos += copyStr(output[pos..], " ");
    pos += copyStr(output[pos..], reason);
    pos += copyStr(output[pos..], "\r\n");
    pos += copyStr(output[pos..], "Content-Type: ");
    pos += copyStr(output[pos..], content_type);
    pos += copyStr(output[pos..], "\r\n");
    pos += copyStr(output[pos..], "Content-Length: ");
    pos += i32ToAscii(@intCast(body.len), output[pos..]);
    pos += copyStr(output[pos..], "\r\n\r\n");
    pos += copyStr(output[pos..], body);

    _ = store;
    _ = grammar_kb_id;

    return @intCast(pos);
}

pub fn renderJsonResult(data: []const u8, confidence_v: i32, output: []u8) i32 {
    var pos: usize = 0;
    pos += copyStr(output[pos..], "{\"result\":");
    pos += copyStr(output[pos..], data);
    pos += copyStr(output[pos..], ",\"confidence\":");
    pos += i32ToAscii(confidence_v, output[pos..]);
    pos += copyStr(output[pos..], "/");
    pos += i32ToAscii(Q16.D, output[pos..]);
    pos += copyStr(output[pos..], "}");
    return @intCast(pos);
}

pub fn renderJsonError(message: []const u8, code: i32, output: []u8) i32 {
    var pos: usize = 0;
    pos += copyStr(output[pos..], "{\"error\":\"");
    pos += copyStr(output[pos..], message);
    pos += copyStr(output[pos..], "\",\"code\":");
    pos += i32ToAscii(code, output[pos..]);
    pos += copyStr(output[pos..], "}");
    return @intCast(pos);
}
