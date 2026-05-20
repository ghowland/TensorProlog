// ============================================================
// src/seed/format_grammars.zig
// ============================================================

const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const kb_types = @import("../kb/types.zig");
const q16_mod = @import("../vdr/q16.zig");
const types_mod = @import("../vdr/types.zig");

const KBStore = kb_store_mod.KBStore;
const VlpFact = kb_types.VlpFact;
const Q16F = q16_mod.Q16;

fn seedProv(kb_id: i32, slot_id: i32) kb_types.VlpProvenance {
    return .{
        .source_type = .vdr_computation,
        .source_kb_id = kb_id,
        .source_slot_id = slot_id,
        .confidence = .{ .v = Q16F.D, .r0 = 0 },
        .timestamp = 0,
        .derivation_rule_id = -1,
    };
}

fn storeTemplate(store: *KBStore, kb_id: i32, slot: i32, template: []const u8) void {
    const ref = store.text.append(template);
    const fact = VlpFact{
        .tag = .text,
        .value = .{ .v = ref.offset, .r0 = @intCast(ref.length) },
        .provenance = seedProv(kb_id, slot),
    };
    _ = fact_mod.factAssert(store, kb_id, slot, &fact);
}

pub fn loadFormatGrammars(store: *KBStore, formats_kb: i32) void {
    var slot: i32 = 0;

    storeTemplate(store, formats_kb, slot, "{{\"type\":\"{type:text}\",\"data\":{data:text}}}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{{\"result\":{result:text},\"confidence\":{confidence:vdr_value}}}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{{\"error\":\"{message:text}\",\"code\":{code:integer}}}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{col1:text},{col2:text},{col3:text}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "| {col1:text} | {col2:text} | {col3:text} |");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "|---|---|---|");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "| {val1:text} | {val2:text} | {val3:text} |");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{key:text}: {value:text}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "- {item:text}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{number:integer}. {item:text}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{name:text} = {value:vdr_value} ({unit:text})");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{{\"active_connections\":{conns:integer},\"total_requests\":{reqs:integer},\"l3_auto_percent\":\"{l3_num:integer}/{l3_den:integer}\",\"active_sessions\":{sessions:integer},\"rules\":{rules:integer},\"facts\":{facts:integer}}}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "HTTP/1.1 {code:integer} {reason:text}\r\n");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{name:text}: {value:text}\r\n");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{{\"code\":{code:integer},\"reason\":\"{reason:text}\"}}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "220 {hostname:text} ESMTP VDR-LLM-Prolog\r\n");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "250 {message:text}\r\n");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "CONNACK:{session_present:integer}:{return_code:integer}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{{\"service\":\"{service:text}\",\"metric\":\"{metric:text}\",\"value\":{value:vdr_value},\"timestamp\":{timestamp:integer}}}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{{\"incident_id\":\"{id:text}\",\"severity\":\"{severity:text}\",\"service\":\"{service:text}\",\"summary\":\"{summary:text}\",\"confidence\":{confidence:vdr_value}}}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{{\"rule_id\":{rule_id:integer},\"pattern\":\"{pattern:text}\",\"fires\":{fires:integer},\"success_rate\":\"{success_num:integer}/{success_den:integer}\"}}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{{\"runner_id\":{id:integer},\"type\":\"{type:text}\",\"state\":\"{state:text}\",\"iterations\":{iterations:integer},\"errors\":{errors:integer}}}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{{\"snapshot_id\":{id:integer},\"session_id\":{session:integer},\"size_bytes\":{size:integer},\"checksum\":{checksum:integer}}}");
    slot += 1;

    storeTemplate(store, formats_kb, slot, "{timestamp:integer} [{level:text}] {source:text}: {message:text}");
    slot += 1;
}
