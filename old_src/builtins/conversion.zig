// ============================================================
// src/builtins/conversion.zig
// ============================================================

const kb_store_mod = @import("../kb/store.zig");
const fact_mod = @import("../kb/fact.zig");
const kb_types = @import("../kb/types.zig");
const KBStore = kb_store_mod.KBStore;
const VlpProvenance = kb_types.VlpProvenance;
const VlpStatus = kb_types.VlpStatus;
const VlpFact = kb_types.VlpFact;

const q16_set = @import("../vdr/q16.zig");
const Q16 = q16_set.Q16;

const dispatch_mod = @import("dispatch.zig");
const BuiltinArgs = dispatch_mod.BuiltinArgs;
const BuiltinResult = dispatch_mod.BuiltinResult;

fn computeProvenance(kb_id: i32, slot_id: i32) VlpProvenance {
    return .{
        .source_type = .vdr_computation,
        .source_kb_id = kb_id,
        .source_slot_id = slot_id,
        .confidence = .{ .v = Q16.D, .r0 = 0 },
        .timestamp = 0,
        .derivation_rule_id = -1,
    };
}

fn skipWs(input: []const u8, start: usize) usize {
    var pos = start;
    while (pos < input.len and (input[pos] == ' ' or input[pos] == '\t' or input[pos] == '\n' or input[pos] == '\r')) {
        pos += 1;
    }
    return pos;
}

pub fn parseJson(input: []const u8, store: *KBStore, target_kb_id: i32) VlpStatus {
    var pos: usize = 0;
    var slot: i32 = 0;
    pos = skipWs(input, pos);
    if (pos >= input.len) return .err_command_parse;
    if (input[pos] == '{') {
        return parseJsonObject(input, &pos, store, target_kb_id, &slot);
    } else if (input[pos] == '[') {
        return parseJsonArray(input, &pos, store, target_kb_id, &slot);
    }
    return .err_command_parse;
}

fn parseJsonObject(input: []const u8, pos: *usize, store: *KBStore, kb_id: i32, slot: *i32) VlpStatus {
    pos.* += 1;
    pos.* = skipWs(input, pos.*);
    while (pos.* < input.len and input[pos.*] != '}') {
        pos.* = skipWs(input, pos.*);
        if (pos.* >= input.len or input[pos.*] != '"') return .err_command_parse;
        pos.* += 1;
        while (pos.* < input.len and input[pos.*] != '"') {
            if (input[pos.*] == '\\' and pos.* + 1 < input.len) {
                pos.* += 2;
            } else {
                pos.* += 1;
            }
        }
        if (pos.* >= input.len) return .err_command_parse;
        pos.* += 1;
        pos.* = skipWs(input, pos.*);
        if (pos.* >= input.len or input[pos.*] != ':') return .err_command_parse;
        pos.* += 1;
        pos.* = skipWs(input, pos.*);
        const status = parseJsonValue(input, pos, store, kb_id, slot);
        if (status != .ok) return status;
        pos.* = skipWs(input, pos.*);
        if (pos.* < input.len and input[pos.*] == ',') pos.* += 1;
    }
    if (pos.* < input.len and input[pos.*] == '}') pos.* += 1;
    return .ok;
}

fn parseJsonArray(input: []const u8, pos: *usize, store: *KBStore, kb_id: i32, slot: *i32) VlpStatus {
    pos.* += 1;
    pos.* = skipWs(input, pos.*);
    while (pos.* < input.len and input[pos.*] != ']') {
        const status = parseJsonValue(input, pos, store, kb_id, slot);
        if (status != .ok) return status;
        pos.* = skipWs(input, pos.*);
        if (pos.* < input.len and input[pos.*] == ',') {
            pos.* += 1;
            pos.* = skipWs(input, pos.*);
        }
    }
    if (pos.* < input.len and input[pos.*] == ']') pos.* += 1;
    return .ok;
}

fn parseJsonValue(input: []const u8, pos: *usize, store: *KBStore, kb_id: i32, slot: *i32) VlpStatus {
    pos.* = skipWs(input, pos.*);
    if (pos.* >= input.len) return .err_command_parse;
    const c = input[pos.*];

    if (c == '"') {
        pos.* += 1;
        const str_start = pos.*;
        while (pos.* < input.len and input[pos.*] != '"') {
            if (input[pos.*] == '\\' and pos.* + 1 < input.len) {
                pos.* += 2;
            } else {
                pos.* += 1;
            }
        }
        const str_end = pos.*;
        if (pos.* < input.len) pos.* += 1;
        const text_ref = store.text.append(input[str_start..str_end]);
        const fact = VlpFact{
            .tag = .text,
            .value = .{ .v = text_ref.offset, .r0 = @intCast(text_ref.length) },
            .provenance = computeProvenance(kb_id, slot.*),
        };
        _ = fact_mod.factAssert(store, kb_id, slot.*, &fact);
        slot.* += 1;
        return .ok;
    }

    if (c == '{') {
        const child_id = store.createKB(.{
            .name = "json_obj",
            .parent_id = kb_id,
            .visibility = .internal,
            .owner = "system",
            .max_facts = 256,
            .max_rules = 0,
            .max_children = 16,
        });
        const ref_fact = VlpFact{
            .tag = .reference,
            .value = .{ .v = child_id, .r0 = 0 },
            .provenance = computeProvenance(kb_id, slot.*),
        };
        _ = fact_mod.factAssert(store, kb_id, slot.*, &ref_fact);
        slot.* += 1;
        var child_slot: i32 = 0;
        return parseJsonObject(input, pos, store, child_id, &child_slot);
    }

    if (c == '[') {
        const child_id = store.createKB(.{
            .name = "json_arr",
            .parent_id = kb_id,
            .visibility = .internal,
            .owner = "system",
            .max_facts = 1024,
            .max_rules = 0,
            .max_children = 0,
        });
        const ref_fact = VlpFact{
            .tag = .reference,
            .value = .{ .v = child_id, .r0 = 0 },
            .provenance = computeProvenance(kb_id, slot.*),
        };
        _ = fact_mod.factAssert(store, kb_id, slot.*, &ref_fact);
        slot.* += 1;
        var child_slot: i32 = 0;
        return parseJsonArray(input, pos, store, child_id, &child_slot);
    }

    if (c == 't') {
        pos.* += 4;
        const fact = VlpFact{
            .tag = .boolean,
            .value = .{ .v = 1, .r0 = 0 },
            .provenance = computeProvenance(kb_id, slot.*),
        };
        _ = fact_mod.factAssert(store, kb_id, slot.*, &fact);
        slot.* += 1;
        return .ok;
    }

    if (c == 'f') {
        pos.* += 5;
        const fact = VlpFact{
            .tag = .boolean,
            .value = .{ .v = 0, .r0 = 0 },
            .provenance = computeProvenance(kb_id, slot.*),
        };
        _ = fact_mod.factAssert(store, kb_id, slot.*, &fact);
        slot.* += 1;
        return .ok;
    }

    if (c == 'n') {
        pos.* += 4;
        const fact = VlpFact{
            .tag = .empty,
            .value = Q16.zero(),
            .provenance = computeProvenance(kb_id, slot.*),
        };
        _ = fact_mod.factAssert(store, kb_id, slot.*, &fact);
        slot.* += 1;
        return .ok;
    }

    if (c == '-' or (c >= '0' and c <= '9')) {
        const q = parseJsonNumber(input, pos);
        const fact = VlpFact{
            .tag = .value,
            .value = q,
            .provenance = computeProvenance(kb_id, slot.*),
        };
        _ = fact_mod.factAssert(store, kb_id, slot.*, &fact);
        slot.* += 1;
        return .ok;
    }

    return .err_command_parse;
}

fn parseJsonNumber(input: []const u8, pos: *usize) Q16 {
    var negative = false;
    if (pos.* < input.len and input[pos.*] == '-') {
        negative = true;
        pos.* += 1;
    }
    var int_part: i64 = 0;
    while (pos.* < input.len and input[pos.*] >= '0' and input[pos.*] <= '9') {
        int_part = int_part * 10 + @as(i64, input[pos.*] - '0');
        pos.* += 1;
    }
    var frac_num: i64 = 0;
    var frac_den: i64 = 1;
    if (pos.* < input.len and input[pos.*] == '.') {
        pos.* += 1;
        while (pos.* < input.len and input[pos.*] >= '0' and input[pos.*] <= '9') {
            frac_num = frac_num * 10 + @as(i64, input[pos.*] - '0');
            frac_den *= 10;
            pos.* += 1;
        }
    }
    if (pos.* < input.len and (input[pos.*] == 'e' or input[pos.*] == 'E')) {
        pos.* += 1;
        if (pos.* < input.len and (input[pos.*] == '+' or input[pos.*] == '-')) pos.* += 1;
        while (pos.* < input.len and input[pos.*] >= '0' and input[pos.*] <= '9') pos.* += 1;
    }
    var value: i64 = int_part * @as(i64, Q16.D);
    if (frac_den > 1) {
        value += @divTrunc(frac_num * @as(i64, Q16.D), frac_den);
    }
    if (negative) value = -value;
    return .{ .v = @intCast(value), .r0 = 0 };
}

pub fn parseCsv(input: []const u8, store: *KBStore, target_kb_id: i32, delimiter: u8) VlpStatus {
    var slot: i32 = 0;
    var pos: usize = 0;
    while (pos < input.len) {
        const line_start = pos;
        while (pos < input.len and input[pos] != '\n' and input[pos] != '\r') pos += 1;
        const line = input[line_start..pos];
        if (pos < input.len and input[pos] == '\r') pos += 1;
        if (pos < input.len and input[pos] == '\n') pos += 1;
        if (line.len == 0) continue;
        var field_start: usize = 0;
        var i: usize = 0;
        while (i <= line.len) {
            const at_delim = (i < line.len and line[i] == delimiter);
            const at_end = (i == line.len);
            if (at_delim or at_end) {
                const field = line[field_start..i];
                if (tryParseNumber(field)) |q| {
                    const fact = VlpFact{
                        .tag = .value,
                        .value = q,
                        .provenance = computeProvenance(target_kb_id, slot),
                    };
                    _ = fact_mod.factAssert(store, target_kb_id, slot, &fact);
                } else {
                    const text_ref = store.text.append(field);
                    const fact = VlpFact{
                        .tag = .text,
                        .value = .{ .v = text_ref.offset, .r0 = @intCast(text_ref.length) },
                        .provenance = computeProvenance(target_kb_id, slot),
                    };
                    _ = fact_mod.factAssert(store, target_kb_id, slot, &fact);
                }
                slot += 1;
                field_start = i + 1;
            }
            i += 1;
        }
    }
    return .ok;
}

fn tryParseNumber(field: []const u8) ?Q16 {
    if (field.len == 0) return null;
    var pos: usize = 0;
    var negative = false;
    if (field[0] == '-') {
        negative = true;
        pos = 1;
    }
    if (pos >= field.len) return null;
    if (field[pos] < '0' or field[pos] > '9') return null;
    var int_part: i64 = 0;
    while (pos < field.len and field[pos] >= '0' and field[pos] <= '9') {
        int_part = int_part * 10 + @as(i64, field[pos] - '0');
        pos += 1;
    }
    var frac_num: i64 = 0;
    var frac_den: i64 = 1;
    if (pos < field.len and field[pos] == '.') {
        pos += 1;
        while (pos < field.len and field[pos] >= '0' and field[pos] <= '9') {
            frac_num = frac_num * 10 + @as(i64, field[pos] - '0');
            frac_den *= 10;
            pos += 1;
        }
    }
    if (pos != field.len) return null;
    var value: i64 = int_part * @as(i64, Q16.D);
    if (frac_den > 1) {
        value += @divTrunc(frac_num * @as(i64, Q16.D), frac_den);
    }
    if (negative) value = -value;
    return .{ .v = @intCast(value), .r0 = 0 };
}

pub fn parseXml(input: []const u8, store: *KBStore, target_kb_id: i32) VlpStatus {
    var slot: i32 = 0;
    var pos: usize = 0;
    while (pos < input.len) {
        if (input[pos] == '<') {
            pos += 1;
            if (pos < input.len and input[pos] == '/') {
                while (pos < input.len and input[pos] != '>') pos += 1;
                if (pos < input.len) pos += 1;
                continue;
            }
            if (pos < input.len and input[pos] == '?') {
                while (pos < input.len and input[pos] != '>') pos += 1;
                if (pos < input.len) pos += 1;
                continue;
            }
            if (pos < input.len and input[pos] == '!') {
                while (pos < input.len and input[pos] != '>') pos += 1;
                if (pos < input.len) pos += 1;
                continue;
            }
            const tag_start = pos;
            while (pos < input.len and input[pos] != '>' and input[pos] != ' ' and input[pos] != '/') pos += 1;
            const tag_name = input[tag_start..pos];
            while (pos < input.len and input[pos] != '>') pos += 1;
            if (pos < input.len) pos += 1;
            const tag_ref = store.text.append(tag_name);
            const tag_fact = VlpFact{
                .tag = .text,
                .value = .{ .v = tag_ref.offset, .r0 = @intCast(tag_ref.length) },
                .provenance = computeProvenance(target_kb_id, slot),
            };
            _ = fact_mod.factAssert(store, target_kb_id, slot, &tag_fact);
            slot += 1;
            const content_start = pos;
            while (pos < input.len and input[pos] != '<') pos += 1;
            if (pos > content_start) {
                const content = input[content_start..pos];
                const trimmed = trimBytes(content);
                if (trimmed.len > 0) {
                    const cref = store.text.append(trimmed);
                    const cfact = VlpFact{
                        .tag = .text,
                        .value = .{ .v = cref.offset, .r0 = @intCast(cref.length) },
                        .provenance = computeProvenance(target_kb_id, slot),
                    };
                    _ = fact_mod.factAssert(store, target_kb_id, slot, &cfact);
                    slot += 1;
                }
            }
        } else {
            pos += 1;
        }
    }
    return .ok;
}

fn trimBytes(input: []const u8) []const u8 {
    var s: usize = 0;
    while (s < input.len and (input[s] == ' ' or input[s] == '\t' or input[s] == '\n' or input[s] == '\r')) s += 1;
    var e = input.len;
    while (e > s and (input[e - 1] == ' ' or input[e - 1] == '\t' or input[e - 1] == '\n' or input[e - 1] == '\r')) e -= 1;
    return input[s..e];
}

pub fn parseYaml(input: []const u8, store: *KBStore, target_kb_id: i32) VlpStatus {
    var slot: i32 = 0;
    var pos: usize = 0;
    while (pos < input.len) {
        while (pos < input.len and (input[pos] == ' ' or input[pos] == '\t')) pos += 1;
        if (pos >= input.len) break;
        if (input[pos] == '#') {
            while (pos < input.len and input[pos] != '\n') pos += 1;
            if (pos < input.len) pos += 1;
            continue;
        }
        if (input[pos] == '\n' or input[pos] == '\r') {
            pos += 1;
            continue;
        }
        if (pos + 2 < input.len and input[pos] == '-' and input[pos + 1] == '-' and input[pos + 2] == '-') {
            while (pos < input.len and input[pos] != '\n') pos += 1;
            if (pos < input.len) pos += 1;
            continue;
        }
        const key_start = pos;
        while (pos < input.len and input[pos] != ':' and input[pos] != '\n') pos += 1;
        if (pos < input.len and input[pos] == ':') {
            const key_end = pos;
            const key_raw = input[key_start..key_end];
            const key = trimBytes(key_raw);
            pos += 1;
            while (pos < input.len and (input[pos] == ' ' or input[pos] == '\t')) pos += 1;
            const val_start = pos;
            while (pos < input.len and input[pos] != '\n' and input[pos] != '\r') pos += 1;
            const val_raw = input[val_start..pos];
            const val = trimBytes(val_raw);
            if (pos < input.len and input[pos] == '\r') pos += 1;
            if (pos < input.len and input[pos] == '\n') pos += 1;
            if (key.len > 0) {
                const kref = store.text.append(key);
                const kfact = VlpFact{
                    .tag = .text,
                    .value = .{ .v = kref.offset, .r0 = @intCast(kref.length) },
                    .provenance = computeProvenance(target_kb_id, slot),
                };
                _ = fact_mod.factAssert(store, target_kb_id, slot, &kfact);
                slot += 1;
                if (val.len > 0) {
                    if (tryParseNumber(val)) |q| {
                        const vfact = VlpFact{
                            .tag = .value,
                            .value = q,
                            .provenance = computeProvenance(target_kb_id, slot),
                        };
                        _ = fact_mod.factAssert(store, target_kb_id, slot, &vfact);
                    } else {
                        const vref = store.text.append(val);
                        const vfact = VlpFact{
                            .tag = .text,
                            .value = .{ .v = vref.offset, .r0 = @intCast(vref.length) },
                            .provenance = computeProvenance(target_kb_id, slot),
                        };
                        _ = fact_mod.factAssert(store, target_kb_id, slot, &vfact);
                    }
                    slot += 1;
                }
            }
        } else {
            while (pos < input.len and input[pos] != '\n') pos += 1;
            if (pos < input.len) pos += 1;
        }
    }
    return .ok;
}

pub fn toJson(store: *KBStore, kb_id: i32, output: []u8) i32 {
    var pos: usize = 0;
    const kb = store.getKB(kb_id) orelse return 0;
    const fc: usize = @intCast(kb.facts_count);
    if (pos < output.len) {
        output[pos] = '{';
        pos += 1;
    }
    var written_first = false;
    for (0..fc) |i| {
        const si: i32 = @intCast(i);
        const fact = fact_mod.factQuery(store, kb_id, si) orelse continue;
        if (fact.tag == .empty) continue;
        if (written_first and pos + 1 < output.len) {
            output[pos] = ',';
            pos += 1;
        }
        written_first = true;
        if (pos + 1 < output.len) {
            output[pos] = '"';
            pos += 1;
        }
        pos += i32ToAscii(si, output[pos..]);
        if (pos + 2 < output.len) {
            output[pos] = '"';
            pos += 1;
            output[pos] = ':';
            pos += 1;
        }
        pos += factToJson(&fact, store, output[pos..]);
    }
    if (pos < output.len) {
        output[pos] = '}';
        pos += 1;
    }
    return @intCast(pos);
}

fn factToJson(fact: *const VlpFact, store: *KBStore, output: []u8) usize {
    var pos: usize = 0;
    switch (fact.tag) {
        .value => {
            pos += vdrToDecimal(fact.value, output[pos..]);
        },
        .text => {
            if (pos < output.len) {
                output[pos] = '"';
                pos += 1;
            }
            const text_offset: usize = @intCast(fact.value.v);
            const text_length: usize = @intCast(fact.value.r0);
            const text_data = store.text.read(text_offset, text_length);
            const copy_len = @min(text_data.len, output.len - pos);
            @memcpy(output[pos .. pos + copy_len], text_data[0..copy_len]);
            pos += copy_len;
            if (pos < output.len) {
                output[pos] = '"';
                pos += 1;
            }
        },
        .boolean => {
            if (fact.value.v != 0) {
                const t = "true";
                const tl = @min(t.len, output.len - pos);
                @memcpy(output[pos .. pos + tl], t[0..tl]);
                pos += tl;
            } else {
                const f = "false";
                const fl = @min(f.len, output.len - pos);
                @memcpy(output[pos .. pos + fl], f[0..fl]);
                pos += fl;
            }
        },
        .empty => {
            const n = "null";
            const nl = @min(n.len, output.len - pos);
            @memcpy(output[pos .. pos + nl], n[0..nl]);
            pos += nl;
        },
        .reference => {
            pos += @intCast(toJson(store, fact.value.v, output[pos..]));
        },
        else => {
            pos += vdrToDecimal(fact.value, output[pos..]);
        },
    }
    return pos;
}

pub fn toCsv(store: *KBStore, kb_id: i32, delimiter: u8, output: []u8) i32 {
    var pos: usize = 0;
    const kb = store.getKB(kb_id) orelse return 0;
    const fc: usize = @intCast(kb.facts_count);
    var written_first = false;
    for (0..fc) |i| {
        const si: i32 = @intCast(i);
        const fact = fact_mod.factQuery(store, kb_id, si) orelse continue;
        if (fact.tag == .empty) continue;
        if (written_first and pos < output.len) {
            output[pos] = delimiter;
            pos += 1;
        }
        written_first = true;
        switch (fact.tag) {
            .value => {
                pos += vdrToDecimal(fact.value, output[pos..]);
            },
            .text => {
                const text_offset: usize = @intCast(fact.value.v);
                const text_length: usize = @intCast(fact.value.r0);
                const text_data = store.text.read(text_offset, text_length);
                const copy_len = @min(text_data.len, output.len - pos);
                @memcpy(output[pos .. pos + copy_len], text_data[0..copy_len]);
                pos += copy_len;
            },
            .boolean => {
                if (fact.value.v != 0) {
                    if (pos + 4 <= output.len) {
                        @memcpy(output[pos .. pos + 4], "true");
                        pos += 4;
                    }
                } else {
                    if (pos + 5 <= output.len) {
                        @memcpy(output[pos .. pos + 5], "false");
                        pos += 5;
                    }
                }
            },
            else => {
                pos += vdrToDecimal(fact.value, output[pos..]);
            },
        }
    }
    return @intCast(pos);
}

fn vdrToDecimal(val: Q16, output: []u8) usize {
    if (output.len == 0) return 0;
    var pos: usize = 0;
    var v: i64 = @intCast(val.v);
    if (v < 0) {
        if (pos < output.len) {
            output[pos] = '-';
            pos += 1;
        }
        v = -v;
    }
    const int_part: i64 = @divTrunc(v, @as(i64, Q16.D));
    const frac_part: i64 = @mod(v, @as(i64, Q16.D));
    pos += i64ToAscii(int_part, output[pos..]);
    if (frac_part > 0 and pos < output.len) {
        output[pos] = '.';
        pos += 1;
        var remainder = frac_part;
        var digits: usize = 0;
        while (remainder > 0 and digits < 6 and pos < output.len) {
            remainder *= 10;
            const digit: i64 = @divTrunc(remainder, @as(i64, Q16.D));
            output[pos] = @intCast(@as(u8, @intCast(digit)) + '0');
            pos += 1;
            remainder = @mod(remainder, @as(i64, Q16.D));
            digits += 1;
        }
    }
    return pos;
}

fn i32ToAscii(val: i32, output: []u8) usize {
    return i64ToAscii(@intCast(val), output);
}

fn i64ToAscii(val: i64, output: []u8) usize {
    if (output.len == 0) return 0;
    if (val == 0) {
        output[0] = '0';
        return 1;
    }
    var v = val;
    var pos: usize = 0;
    if (v < 0) {
        output[pos] = '-';
        pos += 1;
        v = -v;
    }
    var buf: [20]u8 = undefined;
    var len: usize = 0;
    while (v > 0) {
        buf[len] = @intCast(@as(u8, @intCast(@mod(v, 10))) + '0');
        len += 1;
        v = @divTrunc(v, 10);
    }
    var i: usize = 0;
    while (i < len and pos < output.len) {
        output[pos] = buf[len - 1 - i];
        pos += 1;
        i += 1;
    }
    return pos;
}

pub fn toFraction(numerator: i64, denominator: i64) Q16 {
    if (denominator == 0) return Q16.zero();
    const scaled: i64 = @divTrunc(numerator * @as(i64, Q16.D), denominator);
    const rem: i64 = numerator * @as(i64, Q16.D) - scaled * denominator;
    return .{
        .v = @intCast(scaled),
        .r0 = @intCast(@mod(rem, @as(i64, 32768))),
    };
}

pub fn fromFraction(val: Q16) struct { numerator: i64, denominator: i64 } {
    return .{ .numerator = @intCast(val.v), .denominator = @intCast(Q16.D) };
}

pub fn vdrToDecimalString(val: Q16, precision: i32, output: []u8) i32 {
    if (output.len == 0) return 0;
    var pos: usize = 0;
    var v: i64 = @intCast(val.v);
    if (v < 0) {
        if (pos < output.len) {
            output[pos] = '-';
            pos += 1;
        }
        v = -v;
    }
    const int_part: i64 = @divTrunc(v, @as(i64, Q16.D));
    pos += i64ToAscii(int_part, output[pos..]);
    const p: usize = @intCast(@max(precision, 0));
    if (p > 0 and pos < output.len) {
        output[pos] = '.';
        pos += 1;
        var remainder: i64 = @mod(v, @as(i64, Q16.D));
        for (0..p) |_| {
            if (pos >= output.len) break;
            remainder *= 10;
            const digit: i64 = @divTrunc(remainder, @as(i64, Q16.D));
            output[pos] = @intCast(@as(u8, @intCast(digit)) + '0');
            pos += 1;
            remainder = @mod(remainder, @as(i64, Q16.D));
        }
    }
    return @intCast(pos);
}

pub fn decimalStringToVdr(input: []const u8) ?Q16 {
    return tryParseNumber(input);
}

pub fn baseConvert(value: i64, to_base: i32, output: []u8) i32 {
    if (output.len == 0 or to_base < 2 or to_base > 36) return 0;
    if (value == 0) {
        output[0] = '0';
        return 1;
    }
    const digits = "0123456789abcdefghijklmnopqrstuvwxyz";
    var v = value;
    var pos: usize = 0;
    var negative = false;
    if (v < 0) {
        negative = true;
        v = -v;
    }
    var buf: [64]u8 = undefined;
    var len: usize = 0;
    const b: i64 = @intCast(to_base);
    while (v > 0 and len < buf.len) {
        const idx: usize = @intCast(@mod(v, b));
        buf[len] = digits[idx];
        len += 1;
        v = @divTrunc(v, b);
    }
    if (negative and pos < output.len) {
        output[pos] = '-';
        pos += 1;
    }
    var i: usize = 0;
    while (i < len and pos < output.len) {
        output[pos] = buf[len - 1 - i];
        pos += 1;
        i += 1;
    }
    return @intCast(pos);
}

pub fn timestampToFields(timestamp: i32) struct { year: i32, month: i32, day: i32, hour: i32, minute: i32, second: i32 } {
    const t: i64 = @intCast(timestamp);
    const secs_per_day: i64 = 86400;
    const days_since_epoch = @divTrunc(t, secs_per_day);
    const time_of_day = @mod(t, secs_per_day);
    const hour: i32 = @intCast(@divTrunc(time_of_day, 3600));
    const minute: i32 = @intCast(@divTrunc(@mod(time_of_day, 3600), 60));
    const second: i32 = @intCast(@mod(time_of_day, 60));
    var days = days_since_epoch;
    var year: i32 = 1970;
    while (true) {
        const days_in_year: i64 = if (isLeapYear(year)) 366 else 365;
        if (days < days_in_year) break;
        days -= days_in_year;
        year += 1;
    }
    const month_days_normal = [_]i32{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    const month_days_leap = [_]i32{ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    const mdays: []const i32 = if (isLeapYear(year)) &month_days_leap else &month_days_normal;
    var month: i32 = 1;
    for (mdays) |md| {
        if (days < @as(i64, md)) break;
        days -= @as(i64, md);
        month += 1;
    }
    const day: i32 = @intCast(days + 1);
    return .{ .year = year, .month = month, .day = day, .hour = hour, .minute = minute, .second = second };
}

fn isLeapYear(year: i32) bool {
    if (@mod(year, 400) == 0) return true;
    if (@mod(year, 100) == 0) return false;
    if (@mod(year, 4) == 0) return true;
    return false;
}

pub fn builtinParseJson(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const status = parseJson(args.text_in[0], args.store, args.target_kb_id);
    if (status != .ok) return dispatch_mod.errorResult(status);
    return dispatch_mod.emptyResult();
}

pub fn builtinParseCsv(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const delim: u8 = if (args.int_arg_count > 0) @intCast(args.int_args[0]) else ',';
    const status = parseCsv(args.text_in[0], args.store, args.target_kb_id, delim);
    if (status != .ok) return dispatch_mod.errorResult(status);
    return dispatch_mod.emptyResult();
}

pub fn builtinParseXml(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const status = parseXml(args.text_in[0], args.store, args.target_kb_id);
    if (status != .ok) return dispatch_mod.errorResult(status);
    return dispatch_mod.emptyResult();
}

pub fn builtinParseYaml(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const status = parseYaml(args.text_in[0], args.store, args.target_kb_id);
    if (status != .ok) return dispatch_mod.errorResult(status);
    return dispatch_mod.emptyResult();
}

pub fn builtinToJson(args: *BuiltinArgs) BuiltinResult {
    const len = toJson(args.store, args.target_kb_id, args.text_out);
    args.text_out_len.* = len;
    return dispatch_mod.emptyResult();
}

pub fn builtinToCsv(args: *BuiltinArgs) BuiltinResult {
    const delim: u8 = if (args.int_arg_count > 0) @intCast(args.int_args[0]) else ',';
    const len = toCsv(args.store, args.target_kb_id, delim, args.text_out);
    args.text_out_len.* = len;
    return dispatch_mod.emptyResult();
}

pub fn builtinToFraction(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    const result = toFraction(@intCast(args.int_args[0]), @intCast(args.int_args[1]));
    return dispatch_mod.factResult(result);
}

pub fn builtinFromFraction(args: *BuiltinArgs) BuiltinResult {
    const val = args.input_facts[0].value;
    const frac = fromFraction(val);
    return dispatch_mod.intResult(@intCast(frac.numerator));
}

pub fn builtinVdrToDecimalString(args: *BuiltinArgs) BuiltinResult {
    const val = args.input_facts[0].value;
    const prec: i32 = if (args.int_arg_count > 0) args.int_args[0] else 4;
    const len = vdrToDecimalString(val, prec, args.text_out);
    args.text_out_len.* = len;
    return dispatch_mod.emptyResult();
}

pub fn builtinDecimalStringToVdr(args: *BuiltinArgs) BuiltinResult {
    if (args.text_in_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const result = decimalStringToVdr(args.text_in[0]) orelse return dispatch_mod.errorResult(.err_command_parse);
    return dispatch_mod.factResult(result);
}

pub fn builtinBaseConvert(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 2) return dispatch_mod.errorResult(.err_command_parse);
    const val: i64 = @intCast(args.int_args[0]);
    const to_base = args.int_args[1];
    const len = baseConvert(val, to_base, args.text_out);
    args.text_out_len.* = len;
    return dispatch_mod.emptyResult();
}

pub fn builtinTimestampToFields(args: *BuiltinArgs) BuiltinResult {
    if (args.int_arg_count < 1) return dispatch_mod.errorResult(.err_command_parse);
    const fields = timestampToFields(args.int_args[0]);
    return dispatch_mod.intResult(fields.year * 10000 + fields.month * 100 + fields.day);
}
