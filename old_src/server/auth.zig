// ============================================================
// src/server/auth.zig
// ============================================================

const fact_mod = @import("../kb/fact.zig");
const VlpFact = kb_types.VlpFact;

pub const AuthResult = struct {
    status: VlpStatus,
    credential: ServerCredential,
};

pub fn authenticate(store: *KBStore, auth_kb_id: i32, token: []const u8, ttl_seconds: i32) AuthResult {
    var result = AuthResult{
        .status = .err_kb_access_denied,
        .credential = defaultCredential(),
    };

    if (auth_kb_id < 0) return result;
    if (token.len == 0) return result;

    const token_hash = hashCredential(token);

    const kb = store.getKB(auth_kb_id);
    if (kb == null) return result;

    const fc: usize = @intCast(kb.?.facts_count);
    var user_id: i32 = -1;

    for (0..fc) |i| {
        const si: i32 = @intCast(i);
        const fact = fact_mod.factQuery(store, auth_kb_id, si) orelse continue;
        if (fact.tag == .counter and fact.value.v == token_hash) {
            user_id = fact.provenance.source_slot_id;
            break;
        }
    }

    if (user_id < 0) {
        result.status = .err_kb_access_denied;
        return result;
    }

    var visibility: i8 = 0;
    const vis_slot = user_id * 4 + 1;
    const vis_fact = fact_mod.factQuery(store, auth_kb_id, vis_slot);
    if (vis_fact) |vf| {
        visibility = @intCast(vf.value.v);
    }

    const status_slot = user_id * 4 + 3;
    const status_fact = fact_mod.factQuery(store, auth_kb_id, status_slot);
    if (status_fact) |sf| {
        if (sf.value.v != 1) {
            result.status = .err_kb_access_denied;
            return result;
        }
    }

    const now = timestampNow();

    result.credential = .{
        .user_id = user_id,
        .visibility_level = visibility,
        .grants = undefined,
        .n_grants = 0,
        .issued_at = now,
        .expires_at = now + ttl_seconds,
        .valid = true,
    };

    const grant_slot = user_id * 4 + 2;
    const grant_fact = fact_mod.factQuery(store, auth_kb_id, grant_slot);
    if (grant_fact) |gf| {
        const grant_kb_id = gf.value.v;
        loadGrants(store, grant_kb_id, &result.credential);
    }

    result.status = .ok;
    return result;
}

fn loadGrants(store: *KBStore, grant_kb_id: i32, credential: *ServerCredential) void {
    const kb = store.getKB(grant_kb_id) orelse return;
    const fc: usize = @intCast(kb.facts_count);
    var count: i32 = 0;
    for (0..fc) |i| {
        if (count >= 16) break;
        const si: i32 = @intCast(i);
        const fact = fact_mod.factQuery(store, grant_kb_id, si) orelse continue;
        if (fact.tag == .empty) continue;
        const ci: usize = @intCast(count);
        credential.grants[ci] = .{
            .class = @intCast(fact.provenance.source_slot_id),
            .target_hash = fact.value.v,
            .remaining_uses = fact.value.r0,
            .expires_at = fact.provenance.timestamp,
        };
        count += 1;
    }
    credential.n_grants = count;
}

pub fn credentialCheck(credential: *ServerCredential) bool {
    if (!credential.valid) return false;
    const now = timestampNow();
    if (credential.expires_at > 0 and now >= credential.expires_at) {
        credential.valid = false;
        return false;
    }
    return true;
}

pub fn credentialRevoke(credential: *ServerCredential) void {
    credential.valid = false;
}

pub fn hashCredential(token: []const u8) i32 {
    var h: u32 = 2166136261;
    for (token) |byte| {
        h ^= @intCast(byte);
        h *%= 16777619;
    }
    return @bitCast(h);
}

pub fn createAuthKB(store: *KBStore, parent_id: i32) i32 {
    return store.createKB(.{
        .name = "auth",
        .parent_id = parent_id,
        .visibility = .owner_only,
        .owner = "system",
        .max_facts = 1024,
        .max_rules = 0,
        .max_children = 16,
    });
}

pub fn registerUser(store: *KBStore, auth_kb_id: i32, user_id: i32, token: []const u8, visibility: i8) VlpStatus {
    const token_hash = hashCredential(token);
    const base_slot = user_id * 4;

    const hash_fact = VlpFact{
        .tag = .counter,
        .value = .{ .v = token_hash, .r0 = 0 },
        .provenance = .{
            .source_type = .database,
            .source_kb_id = auth_kb_id,
            .source_slot_id = user_id,
            .confidence = .{ .v = Q16.D, .r0 = 0 },
            .timestamp = timestampNow(),
            .derivation_rule_id = -1,
        },
    };
    _ = fact_mod.factAssert(store, auth_kb_id, base_slot, &hash_fact);

    const vis_fact = VlpFact{
        .tag = .counter,
        .value = .{ .v = @intCast(visibility), .r0 = 0 },
        .provenance = hash_fact.provenance,
    };
    _ = fact_mod.factAssert(store, auth_kb_id, base_slot + 1, &vis_fact);

    const grant_fact = VlpFact{
        .tag = .reference,
        .value = .{ .v = -1, .r0 = 0 },
        .provenance = hash_fact.provenance,
    };
    _ = fact_mod.factAssert(store, auth_kb_id, base_slot + 2, &grant_fact);

    const status_fact = VlpFact{
        .tag = .counter,
        .value = .{ .v = 1, .r0 = 0 },
        .provenance = hash_fact.provenance,
    };
    _ = fact_mod.factAssert(store, auth_kb_id, base_slot + 3, &status_fact);

    return .ok;
}

pub fn suspendUser(store: *KBStore, auth_kb_id: i32, user_id: i32) VlpStatus {
    const status_slot = user_id * 4 + 3;
    const status_fact = VlpFact{
        .tag = .counter,
        .value = .{ .v = 0, .r0 = 0 },
        .provenance = .{
            .source_type = .database,
            .source_kb_id = auth_kb_id,
            .source_slot_id = user_id,
            .confidence = .{ .v = Q16.D, .r0 = 0 },
            .timestamp = timestampNow(),
            .derivation_rule_id = -1,
        },
    };
    _ = fact_mod.factAssert(store, auth_kb_id, status_slot, &status_fact);
    return .ok;
}

pub fn reactivateUser(store: *KBStore, auth_kb_id: i32, user_id: i32) VlpStatus {
    const status_slot = user_id * 4 + 3;
    const status_fact = VlpFact{
        .tag = .counter,
        .value = .{ .v = 1, .r0 = 0 },
        .provenance = .{
            .source_type = .database,
            .source_kb_id = auth_kb_id,
            .source_slot_id = user_id,
            .confidence = .{ .v = Q16.D, .r0 = 0 },
            .timestamp = timestampNow(),
            .derivation_rule_id = -1,
        },
    };
    _ = fact_mod.factAssert(store, auth_kb_id, status_slot, &status_fact);
    return .ok;
}
