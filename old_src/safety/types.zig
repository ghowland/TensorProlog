// ============================================================
// src/safety/types.zig
// ============================================================

const vdr_types = @import("../vdr/types.zig");

pub const VlpStatus = vdr_types.VlpStatus;
pub const VlpGrantClass = vdr_types.VlpGrantClass;
pub const VlpGrantState = vdr_types.VlpGrantState;
pub const VlpAuditAction = vdr_types.VlpAuditAction;

pub const VlpGrant = struct {
    id: i32 = -1,
    class: VlpGrantClass = .filesystem,
    state: VlpGrantState = .active,
    holder_user_id: i32 = -1,
    target_offset: i32 = 0,
    target_length: i16 = 0,
    max_uses: i32 = -1,
    remaining_uses: i32 = -1,
    expires_at: i32 = 0,
    created_at: i32 = 0,
    created_by: i32 = -1,
    revoked_at: i32 = 0,
    revoked_by: i32 = -1,
};

pub const GrantCheckResult = struct {
    granted: bool = false,
    grant_id: i32 = -1,
};

pub const VlpAuditEntry = struct {
    timestamp: i32 = 0,
    session_id: i32 = -1,
    user_id: i32 = -1,
    action: VlpAuditAction = .fact_assert,
    target_kb_id: i32 = -1,
    target_slot_id: i32 = -1,
    grant_id: i32 = -1,
    result: i8 = 0,
    detail: i32 = -1,
};

pub const AuditFilter = struct {
    user_id: ?i32 = null,
    action: ?VlpAuditAction = null,
    after_ts: ?i32 = null,
    before_ts: ?i32 = null,
    target_kb_id: ?i32 = null,
};
