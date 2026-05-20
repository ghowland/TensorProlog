// ============================================================
// src/seed/confidence_table.zig
// ============================================================

pub fn loadConfidenceTable(store: *KBStore, conf_kb: i32) void {
    assertValueFact(store, conf_kb, 0, .{ .v = 65536, .r0 = 0 });
    assertValueFact(store, conf_kb, 1, .{ .v = 65536, .r0 = 0 });
    assertValueFact(store, conf_kb, 2, .{ .v = 64225, .r0 = 0 });
    assertValueFact(store, conf_kb, 3, .{ .v = 62259, .r0 = 0 });
    assertValueFact(store, conf_kb, 4, .{ .v = 62259, .r0 = 0 });
    assertValueFact(store, conf_kb, 5, .{ .v = 55705, .r0 = 0 });
    assertValueFact(store, conf_kb, 6, .{ .v = 52428, .r0 = 0 });
    assertValueFact(store, conf_kb, 7, .{ .v = 45875, .r0 = 0 });
    assertValueFact(store, conf_kb, 8, .{ .v = 32768, .r0 = 0 });
    assertValueFact(store, conf_kb, 9, .{ .v = 19660, .r0 = 0 });
    assertValueFact(store, conf_kb, 10, .{ .v = 0, .r0 = 0 });
}
