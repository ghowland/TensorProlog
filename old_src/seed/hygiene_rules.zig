// ============================================================
// src/seed/hygiene_rules.zig
// ============================================================

pub fn loadHygieneRules(store: *KBStore, hygiene_kb: i32) void {
    assertTextFact(store, hygiene_kb, 0, "stale_rule_detector");
    assertIntFact(store, hygiene_kb, 1, 7776000);
    assertTextFact(store, hygiene_kb, 2, "condition:rule_last_fired_age_gt_threshold");
    assertTextFact(store, hygiene_kb, 3, "action:assert_candidate_for_pruning_stale");

    assertTextFact(store, hygiene_kb, 4, "failing_rule_detector");
    assertIntFact(store, hygiene_kb, 5, 5);
    assertIntFact(store, hygiene_kb, 6, 20);
    assertTextFact(store, hygiene_kb, 7, "condition:rule_fire_count_gt_min_and_success_rate_lt_threshold");
    assertTextFact(store, hygiene_kb, 8, "action:assert_candidate_for_pruning_failing");

    assertTextFact(store, hygiene_kb, 9, "orphan_rule_detector");
    assertTextFact(store, hygiene_kb, 10, "condition:rule_references_grant_and_grant_state_revoked");
    assertTextFact(store, hygiene_kb, 11, "action:assert_candidate_for_pruning_orphaned");
}
