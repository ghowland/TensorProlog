// ============================================================
// src/seed/oso_rules.zig
// ============================================================

pub fn loadOsoRules(store: *KBStore, oso_kb: i32) void {
    var slot: i32 = 0;
    assertTextFact(store, oso_kb, slot, "P01:EXACT_ARITHMETIC:all_computation_uses_integer_vdr_triples");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P02:DETERMINISM:same_input_same_state_produces_same_output");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P03:BOUNDED_RESOURCES:all_data_structures_have_declared_capacity");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P04:STRUCTURAL_SAFETY:access_control_by_integer_comparison_not_behavior");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P05:PROVENANCE:every_fact_carries_source_type_and_confidence");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P06:AUDIT:every_operation_produces_append_only_log_entry");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P07:GRANT_BEFORE_EFFECT:operational_primitives_require_positive_grant");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P08:SNAPSHOT_EXACT:save_restore_produces_bit_identical_state");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P09:CLONE_ISOLATION:cow_writes_never_visible_to_parent");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P10:SESSION_SCOPE:data_absent_not_filtered_for_unauthorized_sessions");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P11:SOFTMAX_EXACT:output_row_sums_equal_D_by_integer_equality");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P12:GRAMMAR_STRUCTURAL:every_delimiter_from_template_not_llm");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P13:RECYCLE_FRESH:snapshot_kill_clone_preserves_knowledge_kills_drift");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P14:LEVEL_PROGRESSION:l1_full_llm_l2_rule_invoke_l3_auto_fire");
    slot += 1;
    assertTextFact(store, oso_kb, slot, "P15:NEGATIVE_ACCUMULATION:stale_failing_orphan_rules_detected_and_pruned");
    slot += 1;
}
