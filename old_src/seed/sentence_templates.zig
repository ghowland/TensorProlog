// ============================================================
// src/seed/sentence_templates.zig
// ============================================================

pub fn loadSentenceTemplates(store: *KBStore, sentences_kb: i32) void {
    var slot: i32 = 0;

    assertTextFact(store, sentences_kb, slot, "finding:{service:text} shows {metric:text} at {value:vdr_value} which is {status:enum(normal|elevated|critical)}");
    slot += 1;
    assertTextFact(store, sentences_kb, slot, "triage:{service:text} incident classified as {severity:enum(low|medium|high|critical)} based on {evidence:text}");
    slot += 1;
    assertTextFact(store, sentences_kb, slot, "correlation:{metric_a:text} and {metric_b:text} correlation is {value:vdr_value} over {window:text}");
    slot += 1;
    assertTextFact(store, sentences_kb, slot, "status:{service:text} is {state:enum(healthy|degraded|down)} with confidence {confidence:vdr_value}");
    slot += 1;
    assertTextFact(store, sentences_kb, slot, "alert:{service:text} triggered {rule:text} at {timestamp:integer}");
    slot += 1;
    assertTextFact(store, sentences_kb, slot, "remediation:recommended action for {service:text} is {action:text} with priority {priority:enum(low|medium|high|immediate)}");
    slot += 1;
    assertTextFact(store, sentences_kb, slot, "summary:{count:integer} incidents across {services:integer} services in last {window:text}");
    slot += 1;
    assertTextFact(store, sentences_kb, slot, "coverage:{covered:integer} of {total:integer} services have active triage rules ({percent:vdr_value})");
    slot += 1;
    assertTextFact(store, sentences_kb, slot, "rule_created:new rule {rule_name:text} for {service:text} pattern {pattern:text}");
    slot += 1;
    assertTextFact(store, sentences_kb, slot, "rule_pruned:removed {rule_name:text} reason {reason:enum(stale|failing|orphaned)}");
    slot += 1;
    assertTextFact(store, sentences_kb, slot, "deploy:{service:text} deployed version {version:text} at {timestamp:integer}");
    slot += 1;
    assertTextFact(store, sentences_kb, slot, "rollback:{service:text} rolling back from {current:text} to {previous:text} reason {reason:text}");
    slot += 1;
}
