# Cycle-3 selection (Stage 2, blind)

Rule: 4 slots briefing-mandated (re-freezes + frozen C1 cards); 2 open slots = top labelability*novelty*predicted_temporalness per domain among the new interaction/equality cards; scorer=claude-opus-4-8 (blind: no data, no arch scores)

## Open slots — new interaction/equality cards

| candidate | domain | lab | nov | temp | score | picked |
|---|---|---|---|---|---|---|
| recipe-instruction-phase-runs | text-corpus | 4 | 4 | 4 | 64 | **✓** |
| proof-operation-phase-runs | reasoning-trace | 4 | 3 | 4 | 48 | **✓** |
| subgoal-discourse-segments | reasoning-trace | 3 | 2 | 3 | 18 |  |
| argument-claim-type-segments | text-corpus | 2 | 3 | 2 | 12 |  |

## Scorer reasons

- **proof-operation-phase-runs** — Operation-types are readable from a sentence's own verbs/symbols and the heuristic priority is clean, though verification vs restatement can blur; multi-class dwell/phase structure over 5 operations is a real step beyond the binary self-excitation attempts; math traces genuinely enact coherent sub-task blocks so runs should beat the position-conditional null, though algebra dominance risks starving the cap.
- **subgoal-discourse-segments** — Plan/execute/evaluate/conclude markers are recognizable but execute-step overlaps heavily with algebraic sentences and evaluate/conclude boundaries are fuzzy from a single line; discourse-move segmentation strongly echoes the operation-phase card in the same domain, offering little new axis coverage; execute-step will likely dominate long computations, compressing to one-class-plus-clustering and weakening the segment signal.
- **recipe-instruction-phase-runs** — Grammatical form (imperative verb / quantity listing / caution marker) is a strong single-sentence cue with a crisp heuristic, though non-instructional fineweb docs will dump into class 0; functional-segment runs in web instructional prose is a genuinely new domain+mechanism relative to the reasoning-heavy program and both binary web failures; ingredient/step/tip blocks are strongly contiguous so ACF(1) should clearly beat N2 if the sample is filtered to real how-to docs.
- **argument-claim-type-segments** — Assertion vs concession vs background is highly context-dependent and the assertion heuristic (is/are/should) will over-fire and swamp everything, violating the >75% floor; rhetorical claim-type segmentation is a distinct axis but overlaps conceptually with assumption-consequence and the other web card; assertion dominance plus interleaved evidence/example makes contiguous runs unlikely, so ACF(1) may not clear N1 beyond labeler noise.

## Mandated slots (briefing, not scored)

- **list-item-parallelism-r2** (text-corpus × interaction/equality) — briefing-mandated re-freeze (C2 gate-8 tolerance mis-scaled; strongest C2 signal, likely first text PROCEED)
- **computation-verification-r2** (reasoning-trace × periodic) — briefing-mandated re-freeze (real spectral peak; periodic_hawkes hybrid mirror replaces the periodic-only mirror gate-8 killed)
- **enumeration-cadence** (text-corpus × periodic) — frozen C1 card, fills periodic × text (ledger under-coverage)
- **goal-restatement-recurrence** (reasoning-trace × long-memory) — frozen C1 card, fills long-memory × reasoning (ledger under-coverage)

Full slate (6, 3 per domain): `proof-operation-phase-runs`, `recipe-instruction-phase-runs`, `list-item-parallelism-r2`, `computation-verification-r2`, `enumeration-cadence`, `goal-restatement-recurrence`

