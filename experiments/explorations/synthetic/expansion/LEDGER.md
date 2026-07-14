# Grounded-benchmark expansion — coverage ledger

The **anti-drift invariant** for the autonomous expansion loop (see
[`README.md`](README.md), the standing pipeline doc). Every candidate temporal
property lands in exactly one `domain × temporal-class` cell. Selection each cycle
must (a) keep a **per-domain floor** (≥⌊N/2⌋ calibrated from each domain) and
(b) **prioritize under-covered cells** (empty / abort-only) over cells that
already hold a PROCEED. Update this file at the end of every cycle and report the
grid — an empty domain is then a visible, prioritized gap, not silent drift.

**Cell status:** `—` empty · `prop` proposed (prereg frozen) · `calib:ABORT` ·
`calib:PROCEED` · `SPEC` (PROCEED graduated to a `synthetic/<name>/` benchmark) ·
`SPEC*` (PROCEED but **provisional** — flagged, not stage-6-runnable).

| temporal-class | reasoning-trace | text-corpus |
|---|---|---|
| **DC-slow-drift** (state persists, slow) | **`SPEC*` PROVISIONAL** — uncertainty-hedging-drift (measurement solid: ACF(1)=0.32, κ=0.64, skeptic 5/5; but **mirror INVALID under gate 8** — real ACF is a long-memory *plateau* (~0.13 to lag 8) that neither ar1+trend nor semi-Markov reproduces (both fail ACF(2) ±0.05); Cycle-3: hierarchical-AR(1) menu extension) | `calib:ABORT` — topic_switching (composition, labeler inadequate) · `prop` — hedge-to-assertion-drift (C1, unselected) |
| **AC-order-sensitive** (depends on order) | **`SPEC`** — assumption-then-consequence (**g7 re-exam RESOLVED C2**: strict per-sentence labeler, ctx=0 → asym **0.297** (was 0.135), gate-8 PASS, fresh skeptic 5/5 → `synthetic/assumption_consequence/`, canonical mirror = g7 fit) | `calib:ABORT` — question-answer-adjacency (skeptic kill: definitional leakage) |
| **periodic** (rhythmic/cyclic) | `calib:ABORT` — computation-verification-alternation (C2: spectral peak REAL, spec_peak 3.84 ≫ null 1.15, but **gate-8 mirror fail**: events also bursty, Fano 2.29 vs periodic_rate's 0.87; re-freeze candidate w/ periodic+self-exciting hybrid mirror) | `prop` — enumeration-cadence (C1, unselected) |
| **bursty/self-exciting** (clustered events) | **`SPEC`** — backtracking (the hand-run anchor) · `prop` — error-correction-cascade (C1, unselected; cell PROCEEDed) | `calib:ABORT` — quotation-burst (C1 skeptic kill: circular mirror validation) |
| **interaction/equality** (cross-position compare) | **`SPEC`** — self-reference-echo (C2: ACF(1)=0.31 ≫ nulls ≤0.07, gate-8 PASS on MI(1), skeptic 5/5 → `synthetic/self_reference_echo/`; ⚠ labeler marginal κ=0.30; ⚠ measured signature is run-clustering — class assignment loose, reviewer may relabel) · `calib:ABORT` — operator-alternation (preregistered NEGATIVE sign falsified: real +0.36 clustering, not alternation) | `calib:ABORT` ×2 — greeting-signoff-mirror (gate passed κ=0.67, but periodic_rate mirror produces zero MI(1) vs real 0.027 — mis-keyed mirror) · list-item-parallelism (strongest signal of C2, ACF(1)=0.52, κ=0.64, mirror NEAR-miss: Fano err 0.163 vs frozen tol 0.15 (4% relative) — **prime re-freeze candidate** with magnitude-scaled tolerance) |
| **long-memory** (renewal / heavy-tail) | `prop` — goal-restatement-recurrence (C1, unselected) | `calib:ABORT` — pronoun-referent-recurrence (C2: gap-CV 1.46 barely clears N1 1.21, dies at the noise floor — perturbed 1.08 falls back inside the band) |

## Notes / provenance

- **backtracking** is the hand-run anchor; the automation imitates it.
- **Cycle-1 lessons became design-time gates 7–8** (no-leakage labeler;
  non-fitted-moment mirror) — and Cycle 2 shows them working *before* the
  skeptic: 3 of 5 C2 aborts were cheap gate-8 kills (skeptic skipped).
- **Gate-8 tolerance-scaling lesson (C2):** tolerances preregistered as raw
  absolutes get mis-scaled when the statistic's magnitude is unknown
  (list-item-parallelism died at 4% relative error). Cycle 3 should preregister
  tolerances **relative to the statistic's magnitude or null-band width**.
- **Mirror-menu gap (C2):** two real phenomena the menu can't generate —
  long-memory plateaus (per-sequence levels / slow regimes; hedging) and
  periodic+bursty hybrids (verification). Both are concrete Appendix-B
  extension proposals for Cycle 3.
- **The g7 re-exam vindicated the strict-labeler discipline**: removing the
  relational clause *strengthened* the assumption→consequence asymmetry 2.2×.
- The abstract benches (signed_motion, frequency) are out of this ledger's scope.

## Cycle log

- **Cycle 1 — 2026-07-14 (runpod, autonomous).** 10 cards frozen (5+5); 4
  calibrated (2+2): assumption-then-consequence PROCEED→SPEC,
  uncertainty-hedging-drift PROCEED→SPEC, question-answer-adjacency ABORT
  (skeptic: leakage), quotation-burst ABORT (skeptic: mirror circularity).
  Spend $9.55/$25. Review (mac-local): approved; gates 7–8 added;
  assumption→SPEC* provisional; both mirrors to be gate-8 rechecked.
- **Cycle 2 — 2026-07-14 (runpod, autonomous).** 4 new interaction/equality
  cards frozen (2+2) under gates 7–8; 6 calibrated (3+3, deterministic
  selection) + the g7 re-exam rider + the gate-8 recheck rider. Verdicts:
  self-reference-echo **PROCEED→SPEC**; operator-alternation ABORT (sign
  falsified); computation-verification-alternation ABORT (gate-8: bursty≁periodic
  mirror); greeting-signoff-mirror ABORT (gate-8: zero-MI mirror);
  list-item-parallelism ABORT (gate-8 near-miss, re-freeze candidate);
  pronoun-referent-recurrence ABORT (noise floor). Riders:
  **assumption-consequence g7 re-exam → SPEC upgraded** (asym 0.297, 2.2× the
  contextual labeler); **hedging mirror gate-8 recheck FAILED** (+ preregistered
  semi-Markov attempt also failed) → SPEC downgraded to SPEC* pending a
  hierarchical-AR(1) menu extension. Spend **$14.06/$25**. **Text-corpus PROCEED
  target NOT met** (0/3 — two mirror-fidelity kills + one noise-floor kill; the
  underlying text signals were real in 2 of 3 cases). Next cycle should target:
  re-freeze list-item-parallelism (scaled tolerance) + computation-verification
  (hybrid mirror) for the text/periodic wins; the hierarchical-AR(1) +
  periodic-Hawkes menu extensions; relative gate-8 tolerances; long-memory ×
  reasoning (goal-restatement, frozen) and periodic × text (enumeration-cadence,
  frozen).
