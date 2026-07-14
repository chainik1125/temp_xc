# Grounded-benchmark expansion — coverage ledger

The **anti-drift invariant** for the autonomous expansion loop (see
[`README.md`](README.md), the standing pipeline doc). Every candidate temporal
property lands in exactly one `domain × temporal-class` cell. Selection each cycle
must (a) keep a **per-domain floor** (≥⌊N/2⌋ calibrated from each domain) and
(b) **prioritize under-covered cells** (empty / abort-only) over cells that
already hold a PROCEED. Update this file at the end of every cycle and report the
grid — an empty domain is then a visible, prioritized gap, not silent drift.

**Cell status:** `—` empty · `prop` proposed (prereg frozen) · `calib:ABORT` ·
`calib:PROCEED` · `SPEC` (PROCEED graduated to a `synthetic/<name>/` benchmark).

| temporal-class | reasoning-trace | text-corpus |
|---|---|---|
| **DC-slow-drift** (state persists, slow) | **`SPEC`** — uncertainty-hedging-drift (ACF(1)=0.32 ≫ N1 0.078/N2 0.030; κ=0.64; skeptic 5/5; → `synthetic/hedging_drift/`) | `calib:ABORT` — topic_switching (autocorr 82% per-doc *composition*, not order; labeler inadequate) · `prop` — hedge-to-assertion-drift (C1, unselected) |
| **AC-order-sensitive** (depends on order) | **`SPEC`** — assumption-then-consequence (directed asym 0.135 ≫ nulls ≤0.024; skeptic 5/5; → `synthetic/assumption_consequence/`) | `calib:ABORT` — question-answer-adjacency (gate passed but **skeptic kill: leakage** — ANSWER label definitionally requires a preceding question; circular) |
| **periodic** (rhythmic/cyclic) | `prop` — computation-verification-alternation (C1, unselected) | `prop` — enumeration-cadence (C1, unselected) |
| **bursty/self-exciting** (clustered events) | **`SPEC`** — backtracking (ACF(1)=0.36, ~3.6× self-excitation; Sonnet judge; Hawkes mirror) · `prop` — error-correction-cascade (C1, unselected; cell already PROCEEDed) | `calib:ABORT` — quotation-burst (gate passed, ACF(1)=0.30 ≫ N1 0.128, but **skeptic kill: circularity** — mirror validated only on its fitted moment; non-fitted Fano/excite missed) |
| **interaction/equality** (cross-position compare) | — | — |
| **long-memory** (renewal / heavy-tail) | `prop` — goal-restatement-recurrence (C1, unselected) | `prop` — pronoun-referent-recurrence (C1, unselected) |

## Notes / provenance

- **backtracking** is the anchor — the one property that ran the full measure→
  mirror loop by hand and PROCEEDed (`synthetic/backtracking/`). The Cycle-1
  automation imitates it.
- **topic_switching** is the cautionary ABORT — the loop working correctly on a
  property that turned out to be composition, not order.
- **Cycle-1 skeptic kills are working as designed:** question-answer-adjacency
  died on labeler leakage (the trap the rubric's item (b) names verbatim);
  quotation-burst died on mirror-validation thinness (item (d)) despite a
  statistically solid gate — its record recommends preregistering a
  non-fitted-moment tolerance before any refreeze attempt.
- The abstract benchmarks (signed_motion, frequency) are **not** in this ledger —
  mathematical constructs, out of scope for the grounded loop.
- **interaction/equality remains the only class with zero proposals** — the
  natural priority for Cycle 2, both domains (plus periodic and long-memory,
  proposed but never calibrated).

## Cycle log

_(one line per cycle: date · candidates calibrated (domain split) · verdicts ·
spend · what the next cycle targets)_

- **Cycle 1 — 2026-07-14 (runpod, autonomous).** 10 cards frozen (5+5); 4
  calibrated (2 reasoning + 2 text): assumption-then-consequence **PROCEED→SPEC**,
  uncertainty-hedging-drift **PROCEED→SPEC**, question-answer-adjacency
  **ABORT** (skeptic: leakage), quotation-burst **ABORT** (skeptic: mirror
  circularity). Spend **$9.55** of $25 (Haiku bulk ≈ 123k sentence labels;
  Sonnet interjudge ×4; Opus hypothesize/select/skeptic ×5). Next cycle should
  target: interaction/equality (both domains, empty), text-corpus PROCEEDs
  (0/2 this cycle — both text kills were labeler/mirror-methodology, not
  "text has no temporal structure"), periodic + long-memory (proposed,
  uncalibrated); and preregister a non-fitted-moment mirror tolerance.
