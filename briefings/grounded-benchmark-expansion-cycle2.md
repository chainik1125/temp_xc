---
status: active
created: 2026-07-14
for: runpod
venue: runpod
supersedes: grounded-benchmark-expansion.md (Cycle 1, done + reviewed)
---

# Grounded-benchmark expansion — Cycle 2

Cycle 1 is done and **reviewed (approved)**. The infra, the factory harness, and
the standing pipeline doc all exist — this is a **continuation, not a bootstrap.**
Read first: [`experiments/explorations/synthetic/expansion/README.md`](../experiments/explorations/synthetic/expansion/README.md)
(the pipeline + all guardrails) and [`.../expansion/LEDGER.md`](../experiments/explorations/synthetic/expansion/LEDGER.md)
(coverage + Cycle-1 verdicts). Same loop, same guardrails, same gated cadence:
**hypothesize → select → calibrate → freeze → STOP for review.**

## What changed after the Cycle-1 review (apply these)

Two **preregistered** gates were added to the README (§ guardrails 7–8) and the
prereg template — the Cycle-1 skeptic kills, promoted from reactive to design-time:

- **Gate 7 — no-leakage labeler.** A per-span label must be assignable from the
  span's **own content**, never its relation to neighbours ("answers a preceding
  question", "follows from prior"). Design every labeler to satisfy this up front.
- **Gate 8 — non-fitted-moment mirror.** Every mirror must reproduce **≥1
  statistic it was NOT fit to**, within a **preregistered tolerance**. Name that
  statistic + tolerance in the card's § 7 before fitting.

## Targets (from the ledger — under-coverage bias)

1. **interaction/equality — TOP priority.** The only class with **zero proposals**,
   both domains. *Hypothesize new cards here* (freeze before data). This is the
   axis the abstract changepoint bench probes (equality-pattern latents) — a
   grounded analogue is the prize.
2. **A text-corpus PROCEED.** Cycle 1 went 0/2 on text — but both kills were
   *methodology* (labeler leakage / mirror circularity), not "text has no temporal
   structure." Secure at least one clean text PROCEED under gates 7–8.
3. **periodic + long-memory.** Four **already-frozen, uncalibrated** cards exist
   (`computation-verification-alternation`, `enumeration-cadence`,
   `goal-restatement-recurrence`, `pronoun-referent-recurrence`) — they were frozen
   blind in Cycle 1, so calibrating them now is legitimate. Pull from these to fill
   those cells.

## Scope + budget

- **N ≈ 6, ≥3 per domain** (the loop + harness are proven; scale up from Cycle 1's
  N=4). Selection = the deterministic ledger rule (per-domain floor +
  under-coverage bias): interaction/equality first, then the frozen
  periodic/long-memory cards, keeping the per-domain floor.
- **Cost cap $25/cycle** (same). Cycle 1 was ~$2.4/candidate → N=6 + the two riders
  below fits well under the cap; meter it and hard-stop as before.

## Two riders (from the review — do these in-cycle)

- **Re-examine `assumption-then-consequence` (currently `SPEC*` provisional).**
  Its labeler references context ("follows from prior") → soft leakage (gate 7).
  Rewrite the judge instruction to be **strictly per-sentence** (mark a sentence a
  consequence from its own connectives only, with NO "context tells you" clause),
  re-label + re-measure. If the directed asymmetry **survives** → upgrade
  `SPEC*`→`SPEC`; if it **collapses** → it was leakage → `ABORT`. Either outcome is
  a good outcome (prime directive). This is a dated amendment to its frozen card,
  logged transparently.
- **Re-check both existing PROCEED mirrors against gate 8.** hedging-drift and
  assumption-consequence: confirm each reproduces a **non-fitted** moment within a
  stated tolerance (their Cycle-1 records report ACF/MI/dwell — verify at least one
  was NOT a fit target and passes). Record pass/fail per mirror.

## Acceptance gate (per cycle)

Everything in the README's per-cycle gate, **plus**: gates 7–8 applied to every
new card; interaction/equality no longer empty (≥1 calibrated per domain there is
ideal, ≥1 total is the floor); the assumption-consequence re-exam **resolved**
(SPEC or ABORT, logged); both existing mirrors re-checked against gate 8; LEDGER +
cycle log updated; committed + pushed; **STOP for review. No architecture runs.**

## Hard rules

`TEMP_BENCH_ALLOW_DIRTY=1`; `.venv/bin/python`; **never edit `temp_bench/core/`**;
calibration is text-only (no activations / no arch — those are the later blind
eval); version-pin the corpus snapshot + judge models in every record. Prime
directive: a sound verdict, never a win. When Cycle 2 is done + reviewed, delete
this briefing (the README stays as the standing doc).
