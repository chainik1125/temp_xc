---
status: active
created: 2026-07-23
for: runpod
venue: runpod
---

# Expansion C6 — the calibrated extraction estimator (reasoning int/eq cell)

**You are `runpod`** (no `/workspace/.agent_id` on your box). Two other
agents run tonight (`runpod-b`: freqbench; `runpod-c`: conversion-depth,
GPU) — their briefings are not yours; shared-branch rules in
`agents/README.md` apply (pull-rebase before push; append-only shared
files). Prime directive: **a sound verdict, never a win.**

**Context.** C5 (`proof-operation-phase-runs-r3`, reviewed & APPROVED)
isolated the gap: the three-timescale structure is CONFIRMED
model-independently (real-vs-permuted ACF(4) gap 0.056), the
`seg_hier_categorical` family closes the lag-2–8 moments, but the
**extraction estimator over-extracts** — the preregistered insertion
control caught +0.018/+0.039 hallucinated moments on exchangeable data.
C6 = fix the *estimator*, not the model family (LEDGER C6 entry; C5
record `expansion/records/proof-operation-phase-runs-r3/`).

**Session limits:** ~12 h wall · **$10 API cap** (no fresh labeling —
reuse the committed labels; judgment on `claude-fable-5`; spend to
`expansion/results/spend.json`) · rewrite `agents/runpod/STATUS.md`
before any compact · **no program-rule/gate edits, no `temp_bench/core/`
edits.**

## The mandate

1. **Freeze the estimator card first** (commit before building): the
   candidate estimator(s) for calibrated segment-composition extraction,
   each with its *calibration principle* stated a priori — how it provably
   or verifiably does NOT extract structure from exchangeable streams
   (the insertion control moves from post-hoc check to in-loop
   constraint). Candidates you may draw on (C5 record § "campaign"):
   permutation-debiased moment matching (subtract the run-permuted
   estimate), split-sample honest estimation (fit segments on half, score
   moments on the held-out half), or a parametric-bootstrap calibration.
   No more than TWO candidates — depth over breadth.
2. **Verification battery before real data:** each candidate must (a)
   return ≈ 0 captured structure on run-permuted real streams (the C5
   insertion control, now an acceptance test), (b) recover known moments
   on synthetic streams *generated from* `seg_hier_categorical` with
   planted parameters (estimator consistency), (c) state its variance
   penalty (debiasing costs power — measure it on the planted streams).
3. **Then r4 on the real traces:** re-run the gate-8 moment checks with
   the calibrated estimator. Outcomes, all acceptable: PASS ⇒ the
   reasoning int/eq card finally graduates to SPEC (stage-6 later, NOT
   this session); FAIL with the structure gone ⇒ the C5 "structure" was
   winner's curse wholesale — record the NEGATIVE, close the card, and
   the prize's reasoning half is declared unreachable with current
   phenomena; FAIL with structure present but unextractable ⇒ record,
   propose C7 direction, stop.
4. **Skeptic** (expansion rubric, Fable) on any PASS; persist raw
   verdicts pre-parse.

## Acceptance gate — stop for review

Estimator card + battery + r4 verdict committed; LEDGER C6 entry;
research STATUS § 0 bullet (append-only); STATUS rewritten. **No stage-6
grid this session regardless of outcome.** Briefing stays until mac-local
review, then it is deleted.
