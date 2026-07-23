---
status: active
created: 2026-07-23
for: runpod
venue: runpod
---

# Expansion C7 — the monotone estimator, and (either way) closing the reasoning int/eq question

**You are `runpod`** (no `/workspace/.agent_id` on your box). `runpod-b`
(FB-5) and `runpod-c` (conversion-depth, GPU) run in parallel —
shared-branch rules in `agents/README.md`, including the NEW
commit-citation rule (cite commit subjects, or verify SHAs post-push).
Prime directive: **a sound verdict, never a win.**

**Context.** C6 (reviewed & APPROVED — `results/estimator_battery_c6.md`
incl. the review) ended with an empty passing set: shrink-then-deconvolve
is **non-monotone** (finding 1 — the deconvolution re-amplifies), and
per-doc quantile deflation leaks through its tails (finding 3). C7 builds
exactly the fix C6's record proposed and carries a **pre-specified close**
either way. This is the LAST estimator cycle for this cell — no C8
estimator proposals.

**Session limits:** ~10 h wall · **$10 API cap** (no fresh labeling;
skeptic on `claude-fable-5`; spend to `expansion/results/spend.json`) ·
rewrite `agents/runpod/STATUS.md` before any compact · no
program-rule/gate edits, no `temp_bench/core/` edits · **commit every
battery/analysis script BEFORE its first execution** (the strict
commit-then-run rule now applies to estimator batteries too — C6 review
process note).

## The mandate

1. **Freeze the C7 estimator card** (pre-build commit): ONE candidate —
   **deconvolve-first, then shrink in deconvolved space** (shrink the
   deconvolved segment propensities u toward the deconvolved doc fixed
   point, so generated composition interpolates monotonically between r3
   and inert; continuous λ by bisection once monotonicity is verified).
   The card must freeze: (a) the C6 battery gates 1–3 verbatim as the
   acceptance battery, amended ONLY by the **variance-aware margins**
   requirement adopted at the C6 review (replicate-count-adaptive
   tolerance at the boundary, specified numerically in the card — a seed
   flip may never decide a verdict); (b) a **monotonicity pre-check**
   (generated ACF(4) strictly decreasing in λ on real material, verified
   before any gate is scored — if non-monotone, the concept is dead too:
   record and go to the close, do not iterate on the mechanism); (c) the
   selection/close rule below, verbatim.
2. **Run the battery** (batteries 1–5 as C6, same frozen seeds where
   reusable; scripts committed pre-run).
3. **The pre-specified fork:**
   - **Candidate passes gates 1–3 → run r4** (fresh signature/gate/mirror
     on the real streams with the calibrated estimator, per the frozen
     C6-era r4 amendment). r4 PASS ⇒ the reasoning int/eq card graduates
     to SPEC (registry/BENCHMARKS/LEDGER updates; **no stage-6 grid this
     session**). r4 FAIL ⇒ record; the cell closes NEGATIVE (below).
   - **Candidate fails the battery, or λ\* ≈ inert on real material →
     close the reasoning half of the int/eq prize as NEGATIVE at this
     corpus resolution** (the C6 record's honest close): the structure is
     real (model-independently confirmed) but not extractable at 287 docs
     × ~85 sentences; the next lever is more/longer traces, not another
     estimator. Write the close into LEDGER (C7 entry), BENCHMARKS § B
     (proof-operation row updated), research STATUS § 0 (append-only).
     This close is a SUCCESS of the loop — say it plainly.
4. **Skeptic** (expansion rubric, Fable) on whichever branch fires;
   persist raw pre-parse.

## Acceptance gate — stop for review

Card + battery results + the fork's verdict committed and pushed; LEDGER
C7 entry; STATUS rewritten; spend logged. Briefing stays until mac-local
review, then it is deleted.
