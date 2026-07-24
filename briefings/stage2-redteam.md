---
status: active
created: 2026-07-24
for: runpod-b
venue: runpod (32C CPU)
---

# Stage-2 red team — break the two headline results before anyone else can

**You are `runpod-b`** (32C). The λ̂ Stage-2 QUALIFIED POSITIVE and the
backtracking shuffle receipt are about to carry rebuttal weight. Your
job: an adversarial audit, CPU-only, from committed code + label
artifacts + leaderboard rows (you cannot run activations — audit
what is checkable without them, and STATE what is not). You built the
independent λ̂ label pipeline, so you are the right fresh eyes.
Deliverable: `experiments/explorations/task_hunt/support_stats/REDTEAM.md`
+ any scripts (committed before outputs). **Any finding that threatens
a headline goes to the LOG immediately** (and names which claim it
threatens); cosmetic findings stay in the audit note. By Saturday
morning PT. Findings are wins — the prime directive cuts both ways.

## Target 1 — the λ̂ Stage-2 pipeline (`src/explorations/task_hunt/real_lambda.py` + `temp_bench/evals/lambda_recovery.py`)

1. **Split integrity (the sharpest question).** `_train_lambda_probe`
   splits sequences `n//2`. Establish from the committed datasource
   code + the ward stream/label artifacts: do train and eval halves
   share TRACES? If one Ward trace contributes rows to both halves,
   temporal autocorrelation in λ̂ could leak across the split. Answer
   with receipts (row→trace mapping recomputed from the committed
   builders/labels). If leakage exists: quantify the plausible
   inflation direction and whether it biases the ARCH COMPARISON
   (shared inflation across archs ≠ a ranking confound — say which
   claims survive) — then LOG it.
2. **Label-side confound ladder.** For λ̂_hist on the committed labels:
   what do trivial predictors achieve — position-in-trace (the frozen
   0.59 floor: reproduce it), sentence length, event-count-in-window,
   distance-since-last-event? Any trivial predictor approaching the
   per-token 0.78 or window ceilings is a finding.
3. **Bookkeeping.** Recompute the 84-row panel's eval_key uniqueness,
   seed/config completeness, and `stage2_summary.json` agreement from
   the leaderboard directly (independent re-derivation, not the
   existing crosscheck script).

## Target 2 — the shuffle receipt (`task_hunt/shuffle_receipt.py`)

From the committed script + results JSON: (a) confirm the shuffle is
per-row independent and cannot preserve order structure by accident
(seed handling, permutation scope); (b) confirm per-token/window
columns exactly reproduce the conversion-depth § 3 published values
(committed reference) and that `is_bt` vs anticipation rows are the
same probe rows (the receipt's whole force is identical-rows); (c)
check the σ_null derivation (17 vs 12 cells, which nulls). State
explicitly which parts need activations to re-verify and are
trusted-not-checked.

## Target 3 — the variance receipts (self-audit, brief)

You wrote them; now attack them: exact-test assumptions (exchangeability
across seeds), the BCa coarseness note's implications, and whether the
"6 seeds bound it" power calc survives the sd itself being an n=3
estimate (compute the answer under the sd's upper CI bound — if that
says 8–9 seeds, LOG it so runpod-d buys enough).

## Acceptance gate — stop for review

REDTEAM.md with a verdict per target (CLEAN / FINDING w/ severity /
NOT-CHECKABLE-WITHOUT-GPU); LOG entries for anything headline-
threatening; STATUS rewritten; no reviewer/meeting quotes. Briefing
stays until mac-local review.
