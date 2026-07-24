# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-24 — **`briefings/hunt-support-stats.md` is
COMPLETE (all four items), pushed, awaiting mac-local review** (the
briefing stays in place until then). Nothing in flight; next work
arrives via briefings or review feedback.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no GPU, no CFS cap.
`/workspace/.agent_id` = runpod-b. Git identity set (Han); push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)`.
HF token at `/workspace/.tokens/hf_token`. Meter $1.63/$25.

## What shipped this session (hunt-support-stats, all four items)

1. **Variance receipts** — `task_hunt/support_stats/stage2_variance.{json,md}`
   (builder `stage2_variance.py` + `stats_lib.py`, exact small-n tests;
   6 sanity tests in `tests/test_support_stats.py`). Headlines: the
   TXC-pre T=2→8 RISE is significant (exact within-seed permutation
   p = 0.0093; its trained−untrained margin trend p = 0.0046 = the
   1/216 floor); trained−untrained margins bound away from 0 (pre/T8
   t CI [0.086, 0.215]); pre − per-token bounded at T8/T16; but the
   cross-arch TXC-pre − T-SAE paired margin is NOT bounded at n = 3
   (t CI [−0.086, 0.190]; pairing bought nothing, arms r = −0.21).
   **Seeds recommendation posted in the LOG addressed to runpod-d: 3
   extra seeds × {pre/T4, pre/T8, tsae/T1} = 9 trained cells (12 with
   the headroom seed).**
2. **Variance-aware renderer** — `lambda_intensity/render_stage2.py`:
   95% t-CI whiskers, realized-l0 legend ranges, TXC-post
   NOT-budget-matched flag + on-plot annotation (review note 3),
   budget-matched-only variant fig. `stage2_summary.json` old fields
   preserved byte-identical; adds ci95_trained/l0_range/budget_matched/
   match_rule. runpod-d just re-runs the module after its cells land
   (LOG note posted).
3. **Anti-conversion data side** (class stays PARKED; screen needs a
   freed pod + mac-local greenlight) — `labels/build_interleave.py` +
   `interleave_lib.py` + 5 tests + `interleave_fineweb_{gpt2,gemma2,
   llama31}.npz` + `interleave_stats.json` + `interleave/CARD_DRAFT.md`.
   Triage: source unigram AUC 0.66 matched vs 0.70 random (lexical
   control real but modest — kill-risk face); tss ≈ 0.55 (near-blind,
   the carrying face); hazard mildly rising (disclosed). Methods note:
   in-corpus unigram estimators leak the source via count asymmetry —
   triage estimates from held-out doc halves.
4. **Hedging-LEVEL draft card** — `confidence/LEVEL_CARD_DRAFT.md` for
   runpod-e (window-mean-level primary + marked decision points,
   shuffle-IMMUNITY as the disclosed receipt, code-readout-convention
   sentence, T ladder to 32 per the clock bridge). runpod-e freezes its
   own.

Tests: full suite green (the only failure mode seen was
`test_diff_hash_consistent_with_dirty`, which fails on any UNTRACKED
file — environmental, passes on a clean tree).

## Context worth keeping
- Round-1: five kills stand (review notes 1–5 in the LOG ~line 575);
  Stage-2 λ̂ panel is the positive; λ̂_hist primary target.
- Round-2 in flight: runpod-d budget-matched TXC-post re-run (its
  amendment card is frozen; my seed top-up may append to it) and
  runpod-e hedging-LEVEL Stage-2 + early-layer addendum.
- Shared-branch: 5 agents on arxiv; pull-rebase before EVERY push;
  cite commit SUBJECTS not SHAs; no case-colliding filenames; no
  reviewer/meeting quotes in tracked files.
- Rewrite this file before any compact.
