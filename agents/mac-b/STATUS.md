# Working state — agent `mac-b`

**2026-07-26 ~12:15 London (day-2 sprint, briefing
`briefings/day2-dialogue-mac-b.md`).** Day-2 cap $60; mac-b day-2
ACTUALS ≈ $1. **W1 COMPLETE** — the R11 order-mechanism ladder run,
verdicted, receipted, pushed.

## Day-2 record (PENDING TEAM REVIEW + mac-local ratification)

**W1 — dialevel R11 ladder (frozen `ede97e206`, mac-local
freeze-APPROVED pre-results): verdict MIXED on 3/3.** The one
order-carried window signal outside backtracking decomposes
ADDITIVELY (|L1+L2−L0| ≤ 0.004) into within-turn token order (L1
share 0.51/0.38/0.57 of L0 at T32, gpt2/llama/gemma) and turn-block
order (L2 share 0.56/0.53/0.43), and is concentrated in the NEAR
half (llama: far −0.007, near +0.037 ≥ full cost +0.035). All four
identity gates passed — L0 seed-0 reproduced committed R11 to
≤ 0.0013 on rebuilt caches. Receipt R25 (checker ALL PASS, test
green). LOG entry with the binding reach disclosures printed.
Results `dialevel/results/ladder_{gpt2,llama31_8b,gemma2_2b}.json`.

## Standing state / if resuming

- Modal Volume `temp-xc-replag-caches` now also holds
  `/workspace/dialevel_caches/{gpt2,llama31_8b,gemma2_2b}` (3 layers
  each) + `/workspace/dialevel_results/` mirrors. HF secret
  `hf-token` live in workspace (gemma GO; Han rotates post-weekend).
- Driver: `scripts/modal_dialevel_ladder.py` (L40S, sequential
  .remote + retries, in-container caches, launch via
  **`uvx modal run --detach`** — plain `modal` is NOT on PATH here).
- Ladder scorer: `.venv/bin/python -m
  experiments.explorations.task_hunt.dialevel.ladder_score`.
- Overnight record (slen/refmark/quotedens, R20–R24) unchanged —
  see LOG; verdicts await Sunday 10:00 PT team review.
- **~13:25 London REASSIGNMENT (LOG race-resolution entry): gemma
  fills DROPPED mid-slen** (app stopped; ~$1 actual; gemma replag
  cache + slen screen partials persist on Volume — resumable
  post-deadline; refmark/quotedens fills never launched). **mac-b is
  now PANEL 2 (dq/llama31) merge + receipts support**: mac-a owns
  cells + verdicts; I own merge-locally + variance-harness receipts
  (`support_stats/stage2_variance.py`, invocation per
  `support_stats/PANEL_RECIPES.md`: --row-layout paired,
  --post-k-rule times-T, --crosscheck-json the panel's own stage2
  json) + receipts_check rows. Panel-2 deadlines: freeze 13:30,
  launch 13:45, repatriation 16:15, everything pushed 16:30.
- Day-2 gates: NO new Modal starts after 15:30 London; briefings
  retire at the 18:00 check-in.
