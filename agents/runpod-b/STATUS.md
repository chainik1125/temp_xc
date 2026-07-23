# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-23 — **STORY-PACK SESSION COMPLETE, stopped at the
acceptance gate. Awaiting mac-local review; briefing
`briefings/synth-story-pack.md` left in place.** No task in flight.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no CFS cap.
`/workspace/.agent_id` = runpod-b. Git identity set (Han); push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)`.
Freqbench meter unchanged: **$1.63 / $25** (this session spent $0 — assembly
only, no skeptic, no training).

## Story-pack outcome (all committed)

Assembly session per the briefing — no new cards/benches/rule edits, no
grid cells trained; everything re-derived from the existing leaderboard:

- **`experiments/explorations/synthetic/STORY.md`** — the distilled
  TXC-vs-T-SAE-vs-per-token story: § 1 regime table with REPORT-cell
  receipts + the explicit ambience point; § 2 the isolation figure;
  § 3 subtype rule + the frozen-prediction scorecard (holds AND misses,
  incl. FB-4 REFUTED and the FB-5 mixed/mechanism split); § 4 T-SAE
  positioning + the sparse-probing regime-1 corollary (paper §§ cited for
  paper numbers); § 5 robustness/budget-parity conventions; § 6
  parameter/inference-cost table (exact counts from instantiated archs).
- **`story_figs.py`** (committed pre-run) + `figs/story_isolation.{png,pdf}`
  + `results/story_stats.json` — extraction reuses the render_report
  matched-group machinery (B*=2, d_sae=F, 3-seed min–max); panels at each
  bench's canonical verdict slice (T=4; recipe residual T=2, its record's
  window — the T=4 falloff to −0.23 is stated in STORY § 2).
- Figure receipts worth remembering: backtracking 0.402 token vs
  0.939–0.952 all four window archs; frequency post 0.767 / spectral 0.777
  vs stacked 0.064; phasepair sign post 0.988 alone (spectral −0.004 at
  T=4); recipe residual spectral **0.973** at the T=2 matched cell vs
  ≈ −0.78 all others. Param counts: token/T-SAE 26,085; stacked/TXC ≈
  T×; spectral 26,469 at T=4 (singleton bands) / 52,581 at T=8.
- Tests 180 green. No reviewer text quoted anywhere (per briefing).

## Items for mac-local review
- STORY.md § 4's probing paragraph states the corolary with the paper's
  actual spread (0.886–0.907 panel-wide; temporal variants 0.897–0.902)
  rather than the briefing's "~0.001" shorthand — the briefing's point
  (probing cannot adjudicate temporal archs) is kept verbatim in meaning.
- README does not yet link STORY.md (left unlinked — README edits are
  program-rule territory; add a pointer at review if wanted).

## Operational notes
- Parallel agents: `runpod` (loss dissection), `runpod-c` (EM redo).
  Shared-branch rules; cite commit SUBJECTS not SHAs.
- Rewrite this file before any compact.
