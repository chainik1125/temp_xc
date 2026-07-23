# Working state — agent `runpod`

**Last rewrite:** 2026-07-23, after completing `briefings/expansion-c6.md`.
**State: C6 DONE — STOPPED FOR REVIEW** (briefing's acceptance gate; the
briefing file stays until mac-local review deletes it). No active task.

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am `runpod`
(original box — `/workspace/.agent_id` does NOT exist; do not create it).**
THREE agents tonight: `runpod-b` (freqbench-t16-fbc2), `runpod-c`
(conversion-depth, GPU) — their briefings are NOT mine. Two-agent rules
(agents/README.md): `git pull --rebase origin arxiv` before EVERY push
(commit this STATUS file first — rebase refuses with unstaged changes);
shared files append-only; leaderboard/manifest union-merge. Tokens in
`/workspace/.tokens/` (`gh_token`, `anthropic_key` → export
ANTHROPIC_API_KEY). Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
32 CPU / 128 GB; grids: 28 workers × OMP_NUM_THREADS=1, ~90 min/495 cells;
harness-tracked background Bash (notifies on exit); python -u.

## Just completed: C6 (2026-07-23) — do not redo
**Verdict: NEITHER calibrated candidate passes the frozen battery → NO r4;
r3 ABORT stands.** $0 API spend (cumulative $10.82/$25 unchanged; skeptic
never reached). Full narrative: `expansion/results/estimator_battery_c6.md`
+ `estimator_battery_c6.json` + `estimator_battery_c6_lamscan.json`;
LEDGER C6 cycle-log entry; research STATUS § 0 bullet; BENCHMARKS r3 row +
prize paragraph updated. Key facts a reviewer will ask about:
- Card frozen PRE-BUILD at `8c46a92b` (estimator-card-c6-segment-extraction
  + r4 amendment in proof-operation-phase-runs.md). Battery run exactly as
  frozen; nothing retuned after results.
- Candidates in `src/explorations/synthetic/expansion/mirrors.py`
  (`fit_seg_hier_categorical_cal`, `fit_seg_hier_categorical_deflate` +
  shared `_seg_collect`/`_seg_finish`; r3 fit untouched byte-for-byte).
  MENU keys registered; calibrate.py insertion-control trigger generalized
  to `startswith("seg_hier_categorical")` (one-line, unused since r4 never
  ran).
- Headline mechanism: shrink-then-deconvolve NON-monotone (deconvolution
  re-amplifies; real-material λ-scan flat 0→0.9 then cliff at 1.0 = zero
  extraction; even inert undershoots the permuted mid-lag floor). Deflate
  collapses 75% of segments, still leaks +0.012/+0.016 via tails. Both
  cancel weak-regime winner's curse (raw +34% → ≤3%) — pinned as pytest
  rail `test_c6_calibrated_estimators_cancel_weak_regime_curse`
  (deterministic: replicates battery-4 exactly).
- C7 direction (proposed, NOT frozen): deconvolve-first-then-shrink
  (monotone) + variance-aware in-loop margins; if still inert → close the
  reasoning int/eq half at this corpus resolution.
- NO stage-6 anything from C6 (briefing constraint honored).

## Earlier completed (reviewed or pushed; do not redo)
- **Stage-6 #3b (2026-07-23): recipe_instruction_phase_runs POSITIVE**
  (first grounded regime-3 separation; reviewed & approved).
- **C5 (2026-07-22): r3 ABORT** (doubly informative; reviewed & approved).
- **Stage-6 § 8 STOP → re-scope option 1** (reviewed).

## Repo state
Working tree should be clean and in sync with origin/arxiv after the C6
commits (mirrors.py + battery + card + trackers + this file). Suite 161
passed; `run.py validate` OK. If resuming: nothing is mid-flight; wait for
review or a new briefing.

## Gotchas (this box)
- Harness blocks `sleep` (use `until …; done` loops or background tasks).
  Background python needs `-u`.
- 5-family models reject `temperature` and think by default (client
  handles); calibrations SEQUENTIAL (spend meter is per-process).
- render_report churns fig-PDF timestamps with no content change —
  `git checkout -- experiments/explorations/synthetic/figs/` if only
  binaries moved.
- Concurrent-git race FIXED in grid.py (GIT_OPTIONAL_LOCKS=0, os.replace,
  retry×3).
- Skeptic verdicts: persisted raw pre-parse, cached, NEVER re-rolled.
