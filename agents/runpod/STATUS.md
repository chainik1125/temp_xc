# Working state — agent `runpod`

**Last rewrite:** 2026-07-22 (on the MIGRATED pod; stage-6 grids running).

## Who / where
Remote CC on RunPod (Linux). **Migrated pod (2026-07-22): repo root is
`/workspace/temp_xc`** (the `~/workspace` prediction was wrong — same old
path). Tokens in `/workspace/.tokens/` (present; `gh_token`, `anthropic_key`,
`hf_token`).
Push: `git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`
(from the repo root). Export anthropic_key as ANTHROPIC_API_KEY.

**Migrated-pod resources (measured 2026-07-22):** cgroup v1; `cpu.cfs_quota_us=-1`
(no CFS cap) with cpuset ⇒ **32 CPUs real** (`nproc`=32); memory cgroup says
128 GB but user-stated limit is **64 GB** — size for 64. 28 workers × OMP1
measured ~0.65 GB/worker (18 GB total) — lots of headroom. Old-pod throttling
gotchas don't apply here.
**Checkpoints did NOT survive migration** (`checkpoints/` = manifest only,
3.7 MB). Doesn't matter for done cells: `eval_key` hits the leaderboard row
BEFORE touching checkpoints (keys are pure config hashes, no git SHA), so all
459 assumption rows fast-forward. Only never-run cells train fresh.

## Current task: `briefings/stage6-grounded-eval.md` — IN PROGRESS (grid stage)
Build + blind-evaluate `assumption_consequence` (AC) + `hedging_drift` (DC).
Everything up to the grid is DONE and pushed:

- **Build committed** (`c1a2a24e`): generators (g7 Markov mirror; hier_ar1 w/
  210 empirical levels as code constant `_HEDGING_LEVELS_HIER`), datasources
  `toy_assumption_consequence_d64` / `toy_hedging_drift_d64`, evaluator
  add-ons `assumption_recovery` (state + next-state probes) /
  `hedging_recovery` (ridge R² on c_i) — additive, protocol stays 1.3.0.
  13 new tests; suite 120 passed.
- **§ 8 gates PASS, committed** (`01f79c6b`). Two pre-grid facts on record:
  (1) order-1 mirror ⇒ s_i sufficient ⇒ per-token and raw windows give
  IDENTICAL next-state readouts (0.464/0.466 balacc vs Bayes-balanced oracle
  0.544) — the frozen "per-token blind" prediction is tested against that;
  (2) hedging per-token raw ceiling R² 0.770 (multiplicative-noise bound),
  raw window headroom only +0.005 — spec oracle R²=1 unreachable.
- **Grid drivers committed** (`a8e6fb07`); render_figs + bench_record
  skeletons committed (placeholders — headline/verdict framing MUST be
  rewritten from actual numbers, never trusted as-is).
- **Grid partial state committed: 459/495 assumption cells, 0 failures**
  (leaderboard +458 rows @ 30k steps, seeds {1,2,42}). Hedging NOT started.

## Next actions
1. **RUNNING NOW**: assumption run_grid, 28 workers, OMP1, harness task
   `bwcxzxpqg` (log under the session scratchpad `tasks/`). 459/495
   fast-forwarded from leaderboard; ~36 cells training fresh. Venv python
   3.12.11 works — no uv reinstall needed.
2. When it exits: verify 0 failures, then launch
   `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m
   experiments.explorations.synthetic.hedging_drift.run_grid 28` (all 495
   fresh — checkpoints gone; harness-tracked background, NO nohup/disown).
3. Verify 0 failures → run both `render_figs` → **write prediction-vs-actual
   verdicts from the numbers** (blind discipline: report, never retune) →
   registry.py Bench entries + render_report → REPORT.md per-bench links →
   BENCHMARKS.md rows ✓ + verdicts → synthetic STATUS.md §0 → scoped
   commits → push → STOP for review (briefing stays until reviewed).

## Gotchas (this box — READ BEFORE SIZING ANYTHING)
- **Pod cgroup caps hide behind host numbers**: old pod = 8.5 CPUs
  (`cpu.max`) + 55 GB (`memory.max`) while `nproc`=96 / `free`=503 GB.
  64 workers ⇒ OOM-kill ⇒ silent BrokenProcessPool. 24 workers×OMP2 ⇒ 76%
  CFS-throttled, ~47 min/cell. Size pools by the cgroup quota (~2 GB/worker).
- **Pod restart wipes home dir ⇒ venv python vanishes** (broken symlink into
  `~/.local/share/uv/`). Fix per step 1; site-packages live in volume `.venv/`.
- `pkill`/`pgrep -f` self-match the launching shell/watcher — use
  `pgrep -f "[r]un_grid"`; prefer TaskStop on harness tasks.
- Long jobs: harness-tracked background Bash (notifies on exit). nohup+disown
  detaches from the harness AND died with the pod's process reaper anyway.
- Background python: launch with `-u` or prints sit in the block buffer.
- Claude 5-family models reject `temperature` AND think by default (client
  handles both). Calibrations sequential (spend meter is per-process).
