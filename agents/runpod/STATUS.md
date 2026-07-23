# Working state — agent `runpod`

**Last rewrite:** 2026-07-23 — **stage-6 #3b COMPLETE, STOPPED FOR REVIEW**
(briefing `briefings/stage6-recipe-rescoped.md` acceptance gate met; the
briefing stays until mac-local reviews, then deletes it).

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am `runpod`
(original box — `/workspace/.agent_id` does NOT exist; do not create it).**
`runpod-b` = FreqBench, parallel. Two-agent rules (agents/README.md):
`git pull --rebase origin arxiv` before EVERY push; shared files append-only;
leaderboard/manifest union-merge. Tokens `/workspace/.tokens/`; push
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
32 CPU / 128 GB; 28 workers × OMP1 for grids.

## Task just completed: stage-6 #3b — the re-scoped residual head-to-head

- **Phase A freeze (order provable from the log, all pre-grid):**
  A1 `cf4ae797` — `equality_residual_recovery` (balacc over [0.771, 1.0],
  unclipped; 0.771 = § 8 pair-additive ceiling, frozen constant, cited not
  re-derived; chance-normalized form kept as diagnostic).
  A2 `241845d2` — dated § 5-r re-freeze in bench_spec.md (mac-local's
  predictions restated verbatim + sharpening reasons only).
  A3 `d65349c0` — gating addendum: re-scoped § 8 condition (nonlinear 1.000
  ≫ additive 0.771) satisfied by the existing `b463c4a0` record.
- **Grid:** 495/495 in 89 min (28 workers), 0 failures, 0 duplicate
  eval_keys, canonical runner, protocol 1.3.0.
- **Blind verdict: POSITIVE** (bench_record.md § "Stage-6 #3b"): Spectral-TXC
  T=2 exposes the residual (+0.60/+0.90/+0.96 over d; peak +0.973 at k=2 —
  equality balacc 0.994 ≈ exact rule; k-robust to 4, dead at 8; untrained
  +0.06 ⇒ learning). TXC-post caps at the additive ceiling (best +0.26,
  T=4 d=40; k-fragile exactly as frozen) — the one prediction miss ("positive"
  predicted, ceiling-capped measured). Additive families −0.76…−0.86
  everywhere (falsifier NOT triggered). DC control ≈ oracle (exception:
  TXC-post T=8 0.49, its known squash price). Realistic-regime (win at
  d ≤ F) + capability (gAUC 0.798 ≈ per-token) gates pass.
- **Renders:** registry entry (2 axes, verdict POSITIVE), REPORT 66/66 rows
  (report_figs marker list extended 6→7 with an assert), BENCHMARKS row +
  prize section (text half FULLY claimed), research STATUS § 0 bullet.
  Suite 138 green.

## Next actions
**None — STOPPED.** Await mac-local review. Do not start C6 (reasoning-cell
extraction estimator) — that needs its own briefing. FreqBench/`runpod-b`
files are not mine to touch.

## Gotchas (this box)
- Harness blocks `sleep`; harness-tracked background Bash for long jobs
  (notifies on exit); python -u.
- `git pull --rebase` refuses with unstaged changes — commit STATUS first.
- render_report figs churn PDF timestamps with no content change —
  `git checkout -- figs/` before committing if only binaries moved.
- Concurrent-git race FIXED in grid.py (GIT_OPTIONAL_LOCKS=0, os.replace,
  retry×3) — 495-cell grids run clean at 28 workers.
