# Working state — agent `runpod`

**Last rewrite:** 2026-07-22, late in the 12 h overnight session (Phases 1+2
DONE and pushed; Phase 3 bookkeeping in progress).

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am `runpod`
(original box — `/workspace/.agent_id` does NOT exist here; do not create it).**
`runpod-b` runs FreqBench (`freqbench-c1.md`) in parallel — NOT mine.
Two-agent rules (agents/README.md): `git pull --rebase origin arxiv` before
EVERY push; shared files append-only; leaderboard/manifest union-merge.
Tokens `/workspace/.tokens/`; push
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
32 CPU / 128 GB. Venv fine.

## Current task: `briefings/stage6-recipe-then-c5.md` — Phases 1+2 DONE, stopped-for-review state being finalized

### Phase 1 — stage-6 of recipe_instruction_phase_runs: **DONE — § 8 STOP, no grid**
Build commits `47b62e1b` (generator/datasource/evaluator/tests; suite 136) +
`b463c4a0` (equality-variant STOP-gate fired: raw-linear e_t ceilings 0.614
per-token / 0.720 T=2 ≫ chance — class-conditional continuation-rate leak;
MLP 1.0 so the regime-3 residual is real). Grid withheld per the gate;
frozen § 5 predictions never touched — still blind. Record + re-scope options
in `recipe_instruction_phase_runs/bench_record.md`; BENCHMARKS row ✓+note;
research STATUS bullet. All pushed.

### Phase 2 — expansion C5: **DONE — r3 ABORT, doubly informative, $0 API spend**
- Infra commit `f8c1deb6` (BEFORE calibration, freeze discipline):
  `seg_hier_categorical` in expansion/mirrors.py (three timescales: dwell /
  run-aware BIC-DP segments + per-segment C4 deconvolution / doc tilt; MLE
  tilts; likelihood-only objectives) + **insertion control**
  (`run_permuted_streams` no-adjacent-repeat shuffle; control = re-fit on
  permuted must not hallucinate gate-8 moments beyond real tolerance).
  A measured campaign of automatic shrinkage estimators (complementary
  halves / interleaved splits / analytic floors / permutation-matched DP
  split-half / posterior-mean) all failed on the harness toys — documented
  in mirrors.py docstrings; harness tests pin both control behaviors.
  138 tests pass. r3 card amendment appended (prereg incl. the control).
- Calibration commits `2d41d5cd` + `b7025cfc`:
  **proof-operation-phase-runs-r3 ABORT** — MI(2) PASSES first time
  (0.075 vs 0.065, err 0.010 ≤ 0.013); ACF(4) +21% marginal overshoot
  (0.154 vs 0.127, err 0.0263 vs 0.0255); insertion control FAILS both
  (hallucination +0.018 mi2 / +0.039 acf4 on permuted streams). Three
  timescales CONFIRMED model-independently (real−perm acf4 gap 0.056).
  **C6 gap = calibrated extraction estimator, not model family.**
  Skeptic-header cosmetic done (`_judge_model`; pre-C5 marked untracked).
  Spend $10.82/$25 cumulative (C5 $0.00 — labels cached, skeptic skipped).

### Phase 3 — bookkeeping (in progress at last rewrite)
REPORT/registry render check; self-audit of the Phase-1 record vs README
checklist + validity gates (list gaps honestly, do NOT re-run). Then final
STATUS rewrite → STOP. **Briefing stays until mac-local review.**

## Next actions
Finish Phase-3 render check + self-audit → commit + pull --rebase + push →
STOP for review. No stage-6 § 8/grid for any C5 artifact; do not start C6.

## Gotchas (this box)
- Harness blocks `sleep`; background Bash for long jobs; python -u.
- 5-family models reject `temperature`, think by default (client handles).
- Concurrent-git race FIXED in grid.py (GIT_OPTIONAL_LOCKS=0, os.replace,
  retry×3).
- `git pull --rebase` refuses with unstaged changes — commit this file first.
