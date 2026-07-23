# Working state — agent `runpod`

**Last rewrite:** 2026-07-23, immediately pre-compact. **Next action: read
`briefings/expansion-c6.md` IN FULL (65 lines — only the header was read
pre-compact) and execute it.** It is assigned to me (`for: runpod`,
status: active).

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

## Next task: `briefings/expansion-c6.md` — the calibrated extraction estimator

Known from the header (rest UNREAD — read it first): C6 fixes the
`seg_hier_categorical` EXTRACTION ESTIMATOR (not the model family) for the
reasoning int/eq cell. C5 (`proof-operation-phase-runs-r3`, reviewed +
APPROVED) proved: three-timescale structure CONFIRMED model-independently
(real-vs-permuted ACF(4) gap 0.056); the family closes lag-2–8 (MI(2)
passed first time, ACF(4) +21% marginal overshoot); but the preregistered
insertion control caught the estimator hallucinating +0.018 MI(2) /
+0.039 ACF(4) on run-permuted streams. Limits: ~12 h wall, **$10 API cap**
(NO fresh labeling — reuse committed labels; judgment on claude-fable-5;
spend meter → expansion/results/spend.json), no program-rule/gate edits,
no temp_bench/core edits. Key artifacts: `expansion/mirrors.py`
(seg_hier_categorical + run_permuted_streams + the documented failed
shrinkage campaign — complementary halves / interleaved / analytic floors
/ permutation-matched DP split-half all failed on harness toys, docstrings
say why), `expansion/records/proof-operation-phase-runs-r3/`, LEDGER C5/C6
entries, harness tests `tests/test_expansion_harness.py`
(_seg_hier_truth toy + insertion-control test). Cumulative expansion spend
$10.82/$25 through C5 (C5 itself $0).

## Recently completed (all reviewed or pushed; do not redo)
- **Stage-6 #3b (2026-07-23): recipe_instruction_phase_runs verdict
  POSITIVE** on the re-scoped regime-3 residual axis — first grounded
  regime-3 architecture separation (Spectral-TXC T=2 linearizes the
  residual to +0.97; TXC-post caps at the additive ceiling; additive
  families at the DC-leak line). Freeze A1 `cf4ae797` / A2 `241845d2` /
  A3 `d65349c0`; grid 495/495 clean; briefing reviewed + retired.
  Registry 2 axes, REPORT 66/66, BENCHMARKS text-half-of-prize CLAIMED.
- **C5 (2026-07-22): r3 ABORT** (doubly informative, $0 spend) — reviewed
  + APPROVED; C6 target set.
- **Stage-6 § 8 STOP → re-scope option 1** — reviewed; program rules added
  (gating scripts committed before first run; threshold-optimized
  ceilings).

## Repo state at compact
In sync with origin/arxiv after pull (runpod-b's phasepair bench + FB-C1
landed; suite was 155 passed + 1 skip on the merged tree; `run.py validate`
OK). Working tree clean except this STATUS file. New briefings present:
expansion-c6.md (MINE), freqbench-t16-fbc2.md (runpod-b),
conversion-depth.md (runpod-c).

## Gotchas (this box)
- Harness blocks `sleep`. Background python needs `-u`.
- 5-family models reject `temperature` and think by default (client
  handles); calibrations SEQUENTIAL (spend meter is per-process).
- render_report churns fig-PDF timestamps with no content change —
  `git checkout -- experiments/explorations/synthetic/figs/` if only
  binaries moved.
- Concurrent-git race FIXED in grid.py (GIT_OPTIONAL_LOCKS=0, os.replace,
  retry×3).
- Skeptic verdicts: persisted raw pre-parse, cached, NEVER re-rolled.
