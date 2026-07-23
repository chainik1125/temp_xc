# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-23 ~03:35 UTC (FB-C1 session ~5.3 h in; briefing
`briefings/freqbench-c1.md`, 12 h window ends ~10:15 UTC; stop at the
acceptance gate).

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU. `/workspace/.agent_id`
= runpod-b (I seeded it). Git identity set; push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)` for skeptic.

## FB-C1 state
- **Phase 1 DONE** (94720da7): 132-cell FreqFrac pass; PORT § G.1 (seed
  stability ✓, T=8 high-pass PASS).
- **FB-2 DONE end-to-end** (verdict 309ce8d2): POSITIVE; grid 708/708;
  blind verdict in `multilane/bench_record.md` — sprint band headline
  FAILED its frozen T=8 bar (+0.019 < 0.03, seed-disjoint; edge peaks T=4
  +0.087); spectral collapse at k_pos=8 (−0.583 margin flip); registry row
  + FreqFrac coords done (txc-post conc 0.32→0.84 under superposition).
- **FB-3: build+gates+skeptic DONE; GRID RUNNING** (task b7ptn4n29, log
  scratchpad/colored_grid.log, ~214/582 at last check; results →
  `colored_sources/results/colored_grid_results.json`). Registry row in
  (verdict PENDING — update after the blind verdict). Amendments all
  documented + skeptic-approved (orthonormal null, stream leakage +0.011,
  dilution/window-truncation, one-sided untrained floor).
- **FB-1: card FROZEN + build + gates + skeptic PROCEED 5/5 DONE.**
  Grid NOT launched (CPU) — launch after FB-3 grid:
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GIT_OPTIONAL_LOCKS=0 TQDM_DISABLE=1
  TEMP_BENCH_ALLOW_DIRTY=1 .venv/bin/python -u -m
  experiments.explorations.synthetic.phasepair.run_grid 28` (~700 cells,
  ~80-100 min). Write phasepair/render_figs.py modeled on multilane's
  (keys sign_recovery / pair_recovery / sign_oracle / sign_balacc_pair{p}).
- Spend $1.04/$25 (skeptic ×3). Tests green (159+).

## Next actions (strict order)
1. FB-3 grid completes → `colored_sources.render_figs` (falsifier check
   FIRST: any trained T≤2 or stacked/token cell above the floor band) →
   blind verdict § 4 of `colored_sources/bench_record.md` vs card § 6
   (all-floor NEGATIVE is a pre-registered success; check ρ-quartile
   ordering if any lift) → registry verdict PENDING→actual + FreqFrac
   coords (`freqfrac_report colored_sources --seed 1 --tag
   colored_sources_s1_T4` + `--T 8` variant) → § 5 of record → commit.
2. **Launch FB-1 grid immediately when CPUs free** → FB-3 record work
   meanwhile → quiet-window rebase+push after FB-3 record (pull --rebase
   only when no grid is appending; union drivers handle leaderboard/
   manifest; keep BOTH on data.yaml/synthetic.py conflicts — runpod is
   active on stage-6 #3).
3. FB-1 grid → falsifiers → blind verdict in `phasepair/bench_record.md`
   (headline: dissociation pair≥sign; additive pair>0 sign≈0;
   spectral-edge-shrinks claim) → registry row + FreqFrac coords.
4. REPORT re-render with the new benches; BENCHMARKS.md rows ×3
   (theorem-first; append-only); research STATUS § 0 bullet (append,
   never rewrite runpod's); PORT.md FB-C1 cycle log appended.
5. Acceptance gate: rewrite this file, push everything, **STOP** (briefing
   stays for mac-local review; do NOT delete it).
6. ONLY if hours remain after 1–5: Phase 4 (T=16 frontier addendum +
   freqfrac --T 16) — the acceptance gate outranks it.

## Session gotchas (accumulated)
- One-sided floor checks for eigen/probe artifacts (below-chance ≠ access);
  orthonormal null for eigenbases; all amendments documented +
  skeptic-approved — pattern: fix the CHECK, record the measurement, never
  touch task/tolerance silently.
- Grid pace ≈ 8-9 cells/min at 28 free workers; eval probes cheap (≤2 s).
- `git pull --rebase` only in quiet windows; commit STATUS.md BEFORE
  rebasing (it blocks on unstaged tracked files).
- freqfrac_report needs the bench in registry.py BENCHES first.
- Skeptic: raw persisted pre-parse; Meter at freqbench/results/spend.json.
- BatchTopKSAE.train_step wants (B, d_in); tsae class = TSAEPaper.
