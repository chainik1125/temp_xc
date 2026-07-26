# Does the paper TXC's window-size scaling improve under a pure-BatchTopK activation?

*10h sprint, 2026-07-26 19:38 UTC → ~05:38 UTC. Branch `dmitry-btk-txc-sprint`
(worktree-isolated), based on `origin/arxiv` @ `5e6bfe37`. Author: Claude
(Dmitry's isolated re-run lane of the ACTMIX phase).*

## Executive summary

**Question.** The paper's temporal crosscoder (`txc_base`) applies TopK
selection and then a ReLU on the selected values (`txc_base.py:166-168`), so
window slots that were *selected* but negative are zeroed after the fact. The
team's ACTMIX briefing records that this harm grows with window size T,
biasing the paper's performance-vs-T story downward. This sprint retrained the
paper arch with the composition as the ONLY changed variable — arm
`paper-match` (TopK→ReLU, as published) vs arm `btk-only` (BatchTopK over raw
pre-activations, no ReLU anywhere in the sparsity path) — across
T ∈ {1, 2, 4, 5, 8, 10} on the paper's two §4 synthetic benches, under one
shared parameter set, with per-token baselines frozen at their existing
leaderboard rows.

**Design.** The btk-only arm changes TWO things at once relative to the
paper composite (per-window→batch-pooled selection AND ReLU removal), so a
third arm — `relu-mix` (ReLU then batch-pooled BatchTopK at the paper
budget, the v2-family composition applied to the paper arch) — completes
the 2×2 and attributes any level difference to prior-loss vs pooling.

**Findings.** <!-- FILL: 2–4 findings, each with its graph -->

1. <!-- FILL: gate verdict — coupled gauc slope: composite −0.006 vs
   btk-only +0.027 (pooled non-clipped cells); composite's own k=2 curve
   collapses 0.988→0.829 from T=1→8 (the harm-grows-with-T mechanism,
   visible directly). Final numbers after 3-arm data lands. -->
   - ![d(perf)/dT gate figure](plots/btk_rerun/btk_rerun_dperf_dT.png)
2. <!-- FILL: the level surprise — composite dominates btk-only in absolute
   gauc at nearly all (k, T) (k2/T1: 0.988 vs 0.624); btk-only reconstructs
   far better (k1/T1 nmse 0.003 vs 0.124) while recovering features worse.
   Read: the composite's ReLU acts as a nonnegativity prior that keeps
   decoder atoms single-feature; deleting it trades feature recovery for
   reconstruction. relu-mix arm attribution goes here. -->
3. <!-- FILL: fingerprint — btk-only realizes its full budget everywhere
   (l0/win ≈ k_win, no zero-picks); composite bleeds support even at T=1
   (k2/T1 keeps 0.87 of 2 slots); relu-mix zero-pick numbers. -->
   - ![fingerprint](plots/btk_rerun/btk_rerun_fingerprint.png)
4. <!-- FILL: baselines — composite clears the frozen tsae bar (coupled k1
   gauc 0.809) at every T; btk-only clears it only for T ≤ 5. Markov: both
   arms beat tsae refs for T ≥ 2 until the clipped crash. -->

**Verdict for the re-run gate.** <!-- FILL: quote the pre-registration from
actmix-shared.md ("the PAPER arch's T-curves should improve (that is
Dmitry's re-run gate: does d(perf)/dT improve)"). Expected shape: MIXED —
the slope improves on the headline global-recovery metric and the
composite's high-T degradation is confirmed as an activation artifact, but
btk-only is NOT a better paper arch at these budgets: its levels are
dominated by the composite at almost every (k, T ≤ 10). State plainly; all
verdicts PENDING TEAM REVIEW. -->

## What was run (map)

- **Arms.** `txc_base` (paper composition; black) vs `txc_base_btk` (new,
  this sprint; blue). `txc_base_btk` is a plugin twin — same parameterisation,
  init, AuxK dead-revival, decoder unit-norm, grad-parallel removal; the only
  change is the sparsity path: BatchTopK over raw squashed pre-acts at the
  paper's own budget k_win = min(k_pos·T, d_sae), JumpReLU-threshold eval
  path, signed codes allowed, negative-pick fraction logged
  (`src/temp_bench/archs/txc_base_btk.py`, registered in `configs/archs.yaml`,
  8 contract tests in `tests/test_txc_base_btk.py`).
- **Benches.** §4 synthetic: `toy_markov_n20_d40_noisy` (Denoising) and
  `toy_coupled_K10_M20_d256` (Coupling), exactly the canonical paper
  datasources.
- **Grid.** T ∈ {1,2,4,5,8,10} × k_pos ∈ {1,2,5} × seeds {1,2,3} × 2 arms ×
  2 benches = 216 cells. k_pos ∈ {10,20} and T=20 were dropped because
  k_pos·T ≥ d_sae=20 clips BOTH arms to a dense code (degenerate for a
  selection-rule comparison); remaining clipped cells are greyed in figures
  and excluded from slopes.
- **One shared parameter set.** n_steps 6 000, batch 1024, buffer 2 M tokens,
  lr 3e-4 (schema default), bf16; eval_window_L = 40 (tiles every T above) —
  one uniform eval protocol for every cell in the comparison.
- **Compute.** Modal, app `dmitry-btk-rerun`, A10G ×≤8, detached, freeze
  `2ce33ac5` cloned at pin (`_assert_pinned`), shard results persisted to
  Volume `temp-xc-btk-rerun` and merged locally with dup-key checks. Spend
  ledgered in `briefings/MODAL_SPEND.md` (~$20 este; actuals corrected there).
- **Code paths.** Driver `experiments/explorations/btk_rerun/driver.py`
  (shard = arch × bench × T); analysis
  `experiments/explorations/btk_rerun/analysis.py`; launcher
  `scripts/modal_btk_rerun.py`. Rows land in `results/leaderboard.jsonl`
  with agent tag `dmitry-btk-sprint`.

## Context: how this fits the ACTMIX program

This sprint is the isolated "Dmitry re-run" lane pre-registered in
`briefings/actmix-shared.md`. Complementary lanes (not duplicated here):
runpod-1 = sparse-probing shuffle + T-sweep, runpod-2 = EM shuffle + T-sweep
(both btk-only, T ∈ {1,2,4,8,16}), backtracking = Aniket only, mac-a = hunt
substrate calibration + KEEP survival, mac-c = COMPOSITION_AUDIT (paper-match
pins). mac-a's canonical btk-only convention note had not landed at my pin;
this implementation follows the briefing's recommended shape (selection over
raw pre-acts, threshold gating unchanged at eval, negative-selection count
logged) and is documented as a convention candidate, not the canonical one.

## Caveats (read before quoting numbers)

- <!-- FILL/KEEP: --> Toy scale only: d_sae=20 dictionaries on synthetic
  benches. The real-LM sections (probing, EM, RLHF, backtracking) are other
  lanes' work; nothing here licenses claims about them.
- Baselines were NOT rerun (user directive). The frozen tsae reference rows
  used 10k steps (vs our 6k) and protocol 1.1.0 (gauc/eauc are
  decoder-direction metrics documented as unchanged across protocol bumps;
  NMSE is not comparable and is never overlaid). The ACTMIX briefing
  pre-registers that per-token baselines improve MOST under btk-only — so a
  frozen-baseline comparison is biased *toward* TXC and is labeled as such.
- n_steps 6 000 is sprint-scale (the paper's synthetic rows used 10 000; its
  backtracking mains 300 000). Slopes could shift at full training length.
- eval_window_L=40 differs from the legacy L=5 and hunt L=32 rows; NMSE/l0
  comparisons stay inside this sweep.
- Single param set, no tuning pass. <!-- FILL if a tweak round happened. -->

## Reproduction

```bash
git checkout dmitry-btk-txc-sprint
python -m pytest tests/test_txc_base_btk.py           # contract tests
modal run --detach scripts/modal_btk_rerun.py          # 24 shards
python -m experiments.explorations.btk_rerun.analysis  # figures + slopes
```

## Research log

See `sprint_log.md` for the full timeline including the disk-full incident,
the arxiv-branch discovery, and all freeze SHAs.
