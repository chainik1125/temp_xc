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

**Findings.** <!-- FILL: 2–4 findings, each with its graph -->

1. <!-- FILL: d(perf)/dT verdict — did the btk-only arm's T-slope improve
   (soften downward / steepen upward) relative to paper-match? Quote slopes
   per bench and metric. -->
   - ![d(perf)/dT gate figure](plots/btk_rerun/btk_rerun_dperf_dT.png)
2. <!-- FILL: level effect — btk-only vs paper-match at matched (T, k_pos);
   which cells move most (pre-registered: low-T cells recover; the T=1
   controlled limit). -->
3. <!-- FILL: mixing fingerprint — realized l0 vs nominal per arm
   (paper-match loses support as T grows; btk-only holds it by construction);
   neg_frac provenance. -->
   - ![fingerprint](plots/btk_rerun/btk_rerun_fingerprint.png)
4. <!-- FILL: baseline overlay — does either arm beat the frozen tsae
   reference (coupled gauc 0.81 @ k=1; markov eauc 0.86 @ k=5)? Honest
   caveat: baselines had 10k steps vs our 6k, older protocol (gauc/eauc are
   decoder-direction metrics, stated protocol-stable). -->

**Verdict for the re-run gate.** <!-- FILL: per actmix-shared.md this must
quote the pre-registered expectations: "the PAPER arch's T-curves should
improve (that is Dmitry's re-run gate: does d(perf)/dT improve)". State
SUPPORTED / NOT SUPPORTED / MIXED with numbers. All verdicts PENDING TEAM
REVIEW per house discipline. -->

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
