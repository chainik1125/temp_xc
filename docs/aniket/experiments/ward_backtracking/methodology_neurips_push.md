---
author: Aniket Deshpande
date: 2026-05-03
tags:
  - design
  - complete
  - ward-backtracking
---

## TL;DR

Methods companion to [[results_b_neurips_push]]. Documents the changes to
the Stage B pipeline made for the NeurIPS final push: how we cohort,
calibrate, mine, probe, and orchestrate. Aimed at someone trying to
reproduce or extend any single step.

## Pipeline (high level)

```text
Stage A: cache reasoning traces (DeepSeek-R1-Distill-Llama-8B) + label
         sentences for backtracking (Sonnet 4.6 judge)
                       ↓
       cache_activations.py:  resid_L{8,9,10,11,12}.npy via base Llama-3.1-8B hooks
                       ↓
       train_txc.py --cell <arch>__<hp>__k<k>__s<seed>:
         - TXC, TXC-H8, TopK SAE, TSAE-paper, TFA → single-layer loader
         - MLC → multi-layer loader (5 stacked layer caches)
                       ↓
       mine_features.py --cell <cell>:
         - non-MLC: capture T=6 token windows around labeled sentences
         - MLC: capture (5 layers × sentence token) via simultaneous hooks
         - rank top-32 features by mean-difference (D+ vs D−)
         - emit features/<cell>.npz with pos_act, neg_act, decoder dirs
                       ↓
       run_b3_multi_arch.py → b3_variants.py per (cell × headline arch):
         - generate continuations at each magnitude in densified grid
         - evaluate with answers_match() for math correctness
         - emit per-(arch × question × magnitude) row in phase2_rescue.json
                       ↓
       build_flip_matrix.py → flip_matrix.parquet + mcnemar_table.csv
       calibrate_magnitudes.py → calibration.json (p95 of |feature act|)
       plot/headline_steering.py → headline_calibrated.png + raw + appendix
       plot/repetition_rate.py → judge-free repetition vs magnitude
       detection/build_detection_probe.py → detection_headline.png + AUC table
       build_hygiene_table.py → reconstruction_table.csv + training curves
```

## Architecture set (Dmitry's standardization)

| Label | Registry name | Forward / encoder | Notes |
|---|---|---|---|
| TXC | `txc` | Window-shared TopK + per-position decoder | k=16/pos, T=6, window L0=96 |
| TXC-H8 | `txc_h8` | TXC + InfoNCE multi-distance + Matryoshka | Appendix only; FVE=0.50 |
| SAE | `topk_sae` | Per-token TopK | k=64/token, no T axis |
| TSAE-paper | `tsae` (Han attention TSAE w/ k=20) | Predicted + novel codes via attention; signed | Bhalla 2026 paper-aware via `kval_topk=20`. NOT a faithful Bhalla port; see `notes/tsae_paper_param_audit.md` |
| TFA | `tfa` | Same as TSAE-paper but with sinusoidal positional encodings | TFA = TSAE + pos-enc |
| MLC | `mlc` | TemporalCrosscoder where T axis = simultaneous layers {L8..L12} | k * 5 = 160 active features per "window" |

All architectures: `d_in=4096` (Llama-3.1-8B residual), `d_sae=16384`,
seed=42, hookpoint resid.L10.

## Cohort

Stage A → 150 MATH-500 traces. Filter into:

- 78 unsteered-correct (random sample of 30 used for regression cohort)
- 31 truly-wrong (parsed an answer but it was incorrect)
- 41 token-truncated (no parsed answer; dropped)

Per-arch sweep cohort: 31 + 30 = **61 questions × 25 magnitudes = 1525 panels**.
Total across 6 archs: 9150 panels in `flip_matrix.parquet`.

## Magnitude grid

Densified from the original 9-point grid `[-16, -12, -8, -4, 0, 4, 8, 12, 16]`
to:

```yaml
[-16, -12, -10, -8, -7, -6, -5, -4, -3, -2, -1, -0.5,
  0,
  0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
```

Concentrated in ±0.5 to ±8 where the SAE peak lives; sparse in the tails
where curves are flat. 25 magnitudes total.

## Calibration (Andre's idea, Dmitry-approved)

For each (arch, steered feature) pair:

1. Pool the captured `pos_act + neg_act` from the mining npz (these are the
   per-sentence activations of the top-K features).
2. Take `abs()` (TFA / TSAE codes are signed reconstruction residuals).
3. Filter to nonzero values.
4. Take the 95th percentile.
5. Define "calibrated magnitude 1.0" per arch as 1 × that p95.
6. Plot with `x = raw_magnitude / p95` per arch.

The headline plot has both calibrated and raw versions. The
`abs()`-filtering step matters: without it, TFA/TSAE-paper get p95=0
because their codes are mostly slightly negative (the `flat > 0` filter
drops them all). Detailed in `experiments/ward_backtracking_txc/calibrate_magnitudes.py`.

## Flip matrix + McNemar

For each (arch × magnitude × question) row, compute the 2×2 confusion of
correctness from unsteered → steered:

```text
              after-steering
              correct   incorrect
before correct  n_cc     n_ci   ← regression rate panel
       incorr.  n_ic     n_ii   ← rescue rate panel
```

Per arch, find the magnitude that maximizes `n_ic − n_ci` (net rescues).
Report McNemar χ² (Edwards' continuity correction) at that magnitude on
the discordant cells `(n_ic, n_ci)`. With n_total = 61 per arch the
expected normal approximation is borderline; we additionally report the
exact two-sided binomial p-value (`scipy.stats.binomtest`) for robustness.

Implementation: `experiments/ward_backtracking_txc/build_flip_matrix.py`.

## Detection probe (§3)

`experiments/ward_backtracking_txc/detection/build_detection_probe.py`.

For each arch, load the mined npz → top-32 feature activations on labeled
sentences → fit sparse logistic-regression probes for |S| ∈ {1, 2, 4, 8,
16, 32}.

Per-fold feature selection: top-S features by |mean-diff| computed on the
training fold. 5-fold GroupKFold by `question_id` so test sentences are
never from a question that appears in train.

Cross-arch comparison: paired Wilcoxon signed-rank on AUC across folds,
TXC vs each baseline at |S|=8. Holm-Bonferroni correction across the 5
comparisons.

Sentence set: intersection of `sentence_keys_pos` ∪ `sentence_keys_neg`
across all 6 arches. Drops a few hundred sentences where MLC's
multi-layer alignment failed but TXC's single-layer didn't (or vice-versa).
Final n=23,664 with 12% positive class.

## Sweep orchestration

Three bash scripts implement the autofire chain:

- `run_headline_pipeline.sh`: A→E orchestrator for the primary 4-arch
  sweep (TXC, TXC-H8, SAE, TSAE-paper). Step B (sweep) is GPU-parallel:
  pair1 = TXC + SAE on GPU 0, pair2 = TXC-H8 + TSAE-paper on GPU 1.
- `run_tfa_mlc_extension.sh`: F→L orchestrator that caches the 4 extra
  MLC layers (L8/L9/L11/L12), trains TFA + MLC in parallel, mines, runs
  b3 for both arches in parallel, rebuilds the 5-arch headline.
- `/tmp/autofire_pipeline.sh` and `/tmp/autofire_extension.sh`: process
  watchers that fire the next pipeline once the previous one's trains
  exit.

## Mining caveats

### Multi-layer capture for MLC

`mine_features.py:_capture_multilayer_windows` registers a forward hook
on each of `model.model.layers[ln]` for ln ∈ {8, 9, 10, 11, 12}.
Captured per-layer activations are stacked at the sentence's
representative token, producing `(n_sent, 5, d_model)` instead of the
single-layer `(n_sent, T=6, d_model)` from `_capture_windows`. The
"per-offset selectivity" plot is reinterpreted as "per-layer" for MLC.

### Sentence-key alignment

All non-MLC arches produce identical sentence sets (3023 backtracking
sentences + 20641 non-backtracking = 23664 total). MLC's set is slightly
larger on the negative side (3023 + 20922) — likely a small alignment
difference. We take the intersection across all 6 arches for the
detection probe so the cross-arch comparison is on the same sentences.

## Sonnet judge (rubric reference)

The backtracking-event judge is in
`experiments/ward_backtracking_txc/grade_backtracking.py`. Counts GENUINE
backtracking events: catching errors, missing constraints, rejecting an
approach, re-evaluating assumptions. Excludes filler ("Hmm, let me
think"), pseudo-backtracking (same conclusion restated), looping,
gibberish. Sonnet 4.6, ~$0.002/row, resumable.

For mid-sweep cost we used ~18k judge calls × ~$0.002 ≈ $36.

The 20-transcript blind-validation CSV at
`results/.../judge_validation/blind_pairs.csv` is set up for κ comparison
once Aniket scores blind.

## Reconstruction hygiene

Required for NeurIPS submission. `build_hygiene_table.py` compiles
`reconstruction_table.csv` (per-arch FVU, L0, FVE, training steps,
early-stop status) and renders FVU-vs-step + L0-vs-step PNGs for each
arch under `hygiene/training_curves/`. Data source:
`results/ward_backtracking_txc/logs/<cell>__train.jsonl` (one row per
log_interval).

## Repository layout

New artifacts written by this push:

```text
experiments/ward_backtracking_txc/
  NEURIPS_PUSH.md                  # plan + status; supersedes a transient task list
  build_flip_matrix.py             # 2x2 confusion + McNemar
  calibrate_magnitudes.py          # 95th-pctile of |feature act|
  build_hygiene_table.py           # FVU/L0/FVE table + training-curve PNGs
  build_judge_blind_csv.py         # 20-transcript stratified blind sample
  run_b3_multi_arch.py             # per-arch b3_variants dispatcher
  run_headline_pipeline.sh         # primary 4-arch orchestrator
  run_tfa_mlc_extension.sh         # extension for TFA + MLC
  detection/
    __init__.py
    build_detection_probe.py       # sparse probes + Wilcoxon
  plot/
    headline_steering.py           # 3-panel calibrated + raw + appendix
    repetition_rate.py             # judge-free auxiliary

notes/
  backtracking_appendix_draft.md   # main-vs-appendix manifest + prose drafts
  tsae_paper_param_audit.md        # Bhalla 2026 vs codebase architecture audit
  thought_anchors_taxonomy.md      # Bogdan & Macar 2026 reasoning taxonomy

results/ward_backtracking_txc/b3_math500_cut25/
  headline_calibrated.png          # MAIN TEXT Fig 4a
  headline_raw.png                 # appendix
  appendix_calibrated.png          # 6-line variant w/ TXC-H8
  appendix_raw.png
  repetition_rate.png              # judge-free auxiliary
  repetition_rate_headline.png     # 5-line variant
  flip_matrix.parquet              # 9150 long-form rows
  mcnemar_table.csv
  calibration.json                 # 6 arches; p95 per (cell, feature)
  <arch>__f<id>_<mode>/            # per-arch sweep dirs
    meta.json
    summary.json                   # rescue + regression rates by mag
    phase2_rescue.json             # gitignored (heavy; regenerable)

results/ward_backtracking_txc/detection/
  detection_headline.png           # MAIN TEXT Fig 4b (5 lines)
  detection_appendix.png           # 6 lines incl. TXC-H8
  probe_results.parquet            # (arch, S, fold, auc, f1)
  summary_auc_f1.csv
  wilcoxon_detection_table.csv

results/ward_backtracking_txc/hygiene/
  reconstruction_table.csv         # MAIN TEXT Tab 4a
  training_curves/<arch>.png       # per-arch FVU + L0 vs step

results/ward_backtracking_txc/judge_validation/
  blind_pairs.csv                  # 20 transcripts; awaits human scoring
```

## See also

- [[results_b_neurips_push]] — what we found
- [[results_b]] — prior 4-arch + H8/H13 Stage B run
- [[plan]] — Stage B original plan
