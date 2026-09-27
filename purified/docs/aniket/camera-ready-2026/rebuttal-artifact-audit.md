# Rebuttal artifact audit — 2026-09-27

This audit separates numbers available for immediate figure rebuilding from the inputs needed to rerun an experiment. No datasets, activations, checkpoints, or training runs were downloaded or launched by this audit. Paths below are relative to the repository root unless a Git reference is explicitly given. The root task is preserving selected branch-only files in `purified/artifacts/camera_ready_2026/`; its manifest is authoritative for the final recovered file list.

## Immediate conclusions

- The corrected 300K Backtracking detection replication, both Aniket window sweeps, and writing-revision supplementary results already have tracked compact numeric artifacts and plotting code. These figures can be rebuilt without training or loading large activations.
- The posted Backtracking seed figures must be reconciled before camera-ready tables are updated: the old 20K, T=6, S=32 results round to 0.27/0.27/0.26, while corrected 300K TXC-base detection is 0.1874 ± 0.0080 at S=8 and 0.2568 ± 0.0048 at S=32. They are different protocols and cannot share a seed-variance row.
- The final Shamir W=10 result and its stated episode-disjoint implementation have not been found in the inspected Git refs or the small reviewer HF snapshot. The older committed polynomial-clock implementation and W≤5 results are available, but are not substitutes for the posted 0.96 result.
- Medical steering seeds 1/2/42 are complete at the aggregate level (16.71875/20.359375/22.875). The HF snapshot contains seed-2 stage aggregates and its complete canonical/extended frontier; the seed-1/42 dense extensions still point to external `local_data` files.
- The archived HH-RLHF `agentic_txc_02` three-seed metrics are TXC-pro under the paper's own architecture definitions. Relabeling that evidence as TXC-base would be an error.

## Tracked compact packages already on neurips-aniket

Counts exclude unrelated work. Sizes are decimal bytes and reflect the audit snapshot before recovery packaging.

| Package | Code | Tracked results | Cheap figure input | Expensive input needed only for rerun |
|---|---|---:|---|---|
| Corrected 300K C7 detection | `purified/experiments/backtracking_300k_seeded/` (7 files, 42,464 B) | `purified/results/neurips_rebuttal/backtracking_300k_seeded/` (37 files, 536,877 B) | Four `cells/*/detection.json`, `source_artifact.json`, publication CSVs | Historical activation-training cache, four checkpoints, sentence activation artifact, GPU training/evaluation |
| Original 20K T=1…6 detection/order sweep | `purified/experiments/backtracking_window_sweep/` (21 tracked files, 378,449 B; shared with wide sweep) | `purified/results/neurips_rebuttal/backtracking_window_sweep/full/` (28 files, 2,644,822 B) | 18 `cells/T*_seed*/result.json` plus `publication/*.csv` | Activation cache, all 18 dictionary pairs, sentence activations |
| Wide 20K T={1,2,4,6,10} sweep | Same code, specifically `protocol_t16.py`, `run_t16.py`, `plot_publication.py` | `purified/results/neurips_rebuttal/backtracking_window_sweep_t16/reviewer-five-point-v1/` (26 files, 2,396,795 B) | 15 cell JSONs, grid manifest, publication CSVs | Wider event-aligned activations, checkpoints, GPU training/evaluation |
| Writing-revision destination | `purified/experiments/writing_revision_destination/` (11 files, 200,655 B) | `purified/results/neurips_rebuttal/writing_revision_destination/` (17 files, 1,354,015 B) | Two raw aggregate JSONs, publication CSVs, `frozen_dictionary_t5_v1/result.json` | KLiCKe archive/cohort, roughly 0.51 GB activation cache and 3.76 GB dictionary checkpoints |

The original T=1…6 directory also contains about 57.75 MB of untracked local files, principally held-out prediction NPZs, plus small `summary.json` and `summary.md` files and duplicate figure exports. The prediction files are useful for new bootstrap/calibration analyses, but not necessary to remake the existing plots; don't add them blindly. Untracked `__pycache__` files in code directories are expendable build products.

## Backtracking: protocol and number reconciliation

### Corrected 300K dictionary replication

The source is locked by `purified/docs/aniket/neurips-rebuttal/july29-backtracking-closeout.md` and `purified/experiments/backtracking_300k_seeded/README.md`:

- Historical implementation commit: `284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3` (`origin/extended-300k`).
- Training protocol: `c7-300k-seeded-v1`; evaluation protocol: `c7-detection-seeded-v1`.
- Llama-3.1-8B, layer 10, T=5, d=32,768, k_pos=20, batch=1,024, 300,000 updates; corrected TXC-base seeds 1, 2, 42.
- Cache key `fb2a74be884e512a`, shape `(4044,128,4096)`, float16; cache SHA-256 `dc34dfb117f77abddef4b4396d0d00afc707c39876d0ee36015de1e7b8406914`.
- Detection artifact `sentence_acts_L10.npz`, 1,137,333,114 bytes, shape `(25204,6,4096)`, 3,169 positives, 300 question groups; SHA-256 `1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810`.
- Train keys: seed 1 `a300c63374c3597e`, seed 2 `27078b0d7700ae05`, seed 42 `8787f8fe527218ad`.

| S | Corrected TXC-base mean ± sample SD | T-SAE 16K seed 42 | Submitted T-SAE 32K seed 42 |
|---:|---:|---:|---:|
| 8 | 0.187411 ± 0.007970 | 0.204258 | 0.196 (rounded paper reference) |
| 16 | 0.231444 ± 0.007244 | 0.212982 | 0.213 (rounded paper reference) |
| 32 | 0.256756 ± 0.004782 | 0.228504 | 0.245 (rounded paper reference) |

The corrected S=8 seed values are 0.195031, 0.179131, and 0.188072. TXC-base is below T-SAE at S=8 but above it at S≥16. Present the full probe-budget curve, not a budget-independent ranking. The historical runner did not seed Python/PyTorch before initialization and sampling; its evaluator silently defaulted to seed 42. Corrected values are a new experiment, not a deterministic continuation of the submitted row.

The submitted detection winner was TXC-pro; the submitted steering winner was TXC-base. This package reruns TXC-base detection only. Corrected TXC-pro detection and multi-seed 300K Backtracking steering are still missing.

`plot_detection.py` reads only compact JSON and validates protocol, keys, cohort and source hashes before drawing. Run from the repository root with `--root purified/results/neurips_rebuttal/backtracking_300k_seeded --tsae16-json purified/results/neurips_rebuttal/backtracking_300k_seeded/cells/tsae_paper_d16384_seed42/detection.json --output-dir <new-output-directory>`.

### Old and wide 20K window sweeps

The older HF snapshot's T=6/S=32 TXC seed values are 0.265800, 0.267651, 0.255365, which round to the 0.27/0.27/0.26 triplet. Its T=5 values are 0.264962, 0.265271, 0.260729. Matching a posted rounded table to the former is an inference; preserve the table's stated labels and confirm the exact source before changing any text.

The original local sweep uses only the six available offsets −13…−8 before the labeled sentence. The later directory is named `backtracking_window_sweep_t16`, but the publication grid is T={1,2,4,6,10}, with 15 complete cells and 20K steps, not a T=16 result. Its tracked publication summary reports:

| T | Ordered TXC AP | Fixed-probe shuffled TXC AP | Last-token SAE AP | Invariant SAE AP |
|---:|---:|---:|---:|---:|
| 1 | 0.218 ± 0.005 | 0.218 ± 0.005 | 0.221 ± 0.016 | 0.221 ± 0.016 |
| 2 | 0.229 ± 0.006 | 0.223 ± 0.006 | 0.208 ± 0.004 | 0.211 ± 0.008 |
| 4 | 0.247 ± 0.007 | 0.227 ± 0.006 | 0.210 ± 0.016 | 0.219 ± 0.006 |
| 6 | 0.251 ± 0.006 | 0.227 ± 0.004 | 0.207 ± 0.012 | 0.220 ± 0.007 |
| 10 | 0.255 ± 0.008 | 0.231 ± 0.009 | 0.214 ± 0.007 | 0.223 ± 0.007 |

These are mean ± sample SD across dictionary seeds 1/2/42 at S=32. Shuffle/reversal/circular-shift controls use an ordered-trained fixed probe under covariate shift. They measure representation sensitivity, not a causal decomposition of uniquely temporal information.

The post-hoc T=6 seed-42 positional-SAE sensitivity is particularly important: AP grows from 0.1399 at S=32 to 0.2779 at S=256, overtaking TXC at its fixed S=32 AP of 0.2585. The matched sparse-feature budget is scientifically relevant, but architecture superiority is budget-dependent. Existing paired question-bootstrap intervals and the CSV can be plotted immediately.

`plot_publication.py <root> <output_dir> --windows 1,2,4,6,10 --seeds 1,2,42` is the renderer; these arguments were verified in the current source. It needs result JSONs, not checkpoints. Keep the 300K contextual baseline markers explicitly separate from the 20K curves.

## Shamir / polynomial-clock recovery gap

`origin/dmitry-txcwins-10h@090bb6408` contains `src/v6_colored_sources/`, `tests/test_polynomial_clock.py`, and `results/v6_colored_sources/polynomial_clock_h{1_q31,2_q11,3_q7}.json`. The h=2 file is 19,856 bytes with W_grid=[1,2,3,4,5], H=2,048, 6,000 training steps, batch=128, n_seq=4,096, q=11, d=64, noise=0.1. It is also present unchanged in the inspected dmitry-synthetic, dmitry-em-repl, dmitry-spectral-sprint2, and diffusion-txc refs.

This old runner samples multiple random windows from common episodes, then `_train_logistic_probe` randomly splits rows. It does not implement the later rebuttal draft's claim of episode-disjoint representation training/probe training/validation with one evaluation window per episode. Its W≤5 metrics also do not reproduce the posted W=10 0.96 recovery point. Preserve it as historical code and data; do not describe it as the final result's exact reproduction.

A bounded grep of all `src`, `tests`, and reviewer-response files in the Dmitry branch found no alternate final W=10/episode-disjoint runner. `docs/dmitry/reviewer_responses/window_length_theory.md` independently documents the same overlap problem at lines 155–170 and calls the older higher-degree results descriptive pending an episode-disjoint rerun.

For an import-complete historical source snapshot, include `src/v6_colored_sources/`, `temporal_crosscoders/models.py`, `temporal_crosscoders/han_arch/`, `temporal_crosscoders/han_tsae/`, `src/bench/architectures/_tfa_module.py`, and the relevant package `__init__.py` files. `han_tsae` is a package, not one Python file. Package initializers may import additional architecture modules, so retaining the small `src/bench/architectures/` tree is safer than copying one leaf. External dependencies include PyTorch; `sparsemax` is only needed for the lazily imported SAEStandard branch.

The theorem establishes that W≤h carries no information about the secret, while W≥h+1 permits recovery. It does not prove TXC is the only possible architecture above threshold. A final figure should pair the chance bound with matched windowed controls and keep optimized recovery distinct from representational feasibility. Replotting a final supplied JSON would be cheap; reproducing the posted full sweep should wait for its exact code/config/result snapshot. The old 475-second elapsed record is not a valid estimate for the missing final sweep.

## Capacity, T-SAE widths, and costs

`purified/docs/aniket/neurips-rebuttal/tsae-capacity-audit.md` pins submitted C7 T-SAE width 32,768 to training key `32f27809cdf34da9` and S32 PR-AUC 0.24481534796544918. Width 16,384 produces key `b97e3c00153a5271` and is a new sensitivity experiment. Current `purified/configs/archs.yaml` has a generic 16K T-SAE default, so a naive rerun can differ from the submission; retain `origin/extended-300k:purified/configs/locked_archs.yaml` with the C7 override.

`origin/dmitry-txcwins-10h:docs/dmitry/reviewer_responses/parameter_flops_draft.md` gives the full per-task analytical table. It is sufficient to cheaply recreate a parameter/compute figure, subject to checking its registered widths and the native-forward convention. At C7 d_in=4,096 and d_dict=32,768:

- SAE/T-SAE has 268,472,320 parameters and 0.537 GFLOPs per token.
- T=5 TXC-base has 1,342,230,528 parameters and 2.684 GFLOPs per five-token window.
- Stacked SAE has almost the same leading storage and dense matmul cost as TXC; biases differ slightly.
- Applying an SAE to five tokens has the same leading encoder-plus-decoder multiply-add count as a TXC five-token forward. Sliding TXC overlap changes stream cost; report the stride and support.

These FLOPs are formulas, not latency measurements: one multiply-add is two FLOPs, with TopK/nonlinearities/selection and training-only losses excluded. Equal dictionary width does not mean equal parameter count. The draft also flags sparse-probing T-SAE checkpoints at 16,384 versus the appendix's blanket 18,432 assertion, and distinguishes medical 16K versus 32K results. These need manuscript table corrections before visual polishing.

## Multi-seed C3 / C6 / C8 and Stacked SAE

The small recovered HF snapshot is pinned to revision `3e935fd2fa5feff053da90517907011687bdcb4b` of `dmanningcoe/temp-xc-reviewer-results`; mirror base is `purified/artifacts/camera_ready_2026/hf_reviewer_results/reviewer_seed_audit_2026-07-27/`.

- **C3 sparse probing:** Reviewer drafts quote TXC-base 0.90/0.90/0.90. `origin/final:purified/experiments/c3_probing/results.json` is the historical figure input, while `origin/arxiv:results/leaderboard.jsonl` contains the newer probing grid. Preserve all three seed rows with their probe-budget and aggregation definition before treating rounded agreement as seed robustness. A 20-feature endpoint and an AUC integrated over a feature-budget sweep are different metrics.
- **C6 medical steering:** `medical_em/steering_three_seed_summary.json` records 16.71875, 20.359375, 22.875 for seeds 1/2/42 at coherence≥70 across canonical plus dense-extension alpha cells. Seed 2 has `seed2_steering/{result,stage1_ranking,stage2_screen,stage3_strength,stage4_frontier,wang_full,wang_full_extended}.json`, enabling aggregate frontier redraws. Seed-1/42 canonical frontier source paths are `origin/final:purified/results/runs/c6_{2016074933c41e7f,88a4ddf6819d8057}/stage4_frontier.json`; their dense extensions remain external `local_data/c6_redteam/h100_em_4/sweep_outputs/.../wang_full_extended.json` references. The summary supports the extrema/17-20-23 row; it cannot reconstruct the missing complete extension curves or per-generation uncertainty.
- **C6 detection/windows:** Published seed 1 and 42 detection JSONs and the newer seed-2 exact-T5 run/evaluation are in the HF mirror. Exact paper v1 window-sweep source is `origin/codex/em-paper-window-sweep-s42-20260727:experiments/c6_em/window_sweep.py`, with `src/temp_bench/`, branch configs, and `tests/test_c6_window_sweep.py`. It changes T only from the 25K-step, batch-1024 seed-42 recipe. The archived source alone does not establish that every final T result has been recovered. Existing medical shuffle scores can exceed ordered scores, so this task is not evidence of universal temporal-order benefit.
- **C8 HH-RLHF:** `rlhf/summary.json` and per-seed JSONs preserve ordered/shuffled AUCs 0.622901/0.619592, 0.605258/0.604141, 0.609647/0.597529. Metadata explicitly identifies `MatryoshkaTXCDRContrastiveMultiscale` (`agentic_txc_02`), hence TXC-pro. Seed-1/2 payloads are headline metrics recovered from completed run logs; full payloads were retained on a stopped RunPod volume. Aggregate plots are cheap, while new intervals/per-example feature analysis require those missing payloads or reevaluation. Preference AUC is also a different endpoint from the paper's semantic versus length-spurious top-20 feature counts.
- **Stacked SAE:** Preserve reviewer-response source and branch-specific metrics from `origin/dmitry-stacked-c7-300k` and `origin/dmitry-stacked-em-steer` rather than reconstructing raw values from the prose. The draft C7 steering point is about 0.25 at magnitude −12; the C7 detector must be compared at the same S as the TXC row. The draft's Medical EM stacked AP≈0.65 is not sparsity-calibrated: realized L0 is about 32× nominal versus 6–10× in references under train-to-rollout threshold shift. That row should carry its caveat or be recalibrated before a strong capacity claim is made.

## Late Stacked-SAE recovery finding

A follow-up bounded branch-log search found the missing source on **`origin/dmitry-stacked-arxiv@bb8cf85ca`**, in `docs/dmitry/sprints/2026-07-27_stacked_sae_10h/{summary.md,log.md}`. Its exact compact outputs were uploaded to the **private HF repository `dmanningcoe/stacked-sae-rebuttal-2026-07`**, under per-pod `*/new_leaderboard_rows.jsonl`; the same repository holds manifests and clean/poisoned C7 judge passes. This supersedes treating the final Stacked numbers as anonymous prose-only values, but raw result recovery is complete only once those small HF files have actually been fetched and hash-recorded.

| Final Stacked result | Exact identity/value from sprint source | Provenance caveat |
|---|---|---|
| C7 300K | Train key `26e69fdc60452c27`, seed42; Δgc=0.246 at magnitude −12; AP S1/S2/S4/S8/S16/S32≈0.152/0.152/0.150/0.158/0.179/0.207 | Clean rejudge after first pass failed with −1 sentinels; use clean 1,525-call Sonnet-4.6 pass |
| C3 | `stacked_sae_pooled`, protocol1.2.0, S=20; mean AUC0.8694, untrained0.8026 | Match the 20-feature endpoint, not integrated sparse-probe AUC |
| C6 detection | Train key `8b8231508a1ce6e3`, T5 seed42; AP S16=0.6516, untrained0.3442 | Realized sparsity substantially above nominal; initial T4=0.512 was off-target and retained separately |
| C8 HH-RLHF | Train key `ae17686fd3a23df2`, T5 seed42 k_win500; AUC0.602, untrained0.6174; 1/20 length-spurious features | Realized L0≈533; initial k_win80 run is a different erroneous configuration |
| C6 steering, completed follow-up | Same weights as C6 detection; headline-finalist Δalignment at coherence≥70: Stacked13.3 versus TXC17.1 | Ranking reverses under the pipeline's peak-alignment summary; obtain stage aggregates and declare endpoint explicitly |

The two Stacked-specific branch tips originally inspected add source wiring only. Their committed historical leaderboards contain a **20K** `d08c6498d3fa430e` C7 row (peak0.327869 at +12; S8 AP0.177245), which is not the final 300K rebuttal row. Preserve both with explicit training-scale labels. For runnable final C3/C6/C8 source, preserve the `dmitry-stacked-arxiv` branch's `src/temp_bench/archs/stacked_pooled.py`, configs and `tests/test_stacked_pooled.py` in addition to the common source archive. C7 uses `dmitry-stacked-c7-300k`; C6 steering uses `dmitry-stacked-em-steer`.

## Optional writing-revision addition

The frozen T=5 dictionary package covers 6,224 events from 2,510 writers, one dictionary pair. At S=32 it reports TXC log loss 1.2356, strongest SAE 1.2613, paired gap +0.0257 [0.0100,0.0413]; fixed shuffle and reverse score 1.6461 and 1.6569. The tracked JSON contains aggregate curves and intervals, and `frozen_dictionary.py::_render_plot` draws from that result alone. Raw activation figures use the two `raw_*_singleton_v1.json` inputs through `report.py`.

This is a usable optional supplemental demonstration, not posted rebuttal evidence unless the posted discussion explicitly includes it. Its intervals bootstrap writers, not dictionary seeds. The SAE wins at S=16 and the S=128 gap interval crosses zero, so use the registered S=32 comparison plus the full budget curve. New fits require the KLiCKe cohort/cache/checkpoints; replotting does not.

## Preservation and camera-ready priorities

1. Preserve compact result JSON/CSV, exact historical configs, source snapshots, existing plotting scripts, and source hashes first. Keep raw datasets and model weights outside Git. Check whether the upstream snapshot is a summary or per-example artifact in the manifest.
2. Resolve Shamir final provenance and Backtracking protocol/budget mismatches before changing manuscript numbers. Preserve the posted table separately from recovered raw values so discrepancies remain visible.
3. Make an experiment identity table with architecture family, width, T, seed, train steps, actual L0, detector budget, cohort, metric, and source path. This addresses the main reviewer complaints and prevents figures from mixing legacy and corrected runs.
4. Remake the window-length, ordered/shuffled, probe-budget, and capacity figures directly from the small files above, using one palette and consistent uncertainty semantics. Show both detection and steering identities and retain negative/control results.
5. Add architecture/window/threshold diagrams as vector assets after the narrative is settled; they need no experimental reruns. Keep large-data recreation as an explicit separate task only for the unresolved raw artifacts or genuinely new scientific analyses.
