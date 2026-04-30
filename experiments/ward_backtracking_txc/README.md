# Stage B — Base-only TXC steering for backtracking

Pipeline (run from repo root, in order):

```bash
# Phase 1: cache base Llama-3.1-8B activations at 3 hookpoints (FineWeb corpus)
python -m experiments.ward_backtracking_txc.cache_activations

# Phase 2: train one TemporalCrosscoder per hookpoint
python -m experiments.ward_backtracking_txc.train_txc

# Phase 3: encode Stage A traces through trained TXC, rank features by D+/D- selectivity
python -m experiments.ward_backtracking_txc.mine_features

# Phase 4: B1 — single-feature steering eval (decoder rows used as steering vectors)
python -m experiments.ward_backtracking_txc.b1_steer_eval

# Phase 5: B2 — base-trained TXC encoder run on Stage A reasoning traces
python -m experiments.ward_backtracking_txc.b2_cross_model

# Phase 6: plots
python -m experiments.ward_backtracking_txc.plot.training_curves
python -m experiments.ward_backtracking_txc.plot.feature_firing_heatmap
python -m experiments.ward_backtracking_txc.plot.steering_comparison_bars
python -m experiments.ward_backtracking_txc.plot.per_offset_firing
python -m experiments.ward_backtracking_txc.plot.cosine_matrix
python -m experiments.ward_backtracking_txc.plot.sentence_act_distributions
python -m experiments.ward_backtracking_txc.plot.text_examples
python -m experiments.ward_backtracking_txc.plot.b2_difference_area
python -m experiments.ward_backtracking_txc.plot.decoder_umap          # optional (UMAP install)
python -m experiments.ward_backtracking_txc.plot.decoder_umap_x_umap   # optional

# Or all at once (single-GPU, sequential phases)
bash experiments/ward_backtracking_txc/run_all.sh

# Or, on a multi-GPU pod (auto-detects via nvidia-smi -L)
bash experiments/ward_backtracking_txc/run_grid_2gpu.sh
```

Outputs land under `results/ward_backtracking_txc/`. Hyperparameters are in
`config.yaml`; CLI `--config <path>` lets you swap configs without touching code.

Stage A artifacts (`prompts.json`, `traces.json`, `sentence_labels.json`,
`dom_vectors.pt`) are read in-place — Stage B never re-runs them.

## Multi-GPU usage

`run_grid_2gpu.sh` auto-detects how many GPUs the pod exposes (via
`nvidia-smi -L | wc -l`) and fans cells across them:

- **Phase 1 (cache)**: single GPU, single forward pass.
- **Phase 2 (train), Phase 3 (mine), Phase 5 (B2)**: `(arch, hookpoint)`
  cells run round-robin across GPUs. Each subprocess sees one GPU via
  `CUDA_VISIBLE_DEVICES`. The orchestrator waits between batches so the
  in-flight pool size never exceeds the GPU count.
- **Phase 4 (B1)**: source list is partitioned across GPUs via
  `--source-shard k/N`. Each shard writes a separate JSON
  (`b1_steering_results__shard<k>of<N>.json`); the orchestrator merges
  them into the canonical `b1_steering_results.json` after all shards
  finish (dedupes DoM baselines, which every shard evaluates).
- **Phase 6 (plots)**: CPU-only, single process at the end.

To force a specific GPU count (e.g. for benchmarking): `NUM_GPUS=1 bash …`
or `NUM_GPUS=2 bash …`. To skip a phase: `SKIP_PHASE1=1` or `SKIP_TRAIN=1`.

The phase scripts also accept `--arch <list>` and `--only <hookpoint>` so
you can launch single-cell sub-runs by hand if the orchestrator needs
debugging.

## Architectures sweept

`config.yaml`'s `txc.arch_list` controls which dictionary architectures are
trained/mined/steered. Default: `[txc, topk_sae, stacked_sae, tsae]`.
- `txc` — `TemporalCrosscoder` (shared latent across T positions).
- `topk_sae` — flat TopK SAE applied per-token in the T-window.
- `stacked_sae` — T independent TopK SAEs, one per offset.
- `tsae` — Han's `TemporalSAE` (attention-based predicted+novel codes,
  vendored from `references/TemporalFeatureAnalysis/sae/saeTemporal.py`
  on `han-phase7-unification`).

`config.yaml`'s `txc.arch_kwargs.tsae` controls Han's TSAE-specific
hyperparameters (`n_heads`, `bottleneck_factor`, `sae_diff_type`,
`n_attn_layers`, `tied_weights`).

## Steering magnitude grid

Symmetric across zero per Dmitry's note ("no a-priori reason to favour
positive steering"): `[-16, -12, -8, -4, 0, 4, 8, 12, 16]`. Negative
magnitudes are evidence about direction sign and arch behavior, not
just floor checks.

## Coherence diagnostics

`plot/coherence.py` reads the existing B1 JSON (no new generation) and
reports distinct-2 / TTR / max consecutive same-word run / output length
per `(source, magnitude)` cell. Use it to sanity-check that the kw-rate
edge at high magnitudes isn't being bought with degenerate
"Wait Wait Wait..." loops — see the Stage B writeup for the lesson
learned during the sprint.
