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

# Or all at once
bash experiments/ward_backtracking_txc/run_all.sh
```

Outputs land under `results/ward_backtracking_txc/`. Hyperparameters are in
`config.yaml`; CLI `--config <path>` lets you swap configs without touching code.

Stage A artifacts (`prompts.json`, `traces.json`, `sentence_labels.json`,
`dom_vectors.pt`) are read in-place — Stage B never re-runs them.
