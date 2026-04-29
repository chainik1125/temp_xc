#!/usr/bin/env bash
# Stage B end-to-end runner. Each script is idempotent and skips when its
# output already exists; pass --force on individual phases to re-run.
set -euo pipefail
cd "$(dirname "$0")/../.."

ROOT="experiments.ward_backtracking_txc"

echo "[run_all] Phase 1 — cache base activations"
python -m $ROOT.cache_activations

echo "[run_all] Phase 2 — train TXC × hookpoints"
python -m $ROOT.train_txc

echo "[run_all] Phase 3 — mine features"
python -m $ROOT.mine_features

echo "[run_all] Phase 4 — B1 steering eval"
python -m $ROOT.b1_steer_eval

echo "[run_all] Phase 5 — B2 cross-model"
python -m $ROOT.b2_cross_model

echo "[run_all] Phase 6 — plots"
python -m $ROOT.plot.training_curves
python -m $ROOT.plot.feature_firing_heatmap
python -m $ROOT.plot.steering_comparison_bars
python -m $ROOT.plot.per_offset_firing
python -m $ROOT.plot.cosine_matrix
python -m $ROOT.plot.sentence_act_distributions
python -m $ROOT.plot.text_examples
python -m $ROOT.plot.b2_difference_area
python -m $ROOT.plot.decoder_umap || true       # optional umap
python -m $ROOT.plot.decoder_umap_x_umap || true

echo "[run_all] done"
