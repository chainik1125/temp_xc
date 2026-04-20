#!/usr/bin/env bash
# Phase 5 local orchestrator for RTX 5090 (32 GB VRAM).
#
# Pipeline:
#   1. Build any missing per-layer activation caches (L11, L12, L14, L15)
#   2. Build probe cache for all 27 tasks (25 SAEBench-style + 2 cross-token)
#   3. Train 19 architectures (7 primary + 12 overnight-sprint)
#   4. Extract per-(run, task, aggregation) features to feature_cache/
#   5. Fit probes on cached features (L1 LR, top-k class-sep, AUC + ACC,
#      plus the two required baselines)
#   6. Regenerate the 8 headline plots (2 task sets × 2 aggregations × 2 metrics)
#
# Each step is idempotent: reruns skip already-complete work.

set -eu
REPO="${PHASE5_REPO:-/home/elysium/temp_xc}"
cd "$REPO"

export PHASE5_REPO="$REPO"
export PYTHONPATH="$REPO"
export TQDM_DISABLE=1

LOG_DIR="$REPO/experiments/phase5_downstream_utility/results/logs"
mkdir -p "$LOG_DIR"

say() { echo "[$(date +%H:%M:%S)] $*"; }

# ─── Step 1: cache L11, L12, L14, L15 (idempotent)
say "Step 1: build missing activation layers (L11, L12, L14, L15)"
.venv/bin/python experiments/phase5_downstream_utility/build_multilayer_cache_batch.py \
    --layers 11 12 14 15 --batch-size 16 \
    2>&1 | tee "$LOG_DIR/step1_activation_cache.log"

# ─── Step 2: probe cache (27 tasks)
say "Step 2: build probe cache (all tasks)"
.venv/bin/python experiments/phase5_downstream_utility/probing/build_probe_cache.py \
    --include-crosstoken 2>&1 | tee "$LOG_DIR/step2_probe_cache.log"

# ─── Step 3: train 19 architectures (skip any already in training_index.jsonl)
say "Step 3: train 19 architectures"
ALL_ARCHS=(
    "topk_sae"
    "mlc"
    "txcdr_t5"
    "txcdr_t20"
    "stacked_t5"
    "stacked_t20"
    "matryoshka_t5"
    "txcdr_shared_dec_t5"
    "txcdr_shared_enc_t5"
    "txcdr_tied_t5"
    "txcdr_pos_t5"
    "txcdr_causal_t5"
    "txcdr_block_sparse_t5"
    "txcdr_lowrank_dec_t5"
    "txcdr_rank_k_dec_t5"
    "temporal_contrastive"
    "tfa_small"
    "tfa_pos_small"
    "time_layer_crosscoder_t5"
)
for arch in "${ALL_ARCHS[@]}"; do
    ckpt="$REPO/experiments/phase5_downstream_utility/results/ckpts/${arch}__seed42.pt"
    if [ -f "$ckpt" ]; then
        say "  $arch: ckpt exists, skip"
        continue
    fi
    say "  $arch: training..."
    .venv/bin/python experiments/phase5_downstream_utility/train_primary_archs.py \
        --seeds 42 --max-steps 25000 --archs "$arch" \
        2>&1 | tee "$LOG_DIR/step3_train_${arch}.log"
done

# ─── Step 4: extract features for all (run, task, aggregation)
say "Step 4: extract features"
.venv/bin/python experiments/phase5_downstream_utility/probing/extract_features.py \
    --aggregations last_position full_window \
    2>&1 | tee "$LOG_DIR/step4_extract_features.log"

# ─── Step 5: fit probes (baselines + SAE probes; both metrics)
say "Step 5: fit probes + baselines"
rm -f "$REPO/experiments/phase5_downstream_utility/results/probing_results.jsonl"
.venv/bin/python experiments/phase5_downstream_utility/probing/fit_probes.py \
    --aggregations last_position full_window \
    --k-values 1 2 5 20 \
    2>&1 | tee "$LOG_DIR/step5_fit_probes.log"

# ─── Step 6: plots (8 headline + 8 per-task heatmaps + training curves + SVD)
say "Step 6: plots"
.venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py \
    2>&1 | tee "$LOG_DIR/step6_plots.log"
.venv/bin/python experiments/phase5_downstream_utility/plots/plot_training_curves.py \
    2>&1 | tee -a "$LOG_DIR/step6_plots.log"
.venv/bin/python experiments/phase5_downstream_utility/analyze_decoder_svd.py \
    2>&1 | tee -a "$LOG_DIR/step6_plots.log" || true

say "Phase 5 local pipeline complete."
