#!/bin/bash
# runpod_backfill_metrics.sh — Compute the three pre-registered sprint
# metrics (silhouette, cluster_entropy, mean_auto_mi across lags) on
# the 12 existing step1/step2 checkpoints. RunPod version of
# trillium_backfill_metrics.sh.
#
# Pre-registration: docs/aniket/sprint_coding_dataset_plan.md
#
# Usage:
#   bash scripts/runpod_backfill_metrics.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

mkdir -p reports/step1-gemma-replication reports/step2-deepseek-reasoning

run_metrics() {
    local TAG="$1"
    local ARCH="$2"
    local MODEL="$3"
    local DATASET="$4"
    local LAYER="$5"
    local SHUF_SUFFIX="$6"
    local REPORT_DIR="$7"

    local CKPT="results/nlp/${TAG}/ckpts/${ARCH}__${MODEL}__${DATASET}__${LAYER}__k100__seed42${SHUF_SUFFIX}.pt"
    if [ ! -f "$CKPT" ]; then
        echo ">> SKIP ${TAG}/${ARCH} — ckpt not found: $CKPT"
        return 0
    fi

    local LABEL="${TAG}__${ARCH}"
    echo ""
    echo ">> [${LABEL}] compute_temporal_metrics"
    python scripts/compute_temporal_metrics.py \
        --checkpoint "$CKPT" \
        --arch "$ARCH" \
        --subject-model "$MODEL" \
        --cached-dataset "$DATASET" \
        --layer-key "$LAYER" \
        --k 100 --T 5 \
        --label "$LABEL" \
        --output-dir "$REPORT_DIR" \
        --device cuda
}

# step1: Gemma + FineWeb
for ARCH in topk_sae stacked_sae crosscoder; do
    run_metrics step1-unshuffled "$ARCH" gemma-2-2b fineweb resid_L13 "" \
        reports/step1-gemma-replication
    run_metrics step1-shuffled   "$ARCH" gemma-2-2b fineweb resid_L13 "_shuffled" \
        reports/step1-gemma-replication
done

# step2: DeepSeek + GSM8K
for ARCH in topk_sae stacked_sae crosscoder; do
    run_metrics step2-unshuffled "$ARCH" deepseek-r1-distill-llama-8b gsm8k resid_L12 "" \
        reports/step2-deepseek-reasoning
    run_metrics step2-shuffled   "$ARCH" deepseek-r1-distill-llama-8b gsm8k resid_L12 "_shuffled" \
        reports/step2-deepseek-reasoning
done

echo ""
echo "=== backfill done ==="
echo "Step 1 metrics:"
ls -1 reports/step1-gemma-replication/metrics_*.json 2>/dev/null | sed 's|.*metrics_|  |' || true
echo ""
echo "Step 2 metrics:"
ls -1 reports/step2-deepseek-reasoning/metrics_*.json 2>/dev/null | sed 's|.*metrics_|  |' || true
