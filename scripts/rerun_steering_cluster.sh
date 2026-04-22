#!/bin/bash
# rerun_steering_cluster.sh — re-train one steering vector (single cluster
# index, or bias=-1) using the same config as MODE=hybrid. Used when a
# worker in the parallel Phase 2 dies and we need to retry just that
# vector in isolation.
#
# Usage:
#   bash scripts/rerun_steering_cluster.sh [CLUSTER_IDX]
#
# Defaults:
#   CLUSTER_IDX=1
#   ARCH=sae
#   NUM_GPUS=1  (serial; cleaner logs than 2-worker parallel)
#
# All knobs match MODE=hybrid defaults. Override via env vars:
#   ARCH=tempxc CLUSTER_IDX=3 bash scripts/rerun_steering_cluster.sh
#
# Resume semantics: the other vectors (bias + whichever clusters already
# have .pt files) no-op via the per-vector cache check inside run_steering.
# Only the one you ask for actually trains.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

CLUSTER_IDX="${1:-${CLUSTER_IDX:-1}}"
ARCH="${ARCH:-sae}"
MODEL="${MODEL:-deepseek-r1-distill-llama-8b}"
DATASET="${DATASET:-math500}"
N_TRACES="${N_TRACES:-500}"
LAYER="${LAYER:-6}"
SEED="${SEED:-42}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B}"
THINKING_MODEL_HF="${THINKING_MODEL_HF:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
STEERING_LAYER="${STEERING_LAYER:-12}"
SAE_LAYER="${SAE_LAYER:-6}"
N_CLUSTERS_HYBRID="${N_CLUSTERS_HYBRID:-15}"
STEER_MAX_ITERS="${STEER_MAX_ITERS:-10}"
STEER_N_TRAINING="${STEER_N_TRAINING:-256}"
STEER_N_EVAL="${STEER_N_EVAL:-0}"
STEER_MINIBATCH="${STEER_MINIBATCH:-4}"
NUM_GPUS="${NUM_GPUS:-1}"
ROOT="${ROOT:-results/venhoff_eval}"
VENHOFF_ROOT="${VENHOFF_ROOT:-vendor/thinking-llms-interp}"

mkdir -p logs

LOG_PATH="logs/rerun_steering_${ARCH}_idx${CLUSTER_IDX}_$(date +%Y%m%d_%H%M).log"

echo "[info] rerun_steering | arch=$ARCH | cluster_idx=$CLUSTER_IDX | num_gpus=$NUM_GPUS | log=$LOG_PATH"

python -m src.bench.venhoff.run_steering \
    --root "$ROOT" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --n-traces "$N_TRACES" \
    --layer "$LAYER" \
    --seed "$SEED" \
    --arch "$ARCH" \
    --venhoff-root "$VENHOFF_ROOT" \
    --base-model "$BASE_MODEL" \
    --thinking-model "$THINKING_MODEL_HF" \
    --steering-layer "$STEERING_LAYER" \
    --sae-layer "$SAE_LAYER" \
    --n-clusters "$N_CLUSTERS_HYBRID" \
    --max-iters "$STEER_MAX_ITERS" \
    --n-training-examples "$STEER_N_TRAINING" \
    --n-eval-examples "$STEER_N_EVAL" \
    --optim-minibatch-size "$STEER_MINIBATCH" \
    --cluster-indices "$CLUSTER_IDX" \
    --num-gpus "$NUM_GPUS" \
    2>&1 | tee "$LOG_PATH"

echo "[done] rerun_steering | arch=$ARCH | cluster_idx=$CLUSTER_IDX"
echo "  vector file: $VENHOFF_ROOT/train-vectors/results/vars/optimized_vectors/$(basename "$BASE_MODEL" | tr '[:upper:]' '[:lower:]')_idx${CLUSTER_IDX}_linear.pt"
echo "  log:         $LOG_PATH"
