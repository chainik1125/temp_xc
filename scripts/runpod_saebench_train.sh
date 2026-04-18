#!/bin/bash
# runpod_saebench_train.sh — train one SAEBench-experiment checkpoint.
#
# Unified trainer for {sae, mlc, tempxc} × {protocol A, B} × {T values}.
# Reads the k value from src/bench/saebench/matching_protocols.py so
# there's a single source of truth.
#
# Usage:
#   bash scripts/runpod_saebench_train.sh --arch sae    --protocol A
#   bash scripts/runpod_saebench_train.sh --arch tempxc --protocol B --t 10
#
# Resumable: skips if the checkpoint already exists unless --force.
#
# Outputs:
#   results/saebench/ckpts/<arch>__...pt
#   results/saebench/logs/<arch>__...log

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

ARCH=""
PROTOCOL=""
T=5
SEED=42
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)     ARCH="$2";     shift 2;;
        --protocol) PROTOCOL="$2"; shift 2;;
        --t)        T="$2";        shift 2;;
        --seed)     SEED="$2";     shift 2;;
        --force)    FORCE=1;       shift;;
        *) echo "unknown flag: $1"; exit 2;;
    esac
done

for req in ARCH PROTOCOL; do
    if [ -z "${!req}" ]; then
        echo "FAIL: --${req,,} is required"
        exit 2
    fi
done
case "$ARCH" in
    sae|mlc|tempxc) ;;
    *) echo "FAIL: --arch must be sae, mlc, or tempxc (got: $ARCH)"; exit 2;;
esac
case "$PROTOCOL" in
    A|B) ;;
    *) echo "FAIL: --protocol must be A or B (got: $PROTOCOL)"; exit 2;;
esac

# Resolve k and checkpoint name via Python (single source of truth)
eval "$(python - <<PY
from src.bench.saebench.matching_protocols import protocol_k
from src.bench.saebench.configs import ckpt_name, CKPT_DIR, LOG_DIR
k = protocol_k("$ARCH", "$PROTOCOL", t=$T)
name = ckpt_name("$ARCH", "$PROTOCOL", t=$T, seed=$SEED)
print(f'K={k}')
print(f'CKPT_NAME="{name}"')
print(f'CKPT_DIR="{CKPT_DIR}"')
print(f'LOG_DIR="{LOG_DIR}"')
PY
)"

CKPT_PATH="${CKPT_DIR}/${CKPT_NAME}"
LOG_PATH="${LOG_DIR}/${CKPT_NAME%.pt}.log"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

echo "=== saebench train ==="
echo "  arch:     $ARCH"
echo "  protocol: $PROTOCOL"
echo "  T:        $T"
echo "  k:        $K"
echo "  seed:     $SEED"
echo "  ckpt:     $CKPT_PATH"
echo "  log:      $LOG_PATH"

if [ -f "$CKPT_PATH" ] && [ "$FORCE" -ne 1 ]; then
    echo ">> already exists, skipping (use --force to retrain)"
    exit 0
fi

# Training dispatch. The base sweep driver (src.bench.sweep) runs a
# full sweep; for a single-architecture saebench run we call it with
# --models <arch-key> and a matching results-dir, then rename the
# checkpoint into our saebench layout afterwards.
STEPS="${STEPS:-30000}"  # generous upper bound; paired with plateau early-stop
PLATEAU_PCT="${PLATEAU_PCT:-0.005}"   # 0.5% drop / 2k steps = plateaued
PLATEAU_MIN_STEPS="${PLATEAU_MIN_STEPS:-5000}"

case "$ARCH" in
    sae)
        MODELS="topk_sae"
        T_FLAG=""
        DATASET_FLAGS="--dataset-type cached_activations --cached-layer-key resid_L12"
        ;;
    tempxc)
        MODELS="crosscoder"
        T_FLAG="--T $T"
        DATASET_FLAGS="--dataset-type cached_activations --cached-layer-key resid_L12"
        ;;
    mlc)
        MODELS="mlc"
        # MLC's "T" is n_layers (window width across the layer axis);
        # we reuse --T to keep the sweep CLI uniform.
        T_FLAG="--T 5"
        # Multi-layer cache: sweep builds (B, 5, d_model) inputs by
        # stacking layers 10..14 from separate per-layer .npy caches.
        DATASET_FLAGS="--dataset-type multi_layer_activations \
            --cached-layer-keys resid_L10 resid_L11 resid_L12 resid_L13 resid_L14"
        ;;
esac

SWEEP_RESULTS_DIR="results/saebench/sweeps/${ARCH}_prot${PROTOCOL}_T${T}"
mkdir -p "$SWEEP_RESULTS_DIR"

# Surface protocol + T into the W&B run name (sweep.py reads these).
export SAEBENCH_PROTOCOL="$PROTOCOL"
export SAEBENCH_T="$T"

python -m src.bench.sweep \
    $DATASET_FLAGS \
    --model-name gemma-2-2b \
    --cached-dataset fineweb \
    --models "$MODELS" \
    --k "$K" $T_FLAG --steps "$STEPS" \
    --stop-on-plateau "$PLATEAU_PCT" \
    --plateau-min-steps "$PLATEAU_MIN_STEPS" \
    --results-dir "$SWEEP_RESULTS_DIR" \
    2>&1 | tee "$LOG_PATH"

# Move the sweep-produced checkpoint to our saebench ckpts dir
SWEEP_CKPT=$(ls "$SWEEP_RESULTS_DIR"/ckpts/*.pt 2>/dev/null | head -1)
if [ -z "$SWEEP_CKPT" ]; then
    echo "FAIL: sweep produced no checkpoint under $SWEEP_RESULTS_DIR/ckpts/"
    exit 4
fi
cp "$SWEEP_CKPT" "$CKPT_PATH"
echo ""
echo "=== done ==="
echo "  ckpt copied: $CKPT_PATH"
echo "  size: $(du -h "$CKPT_PATH" | awk '{print $1}')"
