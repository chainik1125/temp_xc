#!/bin/bash
# runpod_step3_gemma_stack.sh — Step 3 of the feature-map sprint:
# Gemma 2B + Stack-Python, the H3 rule-out cell in the 2×3 matrix.
# RunPod version of trillium_step3_gemma_stack.sh — runs in foreground,
# no SLURM dependencies, no login-node prefetch (pod has internet).
#
#   cache Stack-Python activations on Gemma 2 2B
#     -> unshuffled architecture sweep
#     -> shuffled   architecture sweep
#     -> feature_map on crosscoder ckpt (each)
#
# Pre-registration: docs/aniket/sprint_coding_dataset_plan.md
#
# Usage:
#   bash scripts/runpod_step3_gemma_stack.sh
#
# Env overrides:
#   SEQ_LEN=512      longer context
#   NUM_SEQS=6000    more sequences
#   STEPS=20000      longer training
#   SKIP_CACHE=1     reuse existing cache
#   ONLY=unshuffled  run only one shuffle condition

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

MODEL="gemma-2-2b"
DATASET="stack-python"
LAYER="${LAYER:-resid_L13}"
K="${K:-100}"
T="${T:-5}"
STEPS="${STEPS:-10000}"
NUM_SEQS="${NUM_SEQS:-3000}"
SEQ_LEN="${SEQ_LEN:-256}"
SKIP_CACHE="${SKIP_CACHE:-0}"
ONLY="${ONLY:-both}"

REPORT_DIR="reports/step3-gemma-stack"
mkdir -p "$REPORT_DIR"

# ── 1. cache (no login-node prefetch — RunPod has internet) ──────────
if [ "$SKIP_CACHE" = "1" ]; then
    echo ">> SKIP_CACHE=1, not re-caching"
else
    echo ""
    echo "=== step3 cache: $MODEL / $DATASET ==="
    python -m temporal_crosscoders.NLP.cache_activations \
        --model "$MODEL" --dataset "$DATASET" --mode forward \
        --num-sequences "$NUM_SEQS" --seq-length "$SEQ_LEN" \
        --layer_indices 13 25 --components resid
fi

# ── 2. sweep + feature_map per shuffle condition ─────────────────────
run_sweep() {
    local TAG="$1"          # "unshuffled" | "shuffled"
    local SHUF_FLAG="$2"    # "" | "--shuffle-within-sequence"
    local SHUF_SUFFIX="$3"  # "" | "_shuffled"

    local RESULTS="results/nlp/step3-${TAG}"

    echo ""
    echo "=== step3 ${TAG} sweep: $MODEL / $DATASET / $LAYER / k=$K T=$T ==="
    python -m src.bench.sweep \
        --dataset-type cached_activations \
        --model-name "$MODEL" \
        --cached-dataset "$DATASET" \
        --cached-layer-key "$LAYER" \
        --models topk_sae stacked_sae crosscoder \
        --k "$K" --T "$T" --steps "$STEPS" \
        --results-dir "$RESULTS" \
        $SHUF_FLAG

    local CKPT="$RESULTS/ckpts/crosscoder__${MODEL}__${DATASET}__${LAYER}__k${K}__seed42${SHUF_SUFFIX}.pt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "=== feature_map on $CKPT ==="
        python -m temporal_crosscoders.NLP.feature_map \
            --checkpoint "$CKPT" \
            --model crosscoder --subject-model "$MODEL" \
            --k "$K" --T "$T" \
            --include-unlabeled --skip-llm-labels \
            --label "step3-${TAG}" \
            --output-dir "$REPORT_DIR"
    else
        echo "WARN: expected checkpoint not found: $CKPT"
        ls -la "$RESULTS/ckpts/" || true
    fi
}

if [ "$ONLY" != "shuffled" ]; then
    run_sweep "unshuffled" "" ""
fi
if [ "$ONLY" != "unshuffled" ]; then
    run_sweep "shuffled" "--shuffle-within-sequence" "_shuffled"
fi

echo ""
echo "=== step3 done ==="
ls -lh "$REPORT_DIR"/feature_map_step3-*.png 2>/dev/null || true
echo ""
echo "Next:"
echo "  bash scripts/runpod_backfill_metrics.sh   # add step3 to metric JSONs"
