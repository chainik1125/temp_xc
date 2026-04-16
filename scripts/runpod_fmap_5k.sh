#!/bin/bash
# runpod_fmap_5k.sh — Re-render TXCDR feature maps with hover-text
# labels from the 5k autointerp explanations. RunPod version of
# trillium_sbatch_fmap_5k.sh.
#
# Usage (after scan + explain have run):
#   bash scripts/runpod_fmap_5k.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

run_fmap() {
    local TAG="$1"
    local MODEL="$2"
    local DATASET="$3"
    local LAYER="$4"
    local SHUF_SUFFIX="$5"
    local REPORT_DIR="$6"

    local CKPT="results/nlp/$TAG/ckpts/crosscoder__${MODEL}__${DATASET}__${LAYER}__k100__seed42${SHUF_SUFFIX}.pt"
    if [ ! -f "$CKPT" ]; then
        echo ">> SKIP $TAG — ckpt not found: $CKPT"
        return 0
    fi

    echo ""
    echo ">> [$TAG] feature_map (5k labels)"
    python -m temporal_crosscoders.NLP.feature_map \
        --checkpoint "$CKPT" \
        --model crosscoder --subject-model "$MODEL" \
        --k 100 --T 5 \
        --label "$TAG" \
        --output-dir "$REPORT_DIR" \
        --include-unlabeled --skip-llm-labels
}

run_fmap step1-unshuffled gemma-2-2b                     fineweb  resid_L13 "" \
    reports/step1-gemma-replication
run_fmap step1-shuffled   gemma-2-2b                     fineweb  resid_L13 "_shuffled" \
    reports/step1-gemma-replication
run_fmap step2-unshuffled deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "" \
    reports/step2-deepseek-reasoning
run_fmap step2-shuffled   deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "_shuffled" \
    reports/step2-deepseek-reasoning

echo ""
echo "=== fmap 5k done ==="
ls -lh reports/step1-gemma-replication/feature_map_*.png \
       reports/step2-deepseek-reasoning/feature_map_*.png 2>/dev/null || true
