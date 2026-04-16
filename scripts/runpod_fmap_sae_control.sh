#!/bin/bash
# runpod_fmap_sae_control.sh — Geometric gate plot. Renders feature_map
# on the TopKSAE checkpoints from step2 (DeepSeek + GSM8K) so we can
# eyeball whether TopKSAE shows the same isolated island as TXCDR.
# RunPod version of trillium_sbatch_fmap_sae_control.sh.
#
# Usage:
#   bash scripts/runpod_fmap_sae_control.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

mkdir -p reports/sae-control-deepseek

run_fmap() {
    local TAG="$1"
    local ARCH="$2"
    local MODEL="$3"
    local DATASET="$4"
    local LAYER="$5"
    local SHUF_SUFFIX="$6"
    local REPORT_DIR="$7"
    local LABEL="$8"

    local CKPT="results/nlp/$TAG/ckpts/${ARCH}__${MODEL}__${DATASET}__${LAYER}__k100__seed42${SHUF_SUFFIX}.pt"
    if [ ! -f "$CKPT" ]; then
        echo ">> SKIP $LABEL — ckpt not found: $CKPT"
        return 0
    fi

    echo ""
    echo ">> [$LABEL] feature_map ($ARCH)"
    python -m temporal_crosscoders.NLP.feature_map \
        --checkpoint "$CKPT" \
        --model "$ARCH" --subject-model "$MODEL" \
        --k 100 --T 5 \
        --label "$LABEL" \
        --output-dir "$REPORT_DIR" \
        --include-unlabeled --skip-llm-labels
}

run_fmap step2-unshuffled topk_sae deepseek-r1-distill-llama-8b gsm8k resid_L12 "" \
    reports/sae-control-deepseek sae-deepseek-unshuffled
run_fmap step2-shuffled   topk_sae deepseek-r1-distill-llama-8b gsm8k resid_L12 "_shuffled" \
    reports/sae-control-deepseek sae-deepseek-shuffled
run_fmap step2-unshuffled crosscoder deepseek-r1-distill-llama-8b gsm8k resid_L12 "" \
    reports/sae-control-deepseek txcdr-deepseek-unshuffled
run_fmap step2-shuffled   crosscoder deepseek-r1-distill-llama-8b gsm8k resid_L12 "_shuffled" \
    reports/sae-control-deepseek txcdr-deepseek-shuffled

echo ""
echo "=== SAE control done ==="
ls -lh reports/sae-control-deepseek/*.png 2>/dev/null || true
