#!/bin/bash
# trillium_step2_finalize.sh — Same as step1_finalize.sh but for DeepSeek-R1-
# Distill + GSM8K reasoning traces. Runs on the LOGIN NODE.
#
# Cost estimate (Claude Haiku 4.5):
#   per checkpoint: 30 features * 1 explain call (~2k in + ~500 out) ≈ $0.14
#   two checkpoints ≈ $0.28 total
#
# Detection scoring is OFF (see step1_finalize.sh header for rationale).
# HypotheSAEs fidelity scoring is the TODO replacement.
#
# Usage:
#   bash scripts/trillium_step2_finalize.sh

set -euo pipefail

HOST=$(hostname)
if [[ "$HOST" != trig-login* ]]; then
    echo "FAIL: run this from the login node (you are on '$HOST')."
    exit 1
fi

source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "FAIL: ANTHROPIC_API_KEY not set."
    exit 2
fi

cd "$SCRATCH/temp_xc"

MODEL="deepseek-r1-distill-llama-8b"
DATASET="gsm8k"
LAYER="${LAYER:-resid_L12}"
K="${K:-100}"
T="${T:-5}"
REPORT_DIR="reports/step2-deepseek-reasoning"
mkdir -p "$REPORT_DIR"

echo "=== step2 finalize (login-node) ==="
echo "  subject:    $MODEL"
echo "  dataset:    $DATASET"
echo "  layer:      $LAYER"
echo "  k=$K T=$T"
echo "  top-features: 30   (autointerp will explain this many per ckpt)"
echo "  scoring:    OFF  (HypotheSAEs fidelity scoring is the TODO)"
echo "  cost est:   ~\$0.14 per checkpoint, ~\$0.28 total"
echo ""

for TAG in unshuffled shuffled; do
    SUFFIX=""
    [ "$TAG" = "shuffled" ] && SUFFIX="_shuffled"

    RESULTS="results/nlp/step2-${TAG}"
    CKPT="$RESULTS/ckpts/crosscoder__${MODEL}__${DATASET}__${LAYER}__k${K}__seed42${SUFFIX}.pt"
    LABEL="step2-${TAG}"

    if [ ! -f "$CKPT" ]; then
        echo ">> SKIP $TAG — checkpoint not found: $CKPT"
        continue
    fi

    echo ""
    echo ">> [$TAG] autointerp (explanations only, scoring off)"
    # Force CPU — login node GPU is shared and usually saturated. See
    # trillium_step1_finalize.sh for the rationale.
    python -m temporal_crosscoders.NLP.autointerp \
        --checkpoint "$CKPT" \
        --model crosscoder --subject-model $MODEL \
        --cached-dataset $DATASET --layer-key $LAYER \
        --k $K --T $T \
        --label "$LABEL" \
        --output-dir "$REPORT_DIR" \
        --explain-model claude-haiku-4-5-20251001 \
        --top-features 30 \
        --scan-device cpu --explain-device cpu \
        --no-score --no-harm

    echo ""
    echo ">> [$TAG] labeled feature_map"
    python -m temporal_crosscoders.NLP.feature_map \
        --checkpoint "$CKPT" \
        --model crosscoder --subject-model $MODEL \
        --k $K --T $T \
        --label "$LABEL" \
        --output-dir "$REPORT_DIR" \
        --skip-llm-labels
done

echo ""
echo "=== step2 finalize done ==="
ls -lh "$REPORT_DIR" 2>/dev/null || true
