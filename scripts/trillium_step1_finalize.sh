#!/bin/bash
# trillium_step1_finalize.sh — Run autointerp + labeled feature_map on the
# checkpoints produced by trillium_step1_gemma_replication.sh. Runs ENTIRELY
# ON THE LOGIN NODE (CPU for TopKFinder scan, HTTPS to api.anthropic.com for
# Claude explanations). No sbatch / srun.
#
# Prereqs:
#   - trillium_step1_gemma_replication.sh has finished. squeue -u $USER empty
#     for step1-* jobs.
#   - ANTHROPIC_API_KEY is set in ~/.txc_secrets.env
#   - You are on trig-login01 (not inside an allocation)
#
# What it does, per checkpoint found under results/nlp/step1-*/ckpts/:
#   1. autointerp.py: scans activations with the trained crosscoder,
#      finds top-K activating windows per feature, calls Claude Haiku to
#      generate one-sentence explanations for the top-N features, saves
#      feat_*.json under reports/step1-gemma-replication/autointerp/<label>/
#   2. feature_map.py (re-run): reads those explanations and produces a
#      LABELED feature map PNG + HTML. Replaces the unlabeled one that was
#      made inline during the sbatch sweep.
#
# Cost estimate (Claude Haiku 4.5, ~$1/MTok in, $5/MTok out):
#   per checkpoint: 50 features * (~2k in + ~1k out tokens) ≈ $0.35
#   two checkpoints (unshuffled + shuffled) ≈ $0.70 total
#
# Usage:
#   bash scripts/trillium_step1_finalize.sh

set -euo pipefail

HOST=$(hostname)
if [[ "$HOST" != trig-login* ]]; then
    echo "FAIL: run this from the login node (you are on '$HOST')."
    echo "      This script needs outbound HTTPS to api.anthropic.com."
    exit 1
fi

source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "FAIL: ANTHROPIC_API_KEY not set. Paste into ~/.txc_secrets.env."
    exit 2
fi

cd "$SCRATCH/temp_xc"

MODEL="gemma-2-2b"
DATASET="fineweb"
LAYER="resid_L13"
K="${K:-100}"
T="${T:-5}"
REPORT_DIR="reports/step1-gemma-replication"
mkdir -p "$REPORT_DIR"

echo "=== step1 finalize (login-node) ==="
echo "  subject:    $MODEL"
echo "  dataset:    $DATASET"
echo "  layer:      $LAYER"
echo "  k=$K T=$T"
echo "  cost est:   ~\$0.35 per checkpoint, ~\$0.70 total"
echo ""

for TAG in unshuffled shuffled; do
    SUFFIX=""
    [ "$TAG" = "shuffled" ] && SUFFIX="_shuffled"

    RESULTS="results/nlp/step1-${TAG}"
    CKPT="$RESULTS/ckpts/crosscoder__${MODEL}__${DATASET}__${LAYER}__k${K}__seed42${SUFFIX}.pt"
    LABEL="step1-${TAG}"

    if [ ! -f "$CKPT" ]; then
        echo ">> SKIP $TAG — checkpoint not found: $CKPT"
        continue
    fi

    echo ""
    echo ">> [$TAG] autointerp"
    python -m temporal_crosscoders.NLP.autointerp \
        --checkpoint "$CKPT" \
        --model crosscoder --subject-model $MODEL \
        --cached-dataset $DATASET --layer-key $LAYER \
        --k $K --T $T \
        --label "$LABEL" \
        --output-dir "$REPORT_DIR" \
        --explain-model claude-haiku-4-5-20251001 \
        --no-harm

    echo ""
    echo ">> [$TAG] labeled feature_map"
    python -m temporal_crosscoders.NLP.feature_map \
        --checkpoint "$CKPT" \
        --model crosscoder --subject-model $MODEL \
        --k $K --T $T \
        --label "$LABEL" \
        --output-dir "$REPORT_DIR" \
        --skip-llm-labels        # cluster-level labels disabled for now (per-feature labels still appear on hover)
done

echo ""
echo "=== step1 finalize done ==="
echo "Reports under: $REPORT_DIR"
ls -lh "$REPORT_DIR" 2>/dev/null || true
