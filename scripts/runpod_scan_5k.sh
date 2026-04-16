#!/bin/bash
# runpod_scan_5k.sh — Autointerp SCAN phase: 5000 features per checkpoint
# across 4 crosscoder checkpoints (step1/step2 × shuf/unshuf). RunPod
# version of trillium_scan_5k_sbatch.sh — no SLURM, runs in foreground.
#
# Outputs feat_*.json files with top_texts/top_activations populated but
# empty `explanation` fields (the explain phase fills those in).
#
# Usage:
#   bash scripts/runpod_scan_5k.sh
#   TOP_FEATURES=500 bash scripts/runpod_scan_5k.sh   # cheaper variant

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

TOP_FEATURES="${TOP_FEATURES:-5000}"

run_scan() {
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
    echo ">> [$TAG] autointerp scan (top_features=$TOP_FEATURES)"
    python -m temporal_crosscoders.NLP.autointerp \
        --phase scan \
        --checkpoint "$CKPT" \
        --model crosscoder --subject-model "$MODEL" \
        --cached-dataset "$DATASET" --layer-key "$LAYER" \
        --k 100 --T 5 \
        --label "$TAG" \
        --output-dir "$REPORT_DIR" \
        --top-features "$TOP_FEATURES" \
        --scan-device cuda
}

mkdir -p reports/step1-gemma-replication reports/step2-deepseek-reasoning

run_scan step1-unshuffled gemma-2-2b                     fineweb  resid_L13 "" \
    reports/step1-gemma-replication
run_scan step1-shuffled   gemma-2-2b                     fineweb  resid_L13 "_shuffled" \
    reports/step1-gemma-replication
run_scan step2-unshuffled deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "" \
    reports/step2-deepseek-reasoning
run_scan step2-shuffled   deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "_shuffled" \
    reports/step2-deepseek-reasoning

echo ""
echo "=== scan 5k done ==="
for d in reports/step1-gemma-replication/autointerp/step1-* \
         reports/step2-deepseek-reasoning/autointerp/step2-*; do
    [ -d "$d" ] && echo "  $(basename "$d"): $(find "$d" -name 'feat_*.json' | wc -l) features"
done
