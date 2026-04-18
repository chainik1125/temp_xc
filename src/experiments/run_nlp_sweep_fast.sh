#!/bin/bash
# =================================================================
# FAST NLP Sweep: TXCDR + Stacked SAE only (TFA debug in parallel)
# Runs on cached Gemma activations, all 4 sweep batches.
# ~2h total for Gemma (no TFA = fast).
# =================================================================

set -euo pipefail

export TQDM_DISABLE=1
export PYTHONPATH=/home/elysium/temp_xc
export PYTHONUNBUFFERED=1

PYTHON="/home/elysium/miniforge3/envs/torchgpu/bin/python"
RESULTS_DIR="results/nlp_sweep"
STEPS=10000

mkdir -p "$RESULTS_DIR" logs

echo "============================================================"
echo "  FAST NLP SWEEP (TXCDR + Stacked only) — $(date)"
echo "============================================================"

# ── Gemma sweeps: 4 batches x 2 archs x 2 k = 16 runs @ ~7 min each ~2h
for LAYER in resid_L25 resid_L13; do
    for SHUF_FLAG in "" "--shuffle-within-sequence"; do
        MODE="unshuffled"
        if [ -n "$SHUF_FLAG" ]; then MODE="shuffled"; fi
        echo ""
        echo "---- Gemma | ${LAYER} | ${MODE} | $(date) ----"

        $PYTHON -u -m src.bench.sweep \
            --dataset-type cached_activations \
            --model-name gemma-2-2b-it \
            --cached-dataset fineweb \
            --cached-layer-key "$LAYER" \
            --models stacked_sae crosscoder \
            --k 50 100 \
            --T 5 \
            --steps $STEPS \
            --expansion-factor 8 \
            --results-dir "${RESULTS_DIR}/gemma" \
            $SHUF_FLAG
    done
done

echo "  Gemma TXCDR+Stacked complete: $(date)"

# ── DeepSeek caching + sweep (3h cache + 1.5h train ~4.5h)
echo ""
echo "==== Cache DeepSeek-R1-8B (12K x 128 tok) ===="
$PYTHON -u -m src.bench.cache_activations \
    --model deepseek-r1-distill-llama-8b \
    --dataset fineweb \
    --mode forward \
    --num-sequences 12000 \
    --seq-length 128 \
    --batch-size 16 \
    --layer_indices 12 \
    --components resid

echo ""
for SHUF_FLAG in "" "--shuffle-within-sequence"; do
    MODE="unshuffled"
    if [ -n "$SHUF_FLAG" ]; then MODE="shuffled"; fi
    echo ""
    echo "---- DeepSeek | resid_L12 | ${MODE} | $(date) ----"

    $PYTHON -u -m src.bench.sweep \
        --dataset-type cached_activations \
        --model-name deepseek-r1-distill-llama-8b \
        --cached-dataset fineweb \
        --cached-layer-key resid_L12 \
        --models stacked_sae crosscoder \
        --k 50 100 \
        --T 5 \
        --steps $STEPS \
        --expansion-factor 4 \
        --results-dir "${RESULTS_DIR}/deepseek" \
        $SHUF_FLAG
done

echo ""
echo "============================================================"
echo "  FAST SWEEP DONE — $(date)"
echo "============================================================"
