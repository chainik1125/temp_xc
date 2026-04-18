#!/bin/bash
# =================================================================
# FULL NLP Sweep: TFA-pos + TXCDR + Stacked SAE on Gemma + DeepSeek
# Uses validated TFA-stable hyperparameters (lr=3e-4, bs=32, bf=8, norm_decoder)
# 5K steps for TFA (not 10K) to fit in budget.
# Resumable: skips completed runs, saves after each one.
# =================================================================

set -euo pipefail

export TQDM_DISABLE=1
export PYTHONPATH=/home/elysium/temp_xc
export PYTHONUNBUFFERED=1

PYTHON="/home/elysium/miniforge3/envs/torchgpu/bin/python"
RESULTS_DIR="results/nlp_sweep"
STEPS_TXCDR=10000  # Stacked/TXCDR stable at 10K
STEPS_TFA=5000     # TFA needs bigger batches, so fewer steps at same token-count

mkdir -p "$RESULTS_DIR" logs

echo "============================================================"
echo "  FULL NLP SWEEP — $(date)"
echo "============================================================"

# ── Phase A: Gemma TXCDR + Stacked (fast, unblocks TFA work) ───
echo ""
echo "==== PHASE A: Gemma TXCDR + Stacked (16 runs) ===="
for LAYER in resid_L25 resid_L13; do
    for SHUF in "" "--shuffle-within-sequence"; do
        MODE="unshuffled"
        [ -n "$SHUF" ] && MODE="shuffled"
        echo ""
        echo "---- Gemma | ${LAYER} | ${MODE} | $(date) ----"
        $PYTHON -u -m src.pipeline.sweep \
            --dataset-type cached_activations \
            --model-name gemma-2-2b-it \
            --cached-dataset fineweb \
            --cached-layer-key "$LAYER" \
            --models stacked_sae crosscoder \
            --k 50 100 --T 5 --steps $STEPS_TXCDR \
            --expansion-factor 8 \
            --results-dir "${RESULTS_DIR}/gemma" \
            $SHUF
    done
done

# ── Phase B: Gemma TFA-pos (slow but critical) ────────────────
echo ""
echo "==== PHASE B: Gemma TFA-pos (8 runs) ===="
for LAYER in resid_L25 resid_L13; do
    for SHUF in "" "--shuffle-within-sequence"; do
        MODE="unshuffled"
        [ -n "$SHUF" ] && MODE="shuffled"
        echo ""
        echo "---- Gemma TFA | ${LAYER} | ${MODE} | $(date) ----"
        $PYTHON -u -m src.pipeline.sweep \
            --dataset-type cached_activations \
            --model-name gemma-2-2b-it \
            --cached-dataset fineweb \
            --cached-layer-key "$LAYER" \
            --models tfa_pos \
            --k 50 100 --T 5 --steps $STEPS_TFA \
            --expansion-factor 8 \
            --tfa-bottleneck-factor 8 \
            --tfa-batch-size 32 \
            --results-dir "${RESULTS_DIR}/gemma" \
            $SHUF
    done
done

# ── Phase C: DeepSeek caching ───────────────────────────────────
echo ""
echo "==== PHASE C: Cache DeepSeek-R1-8B (12K x 128 tok, L12) ===="
$PYTHON -u -m src.data.nlp.cache_activations \
    --model deepseek-r1-distill-llama-8b \
    --dataset fineweb --mode forward \
    --num-sequences 12000 --seq-length 128 \
    --batch-size 16 \
    --layer_indices 12 \
    --components resid

# ── Phase D: DeepSeek full sweep ────────────────────────────────
echo ""
echo "==== PHASE D: DeepSeek TXCDR + Stacked + TFA (12 runs) ===="
for SHUF in "" "--shuffle-within-sequence"; do
    MODE="unshuffled"
    [ -n "$SHUF" ] && MODE="shuffled"
    echo ""
    echo "---- DeepSeek | resid_L12 | ${MODE} | $(date) ----"

    # Fast archs first
    $PYTHON -u -m src.pipeline.sweep \
        --dataset-type cached_activations \
        --model-name deepseek-r1-distill-llama-8b \
        --cached-dataset fineweb \
        --cached-layer-key resid_L12 \
        --models stacked_sae crosscoder \
        --k 50 100 --T 5 --steps $STEPS_TXCDR \
        --expansion-factor 4 \
        --results-dir "${RESULTS_DIR}/deepseek" \
        $SHUF

    # TFA
    $PYTHON -u -m src.pipeline.sweep \
        --dataset-type cached_activations \
        --model-name deepseek-r1-distill-llama-8b \
        --cached-dataset fineweb \
        --cached-layer-key resid_L12 \
        --models tfa_pos \
        --k 50 100 --T 5 --steps $STEPS_TFA \
        --expansion-factor 4 \
        --tfa-bottleneck-factor 8 \
        --tfa-batch-size 32 \
        --results-dir "${RESULTS_DIR}/deepseek" \
        $SHUF
done

echo ""
echo "============================================================"
echo "  FULL SWEEP DONE — $(date)"
echo "============================================================"
