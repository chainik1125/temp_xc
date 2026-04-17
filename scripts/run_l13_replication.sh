#!/bin/bash
# Replication sweep: resid_L13 on Gemma-2-2B-IT, k=50, unshuffled.
# Trains Stacked + Crosscoder + TFA-pos on already-cached L13 activations,
# then re-runs the full autointerp analysis pipeline against L13.
#
# Expected runtime on A40: 2-3h training + ~20 min analysis.
# Set SKIP_TRAINING=1 to reuse existing ckpts and only re-run analysis.

set -euo pipefail

export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

PYTHON="${PYTHON:-/workspace/temp_xc/.venv/bin/python}"
RESULTS_DIR="results/nlp_sweep"
CKPT_DIR="${RESULTS_DIR}/gemma/ckpts"
SCAN_DIR="${RESULTS_DIR}/gemma/scans"
FIG_DIR="${RESULTS_DIR}/gemma/figures"
LAYER="resid_L13"
K=50
MODEL="gemma-2-2b-it"

mkdir -p "$CKPT_DIR" "$SCAN_DIR" "$FIG_DIR" logs

echo "============================================================"
echo "  L13 REPLICATION — $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

# ── PHASE 1: training ──────────────────────────────────────────
if [ "${SKIP_TRAINING:-0}" != "1" ]; then
    echo ""
    echo "==== TRAINING — $(date) ===="
    $PYTHON -u -m src.bench.sweep \
        --dataset-type cached_activations \
        --model-name "$MODEL" \
        --cached-dataset fineweb \
        --cached-layer-key "$LAYER" \
        --models tfa_pos stacked_sae crosscoder \
        --k $K \
        --T 5 \
        --steps 10000 \
        --expansion-factor 8 \
        --tfa-bottleneck-factor 8 \
        --tfa-batch-size 16 \
        --results-dir "${RESULTS_DIR}/gemma"
else
    echo "SKIP_TRAINING=1 — reusing existing ckpts"
fi

# ── PHASE 2: analysis pipeline ─────────────────────────────────
echo ""
echo "==== SCAN FEATURES — $(date) ===="
$PYTHON -u -m temporal_crosscoders.NLP.scan_features \
    --arches stacked_sae crosscoder tfa_pos tfa_pos_pred \
    --subject-model "$MODEL" \
    --cached-dataset fineweb \
    --layer-key "$LAYER" \
    --k $K --T 5 \
    --sample-chains 1000 \
    --top-features 300 \
    --top-k 10 \
    --ckpt-dir "$CKPT_DIR" \
    --out-dir "$SCAN_DIR"

echo ""
echo "==== TEMPORAL SPREAD — $(date) ===="
for ARCH in stacked_sae tfa_pos tfa_pos_pred; do
    CKPT="$CKPT_DIR/${ARCH%_pred}__${MODEL}__fineweb__${LAYER}__k${K}__seed42.pt"
    if [ "$ARCH" = "tfa_pos_pred" ]; then
        CKPT="$CKPT_DIR/tfa_pos__${MODEL}__fineweb__${LAYER}__k${K}__seed42.pt"
    fi
    MODEL_TYPE="$ARCH"
    $PYTHON -u -m temporal_crosscoders.NLP.temporal_spread \
        --scan "$SCAN_DIR/scan__${ARCH}__${LAYER}__k${K}.json" \
        --model-type "$MODEL_TYPE" \
        --ckpt "$CKPT" \
        --subject-model "$MODEL" \
        --cached-dataset fineweb \
        --layer-key "$LAYER" \
        --k $K --T 5 \
        --out "$SCAN_DIR/tspread__${ARCH}__${LAYER}__k${K}.json"
done

echo ""
echo "==== TFA PRED/NOVEL SPLIT — $(date) ===="
$PYTHON -u -m temporal_crosscoders.NLP.tfa_pred_novel_split \
    --scan "$SCAN_DIR/scan__tfa_pos__${LAYER}__k${K}.json" \
    --ckpt "$CKPT_DIR/tfa_pos__${MODEL}__fineweb__${LAYER}__k${K}__seed42.pt" \
    --subject-model "$MODEL" \
    --cached-dataset fineweb \
    --layer-key "$LAYER" \
    --k $K --T 5 \
    --out "$SCAN_DIR/tfa_pred_novel__${LAYER}__k${K}.json"

echo ""
echo "==== FEATURE MATCH (decoder cosine) — $(date) ===="
$PYTHON -u -m temporal_crosscoders.NLP.feature_match \
    --archs stacked_sae crosscoder tfa_pos \
    --ckpts \
        "$CKPT_DIR/stacked_sae__${MODEL}__fineweb__${LAYER}__k${K}__seed42.pt" \
        "$CKPT_DIR/crosscoder__${MODEL}__fineweb__${LAYER}__k${K}__seed42.pt" \
        "$CKPT_DIR/tfa_pos__${MODEL}__fineweb__${LAYER}__k${K}__seed42.pt" \
    --subject-model "$MODEL" \
    --k $K --T 5 \
    --out "$SCAN_DIR/feature_match__${LAYER}__k${K}.json"

echo ""
echo "==== HIGH-SPAN COMPARISON — $(date) ===="
$PYTHON -u -m temporal_crosscoders.NLP.high_span_comparison \
    --layer-key "$LAYER" --k $K --fig-suffix "_L13"

echo ""
echo "==== CONTENT-BASED MATCHING — $(date) ===="
$PYTHON -u -m temporal_crosscoders.NLP.content_based_match \
    --scan-dir "$SCAN_DIR" \
    --out-dir "$FIG_DIR" \
    --layer-key "$LAYER" --k $K
mv "$FIG_DIR/content_based_match.png" "$FIG_DIR/content_based_match_L13.png" 2>/dev/null || true
mv "$FIG_DIR/content_based_match.doc.png" "$FIG_DIR/content_based_match_L13.doc.png" 2>/dev/null || true
mv "$FIG_DIR/content_based_match.thumb.png" "$FIG_DIR/content_based_match_L13.thumb.png" 2>/dev/null || true

echo ""
echo "==== EXPLAIN FEATURES (Claude Haiku) — $(date) ===="
export ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key)
for ARCH in stacked_sae crosscoder tfa_pos tfa_pos_pred; do
    $PYTHON -u -m temporal_crosscoders.NLP.explain_features \
        --scan "$SCAN_DIR/scan__${ARCH}__${LAYER}__k${K}.json" \
        --out  "$SCAN_DIR/labels__${ARCH}__${LAYER}__k${K}.json" \
        --top-features 50 --concurrency 2
done

echo ""
echo "==== SUMMARY PLOTS — $(date) ===="
$PYTHON -u -m temporal_crosscoders.NLP.plot_autointerp_summary \
    --layer-key "$LAYER" --k $K --fig-suffix "_L13"

echo ""
echo "============================================================"
echo "  L13 REPLICATION COMPLETE — $(date)"
echo "  Artifacts:"
echo "    scans:   $SCAN_DIR/*__${LAYER}__k${K}.json"
echo "    figures: $FIG_DIR/*_L13.*"
echo "============================================================"
