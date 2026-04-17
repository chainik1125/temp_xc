#!/bin/bash
# tmp_finish_step3.sh — ONE-OFF recovery script for the partial step3 run
# that got Ctrl+C'd during TXCDR unshuffled training. Re-uses existing
# TopKSAE + Stacked unshuffled checkpoints (already on disk), trains
# only what's missing.
#
# Delete this script once step3 completes successfully and the feature
# maps are rendered — it's not part of the permanent sprint pipeline.
#
# Usage (from inside tmux so it survives disconnect):
#   tmux new -s step3
#   bash scripts/tmp_finish_step3.sh 2>&1 | tee logs/step3-finish-$(date +%Y%m%d-%H%M).log
#   # Ctrl+b, d to detach

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

MODEL="gemma-2-2b"
DATASET="stack-python"
LAYER="resid_L13"
K=100
T=5
STEPS=10000

REPORT_DIR="reports/step3-gemma-stack"
mkdir -p "$REPORT_DIR"

UNSHUF_DIR="results/nlp/step3-unshuffled"
SHUF_DIR="results/nlp/step3-shuffled"

echo "=== step3 finish ==="
echo ""
echo "Existing checkpoints:"
ls -1 "$UNSHUF_DIR/ckpts/" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
ls -1 "$SHUF_DIR/ckpts/"   2>/dev/null | sed 's/^/  /' || echo "  (none)"
echo ""

# ─── 1. finish TXCDR unshuffled ──────────────────────────────────────────
XCDR_UNSHUF_CKPT="$UNSHUF_DIR/ckpts/crosscoder__${MODEL}__${DATASET}__${LAYER}__k${K}__seed42.pt"
if [ -f "$XCDR_UNSHUF_CKPT" ]; then
    echo ">> SKIP — TXCDR unshuffled already trained: $XCDR_UNSHUF_CKPT"
else
    echo ""
    echo "=== [1/4] step3 unshuffled — TXCDR only ==="
    python -m src.bench.sweep \
        --dataset-type cached_activations \
        --model-name "$MODEL" \
        --cached-dataset "$DATASET" \
        --cached-layer-key "$LAYER" \
        --models crosscoder \
        --k "$K" --T "$T" --steps "$STEPS" \
        --results-dir "$UNSHUF_DIR"
fi

# ─── 2. full shuffled sweep (all 3 archs) ────────────────────────────────
SHUF_TOPK="$SHUF_DIR/ckpts/topk_sae__${MODEL}__${DATASET}__${LAYER}__k${K}__seed42_shuffled.pt"
SHUF_STACKED="$SHUF_DIR/ckpts/stacked_sae__${MODEL}__${DATASET}__${LAYER}__k${K}__seed42_shuffled.pt"
SHUF_XCDR="$SHUF_DIR/ckpts/crosscoder__${MODEL}__${DATASET}__${LAYER}__k${K}__seed42_shuffled.pt"

NEED_SHUF_MODELS=()
[ ! -f "$SHUF_TOPK"    ] && NEED_SHUF_MODELS+=("topk_sae")
[ ! -f "$SHUF_STACKED" ] && NEED_SHUF_MODELS+=("stacked_sae")
[ ! -f "$SHUF_XCDR"    ] && NEED_SHUF_MODELS+=("crosscoder")

if [ ${#NEED_SHUF_MODELS[@]} -eq 0 ]; then
    echo ">> SKIP — all shuffled checkpoints already exist"
else
    echo ""
    echo "=== [2/4] step3 shuffled — ${NEED_SHUF_MODELS[*]} ==="
    python -m src.bench.sweep \
        --dataset-type cached_activations \
        --model-name "$MODEL" \
        --cached-dataset "$DATASET" \
        --cached-layer-key "$LAYER" \
        --models "${NEED_SHUF_MODELS[@]}" \
        --k "$K" --T "$T" --steps "$STEPS" \
        --results-dir "$SHUF_DIR" \
        --shuffle-within-sequence
fi

# ─── 3 & 4. feature maps on both crosscoder checkpoints ──────────────────
run_fmap() {
    local CKPT="$1"
    local LABEL="$2"
    if [ ! -f "$CKPT" ]; then
        echo ">> SKIP fmap — ckpt missing: $CKPT"
        return 0
    fi
    echo ""
    echo "=== [$LABEL] feature_map ==="
    python -m temporal_crosscoders.NLP.feature_map \
        --checkpoint "$CKPT" \
        --model crosscoder --subject-model "$MODEL" \
        --k "$K" --T "$T" \
        --label "$LABEL" \
        --output-dir "$REPORT_DIR" \
        --include-unlabeled --skip-llm-labels
}

run_fmap "$XCDR_UNSHUF_CKPT" step3-unshuffled
run_fmap "$SHUF_XCDR"        step3-shuffled

echo ""
echo "=== step3 finish done ==="
ls -lh "$REPORT_DIR"/feature_map_step3-*.png 2>/dev/null || true
echo ""
echo "Next (optional, on existing step1/2 checkpoints too):"
echo "  bash scripts/runpod_backfill_metrics.sh"
