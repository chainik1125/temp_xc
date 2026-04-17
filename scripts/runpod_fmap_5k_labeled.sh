#!/bin/bash
# runpod_fmap_5k_labeled.sh — One-off script for a dedicated relabelling
# pod. Re-renders the 4 crosscoder feature maps (step1+step2, un/shuf)
# WITH gemma-2-2b-it cluster summaries in the legend. Per-feature hover
# labels are pulled from the existing 5k autointerp JSONs.
#
# Difference from runpod_fmap_5k.sh: drops --skip-llm-labels, so gemma
# generates thematic cluster summaries instead of just "Cluster 0..19".
# Worth ~10-20 min of A40 time per checkpoint for paper-grade legends.
#
# Prerequisites (do these once after spinning up a fresh pod):
#   1. SSH in, clone repo to /workspace, git checkout aniket
#   2. bash scripts/runpod_setup.sh       (installs uv, writes .env)
#   3. Fill in .env with ANTHROPIC_API_KEY + HF_TOKEN, accept gemma-2-2b-it
#      license at https://huggingface.co/google/gemma-2-2b-it
#   4. source scripts/runpod_activate.sh
#
# Usage:
#   bash scripts/runpod_fmap_5k_labeled.sh
#
# Cost estimate:
#   rsync from Trillium:   5-10 min (bandwidth-bound, ~3 GB)
#   gemma-2-2b-it download: 1-2 min (~5 GB from HF)
#   4 × fmap with labeling: 10-20 min (80 gemma calls per checkpoint)
#   Total: ~30-45 min on A40 = ~$0.35 at $0.44/hr

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

if [ -z "${HF_TOKEN:-}" ]; then
    echo "FAIL: HF_TOKEN not set — needed for gemma-2-2b-it download."
    exit 1
fi

TRILLIUM_USER="${TRILLIUM_USER:-aniketrd}"
TRILLIUM_HOST="${TRILLIUM_HOST:-trillium-gpu.scinet.utoronto.ca}"
TRILLIUM_BASE="${TRILLIUM_BASE:-/scratch/aniketrd/temp_xc}"

# ─── 1. rsync feat_*.json + crosscoder checkpoints from Trillium ────────
echo "=== [1/3] rsync from Trillium ==="

for subdir in reports/step1-gemma-replication reports/step2-deepseek-reasoning; do
    mkdir -p "$subdir"
    echo ""
    echo ">> $subdir/autointerp/"
    rsync -avzP --prune-empty-dirs \
        --include='*/' \
        --include='autointerp/**' \
        --include='feat_*.json' \
        --include='summary.json' \
        --exclude='*' \
        "${TRILLIUM_USER}@${TRILLIUM_HOST}:${TRILLIUM_BASE}/${subdir}/" \
        "${subdir}/"
done

echo ""
echo ">> results/nlp/*/ckpts/crosscoder__*.pt"
mkdir -p results/nlp
rsync -avzP \
    --include='*/' \
    --include='crosscoder__*.pt' \
    --exclude='*' \
    "${TRILLIUM_USER}@${TRILLIUM_HOST}:${TRILLIUM_BASE}/results/nlp/" \
    results/nlp/

# ─── 2. render all 4 feature maps WITH cluster labels ───────────────────
echo ""
echo "=== [2/3] rendering 4 feature maps (gemma-2-2b-it cluster labels) ==="

run_fmap_labeled() {
    local TAG="$1"          # e.g. step1-unshuffled
    local MODEL="$2"
    local DATASET="$3"
    local LAYER="$4"
    local SHUF_SUFFIX="$5"
    local REPORT_DIR="$6"

    local CKPT="results/nlp/${TAG}/ckpts/crosscoder__${MODEL}__${DATASET}__${LAYER}__k100__seed42${SHUF_SUFFIX}.pt"
    if [ ! -f "$CKPT" ]; then
        echo ">> SKIP $TAG — ckpt not found: $CKPT"
        return 0
    fi

    echo ""
    echo ">> [$TAG] feature_map with gemma-2-2b-it cluster labels"
    python -m temporal_crosscoders.NLP.feature_map \
        --checkpoint "$CKPT" \
        --model crosscoder --subject-model "$MODEL" \
        --k 100 --T 5 \
        --label "$TAG" \
        --output-dir "$REPORT_DIR" \
        --include-unlabeled \
        --explain-model google/gemma-2-2b-it \
        --explain-device cuda:0
}

run_fmap_labeled step1-unshuffled gemma-2-2b                     fineweb  resid_L13 "" \
    reports/step1-gemma-replication
run_fmap_labeled step1-shuffled   gemma-2-2b                     fineweb  resid_L13 "_shuffled" \
    reports/step1-gemma-replication
run_fmap_labeled step2-unshuffled deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "" \
    reports/step2-deepseek-reasoning
run_fmap_labeled step2-shuffled   deepseek-r1-distill-llama-8b   gsm8k    resid_L12 "_shuffled" \
    reports/step2-deepseek-reasoning

# ─── 3. summary ──────────────────────────────────────────────────────────
echo ""
echo "=== [3/3] outputs ==="
echo ""
ls -lh reports/step1-gemma-replication/feature_map_*.png  reports/step1-gemma-replication/feature_map_*.html \
       reports/step2-deepseek-reasoning/feature_map_*.png reports/step2-deepseek-reasoning/feature_map_*.html \
       2>/dev/null | awk '{print "  " $5 "  " $9}'

echo ""
echo "Done. To pull these back to your laptop:"
echo "  On your laptop, from the repo root:"
echo "    bash scripts/fetch_runpod_fmap.sh 'root@<pod-ip> -p <port>'"
echo ""
echo "Then destroy this pod from the RunPod UI."
