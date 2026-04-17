#!/bin/bash
# fetch_trillium_results.sh — Pull the Trillium sprint outputs down to
# the laptop's local repo, organized to match the on-Trillium layout.
#
# What gets pulled:
#   reports/step1-gemma-replication/   — 5k labeled PNGs + HTMLs + feat_*.json
#   reports/step2-deepseek-reasoning/  — same for DeepSeek+GSM8K
#   reports/sae-control-deepseek/      — TopKSAE geometric gate PNGs
#   results/nlp/step1-*/results_*.json — sweep metric JSONs
#   results/nlp/step2-*/results_*.json — same
#
# What gets SKIPPED:
#   results/nlp/step*/ckpts/*.pt       — large (~0.5-1 GB each); stays on
#                                        Trillium. Fetch individually if
#                                        you need a specific one locally.
#   data/cached_activations/           — tens of GB; regenerate on RunPod
#                                        instead.
#
# Usage (from laptop, inside the repo root):
#   bash scripts/fetch_trillium_results.sh
#
# Re-running is safe — rsync only transfers changed files.

set -euo pipefail

REMOTE_USER="aniketrd"
REMOTE_HOST="trillium-gpu.scinet.utoronto.ca"
REMOTE_BASE="/scratch/aniketrd/temp_xc"

cd "$(git rev-parse --show-toplevel)"

echo "=== fetching Trillium results → $(pwd) ==="
echo ""

# ─── reports/ — labeled feature maps, autointerp JSONs, gate plots ──────
echo ">> reports/ (feature maps + autointerp)"
rsync -avzP --delete-after \
    --exclude='*.tmp' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/reports/" \
    reports/

# ─── results/nlp/ — sweep JSONs only, skip checkpoints ─────────────────
echo ""
echo ">> results/nlp/ (sweep JSONs, skipping .pt checkpoints)"
mkdir -p results/nlp
rsync -avzP \
    --include='*/' \
    --include='*.json' \
    --include='*.jsonl' \
    --exclude='ckpts/***' \
    --exclude='*.pt' \
    --exclude='*' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/results/nlp/" \
    results/nlp/

echo ""
echo "=== done ==="
echo ""
echo "Summary of what landed:"
for d in reports/step1-gemma-replication reports/step2-deepseek-reasoning reports/sae-control-deepseek; do
    if [ -d "$d" ]; then
        png_count=$(find "$d" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
        html_count=$(find "$d" -name "*.html" 2>/dev/null | wc -l | tr -d ' ')
        feat_count=$(find "$d" -path '*/autointerp/*' -name 'feat_*.json' 2>/dev/null | wc -l | tr -d ' ')
        echo "  $d: ${png_count} PNGs, ${html_count} HTMLs, ${feat_count} feat_*.json"
    fi
done
echo ""
if [ -d results/nlp ]; then
    json_count=$(find results/nlp -name 'results_*.json' 2>/dev/null | wc -l | tr -d ' ')
    echo "  results/nlp/: ${json_count} sweep result JSONs"
fi
echo ""
echo "Next:"
echo "  open reports/step1-gemma-replication/feature_map_step1-unshuffled.html"
echo "  open reports/step2-deepseek-reasoning/feature_map_step2-unshuffled.html"
