#!/bin/bash
# fetch_from_trillium.sh — Pull sprint artifacts from Trillium to the local
# checkout. Runs on your LAPTOP, not on Trillium.
#
# By default pulls:
#   - reports/           (markdown + PNGs for Slack posting)
#   - results/nlp/       (per-run JSONs from real-LM sweeps)
#   - logs/slurm/        (SLURM .out and .err, for debugging)
#
# Does NOT pull:
#   - data/cached_activations/   (too big, stays pinned on $SCRATCH)
#   - $HF_HOME model weights     (too big, stays pinned on $SCRATCH)
#
# Usage:
#   bash scripts/fetch_from_trillium.sh                   # default: all 3
#   bash scripts/fetch_from_trillium.sh reports           # only reports
#   bash scripts/fetch_from_trillium.sh results logs      # subset
#   bash scripts/fetch_from_trillium.sh --dry-run         # preview
#
# Env overrides:
#   TRILLIUM_USER      (default: aniketrd)
#   TRILLIUM_HOST      (default: trillium-gpu.scinet.utoronto.ca)
#   TRILLIUM_REPO      (default: /scratch/aniketrd/temp_xc)

set -euo pipefail

USER="${TRILLIUM_USER:-aniketrd}"
HOST="${TRILLIUM_HOST:-trillium-gpu.scinet.utoronto.ca}"
REMOTE="${TRILLIUM_REPO:-/scratch/aniketrd/temp_xc}"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

DRY=""
TARGETS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY="--dry-run" ;;
        reports|results|logs) TARGETS+=("$arg") ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=(reports results logs)
fi

RSYNC_OPTS=(-avz --progress --human-readable $DRY)

for t in "${TARGETS[@]}"; do
    case "$t" in
        reports)
            echo ""
            echo ">> reports/"
            mkdir -p "$LOCAL/reports"
            rsync "${RSYNC_OPTS[@]}" \
                "$USER@$HOST:$REMOTE/reports/" "$LOCAL/reports/"
            ;;
        results)
            echo ""
            echo ">> results/nlp/"
            mkdir -p "$LOCAL/results/nlp"
            rsync "${RSYNC_OPTS[@]}" \
                --include='*/' --include='*.json' --exclude='*' \
                "$USER@$HOST:$REMOTE/results/nlp/" "$LOCAL/results/nlp/"
            ;;
        logs)
            echo ""
            echo ">> logs/slurm/"
            mkdir -p "$LOCAL/logs/slurm"
            rsync "${RSYNC_OPTS[@]}" \
                --include='*/' --include='*.out' --include='*.err' --exclude='*' \
                "$USER@$HOST:$REMOTE/logs/slurm/" "$LOCAL/logs/slurm/"
            ;;
    esac
done

echo ""
echo "=== Done ==="
echo "Reports: $LOCAL/reports/"
echo "Results: $LOCAL/results/nlp/"
echo "Logs:    $LOCAL/logs/slurm/"
echo ""
echo "To re-aggregate locally (uses the pulled JSONs):"
echo "  python scripts/aggregate_results.py --root results/nlp --out reports/latest"
