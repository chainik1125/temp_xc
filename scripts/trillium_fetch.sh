#!/bin/bash
# trillium_fetch.sh — Pull results and logs from Trillium to local.
# Run locally: bash scripts/trillium_fetch.sh
set -euo pipefail

REMOTE="aniketrd@trillium-gpu.scinet.utoronto.ca"
REMOTE_DIR="\$SCRATCH/temp_xc"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Fetching results from Trillium..."
rsync -avz "${REMOTE}:${REMOTE_DIR}/results/bench/" "${LOCAL_DIR}/results/bench/" 2>/dev/null || true
rsync -avz "${REMOTE}:${REMOTE_DIR}/results/coupled_validation/" "${LOCAL_DIR}/results/coupled_validation/" 2>/dev/null || true

echo ""
echo "Fetching SLURM logs..."
rsync -avz "${REMOTE}:${REMOTE_DIR}/logs/slurm/" "${LOCAL_DIR}/logs/slurm/" 2>/dev/null || true
rsync -avz "${REMOTE}:${REMOTE_DIR}/slurm-*.out" "${LOCAL_DIR}/logs/slurm/" 2>/dev/null || true

echo ""
echo "Done. Results in: ${LOCAL_DIR}/results/bench/"
