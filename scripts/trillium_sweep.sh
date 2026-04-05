#!/bin/bash
# trillium_sweep.sh — Submit the bench sweep as a SLURM job on Trillium.
# Run from the Trillium login node: bash scripts/trillium_sweep.sh
set -euo pipefail

PARTITION="compute_full_node"
ACCOUNT="rrg-aspuru"
JOB_NAME="bench-sweep"
TIME="04:00:00"
REPO_DIR="$SCRATCH/temp_xc"
LOG_DIR="$REPO_DIR/logs/slurm"

mkdir -p "$LOG_DIR"

# Pull latest code
cd "$REPO_DIR"
git pull origin aniket

sbatch \
  --partition="${PARTITION}" \
  --gpus-per-node=4 \
  --cpus-per-task=8 \
  --time="${TIME}" \
  --account="${ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
  --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
  scripts/trillium_run_sweep.sh

echo ""
echo "Submitted. Monitor with: squeue -u \$USER"
echo "Logs: ${LOG_DIR}/"
