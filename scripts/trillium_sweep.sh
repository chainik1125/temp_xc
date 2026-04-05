#!/bin/bash
# trillium_sweep.sh — Pull latest code and submit the sweep job.
# Run from Trillium login node: bash scripts/trillium_sweep.sh
set -euo pipefail

cd "$SCRATCH/temp_xc"
git pull origin aniket
mkdir -p logs/slurm

sbatch scripts/trillium_run_sweep.sh

echo ""
echo "Submitted. Monitor with: squeue -u \$USER"
echo "Logs: logs/slurm/"
