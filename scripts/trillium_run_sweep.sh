#!/bin/bash
# trillium_run_sweep.sh — The actual SLURM job script.
# Submitted by trillium_sweep.sh — do not run directly.
set -euo pipefail

module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/txc/bin/activate"
cd "$SCRATCH/temp_xc"

export PYTHONUNBUFFERED=1
# Use only one GPU (the sweep is single-GPU, Trillium requires 4 per node)
export CUDA_VISIBLE_DEVICES=0

echo "=== Bench Sweep ==="
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date: $(date)"
echo ""

# Full sweep: 7 models x 4 k-values x 3 rho-values = 84 jobs @ 30k steps
python3 -m src.bench.sweep \
  --results-dir results/bench \
  --steps 30000

echo ""
echo "=== Done: $(date) ==="
