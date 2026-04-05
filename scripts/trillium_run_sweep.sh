#!/bin/bash
#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --account=rrg-aspuru
#SBATCH --job-name=bench-sweep
#SBATCH --output=logs/slurm/bench-sweep_%j.out
#SBATCH --error=logs/slurm/bench-sweep_%j.err

# trillium_run_sweep.sh — SLURM job script for the full bench sweep.
# Submit with: sbatch scripts/trillium_run_sweep.sh
set -euo pipefail

module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/txc/bin/activate"
cd "$SCRATCH/temp_xc"

export PYTHONUNBUFFERED=1
# Use only one GPU (the sweep is single-GPU, Trillium requires 4 per node)
# Trillium requires 4 GPUs per node but we only need one
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
