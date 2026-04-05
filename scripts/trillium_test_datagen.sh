#!/bin/bash
# trillium_test_datagen.sh — Run data generation smoke tests as a SLURM job.
# Submitted by: sbatch scripts/trillium_test_datagen.sh
# Or via: bash scripts/trillium_submit_tests.sh
set -euo pipefail

#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=rrg-aspuru
#SBATCH --job-name=test-datagen
#SBATCH --output=logs/slurm/test-datagen_%j.out
#SBATCH --error=logs/slurm/test-datagen_%j.err

module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a
source "$HOME/envs/txc/bin/activate"
cd "$SCRATCH/temp_xc"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "=== Data Generation Smoke Tests ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

python3 src/data_generation/test.py

echo ""
echo "=== Bench Module Tests ==="
python3 -m pytest tests/bench/ -v

echo ""
echo "=== Done: $(date) ==="
