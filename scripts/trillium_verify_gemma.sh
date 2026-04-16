#!/bin/bash
# trillium_verify_gemma.sh — Sbatch a single-GPU verify_gpu_fit run for
# gemma-2-2b to get the actual peak GPU memory on a compute node.
# Login node OOMs because its GPU is shared — use a compute allocation.
#
# Usage:
#   bash scripts/trillium_verify_gemma.sh
#
# Then once the job finishes:
#   cat logs/slurm/verify-gemma_*.out

set -euo pipefail

cd "$SCRATCH/temp_xc"
mkdir -p logs/slurm

JOB=$(mktemp "logs/slurm/verify-gemma-XXXX.sh")
cat > "$JOB" <<'EOF'
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=verify-gemma
#SBATCH --output=logs/slurm/verify-gemma_%j.out
#SBATCH --error=logs/slurm/verify-gemma_%j.err
set -euo pipefail
source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"
cd "$SCRATCH/temp_xc"
python scripts/verify_gpu_fit.py --model gemma-2-2b
EOF

JOB_ID=$(sbatch --parsable "$JOB")
echo ""
echo "=== verify-gemma sbatch submitted ==="
echo "  job:   $JOB_ID"
echo "  watch: squeue -u \$USER"
echo "  read:  cat logs/slurm/verify-gemma_${JOB_ID}.out"
