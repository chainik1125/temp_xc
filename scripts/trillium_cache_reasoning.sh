#!/bin/bash
# trillium_cache_reasoning.sh — Cache reasoning traces from a thinking model
# on GSM8K or MATH500. Two profiles:
#
#   MODE=smoke : 100 seqs, 512 tok, 1 GPU, 1h wall. Run first to validate.
#   MODE=full  : 1000 seqs, 1024 tok, 1 GPU, 8h wall. Run after smoke passes.
#
# Usage:
#   bash scripts/trillium_cache_reasoning.sh                    # smoke deepseek gsm8k
#   MODE=full bash scripts/trillium_cache_reasoning.sh
#   MODE=smoke MODEL=deepseek-r1-distill-llama-8b DATASET=math500 bash scripts/trillium_cache_reasoning.sh
#
# Smoke is interactive (srun --pty). Full is submitted via sbatch and returns
# the job id; tail with `squeue -u $USER` and `tail -f logs/slurm/<id>.out`.
set -euo pipefail

MODE="${MODE:-smoke}"
MODEL="${MODEL:-deepseek-r1-distill-llama-8b}"
DATASET="${DATASET:-gsm8k}"
LAYERS="${LAYERS:-12 24}"

case "$MODE" in
  smoke)
    NUM_SEQS=100
    MAX_NEW=512
    WALL=01:00:00
    ;;
  full)
    NUM_SEQS=1000
    MAX_NEW=1024
    WALL=08:00:00
    ;;
  *)
    echo "Unknown MODE=$MODE  (use smoke|full)"; exit 1 ;;
esac

cd "$SCRATCH/temp_xc"
mkdir -p logs/slurm

CMD="python scripts/cache_reasoning_traces.py \
  --model $MODEL --dataset $DATASET \
  --num-sequences $NUM_SEQS --gen_max_new_tokens $MAX_NEW \
  --layer_indices $LAYERS"

if [ "$MODE" = "smoke" ]; then
    echo "=== smoke cache: $MODEL / $DATASET / $NUM_SEQS seqs ==="
    srun --account=rrg-aspuru --nodes=1 --gpus-per-node=1 --cpus-per-task=4 \
         --time=$WALL --job-name=txc-cache-smoke --pty bash -lc "
source \$SCRATCH/temp_xc/scripts/trillium_activate.sh
$CMD
"
else
    JOB=$(mktemp logs/slurm/cache-full-XXXX.sh)
    cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --partition=compute_full_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=$WALL
#SBATCH --job-name=txc-cache-full
#SBATCH --output=logs/slurm/cache-full_%j.out
#SBATCH --error=logs/slurm/cache-full_%j.err
set -euo pipefail
source "\$SCRATCH/temp_xc/scripts/trillium_activate.sh"
export CUDA_VISIBLE_DEVICES=0
$CMD
EOF
    sbatch "$JOB"
    echo "Submitted full cache job. squeue -u \$USER to watch."
fi
