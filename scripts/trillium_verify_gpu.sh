#!/bin/bash
# trillium_verify_gpu.sh — Allocate a single GPU and run the fit check for a
# given registered model. Blocks until done, streams output.
#
#   bash scripts/trillium_verify_gpu.sh                                 # default: deepseek-r1-distill-llama-8b
#   bash scripts/trillium_verify_gpu.sh gemma-2-2b
#   bash scripts/trillium_verify_gpu.sh llama-3.1-8b 1024 8             # model, seq_len, batch_size
set -euo pipefail

MODEL="${1:-deepseek-r1-distill-llama-8b}"
SEQ_LEN="${2:-512}"
BATCH="${3:-4}"

cd "$SCRATCH/temp_xc"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"
    python scripts/verify_gpu_fit.py --model "$MODEL" --seq-len "$SEQ_LEN" --batch-size "$BATCH"
    exit 0
fi

srun --account=rrg-aspuru --nodes=1 --gpus-per-node=1 --cpus-per-task=4 \
     --time=00:30:00 --job-name=txc-verify --pty bash -lc "
source \$SCRATCH/temp_xc/scripts/trillium_activate.sh
python scripts/verify_gpu_fit.py --model '$MODEL' --seq-len $SEQ_LEN --batch-size $BATCH
"
