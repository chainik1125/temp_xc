#!/bin/bash
# runpod_verify_gpu.sh — Load a registered subject model in its declared
# dtype, run one forward pass, report peak GPU memory.
#
# RunPod has a single GPU per pod by default — no SLURM, no allocation
# dance. Just runs.
#
#   bash scripts/runpod_verify_gpu.sh                   # default: deepseek-r1-distill-llama-8b
#   bash scripts/runpod_verify_gpu.sh gemma-2-2b
#   bash scripts/runpod_verify_gpu.sh llama-3.1-8b 1024 8

set -euo pipefail

MODEL="${1:-deepseek-r1-distill-llama-8b}"
SEQ_LEN="${2:-512}"
BATCH="${3:-4}"

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

python scripts/verify_gpu_fit.py --model "$MODEL" --seq-len "$SEQ_LEN" --batch-size "$BATCH"
