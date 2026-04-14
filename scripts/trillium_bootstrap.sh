#!/bin/bash
# trillium_bootstrap.sh — Zero-to-first-real-run bootstrap. Run once per
# Trillium account after filling in ~/.txc_secrets.env. Idempotent.
#
# Steps:
#   1. env setup (installs deps, creates venv)
#   2. download all registry models (~40 GB)
#   3. allocate 1 GPU for 30 min, verify DeepSeek fits + smoke cache GSM8K
#      100 seqs, 512 tok
#   4. spotcheck the resulting acts
#
# Usage:
#   bash scripts/trillium_bootstrap.sh
set -euo pipefail

cd "$SCRATCH/temp_xc" 2>/dev/null || {
    echo "expected repo at \$SCRATCH/temp_xc — clone it first"
    exit 1
}

git pull origin aniket

echo ""
echo "=== [1/4] env setup ==="
bash scripts/trillium_setup.sh

if [ ! -s "$HOME/.txc_secrets.env" ] || ! grep -q 'HF_TOKEN="hf' "$HOME/.txc_secrets.env"; then
    echo ""
    echo "STOP: ~/.txc_secrets.env is empty or missing HF_TOKEN."
    echo "Fill it in, then rerun this bootstrap script."
    exit 1
fi

echo ""
echo "=== [2/4] download models ==="
bash scripts/trillium_download_models.sh

echo ""
echo "=== [3/4] 1-GPU allocation: verify fit + smoke cache ==="
srun --account=rrg-aspuru --nodes=1 --gpus-per-node=1 --cpus-per-task=4 \
     --time=01:00:00 --job-name=txc-bootstrap --pty bash -lc '
set -euo pipefail
source $SCRATCH/temp_xc/scripts/trillium_activate.sh
python scripts/verify_gpu_fit.py --model deepseek-r1-distill-llama-8b
python scripts/cache_reasoning_traces.py \
    --model deepseek-r1-distill-llama-8b \
    --dataset gsm8k --num-sequences 100 \
    --gen_max_new_tokens 512 --layer_indices 12 24
'

echo ""
echo "=== [4/4] spotcheck ==="
bash scripts/trillium_spotcheck_acts.sh deepseek-r1-distill-llama-8b gsm8k resid_L12

echo ""
echo "=== bootstrap complete ==="
echo "Next: MODE=full bash scripts/trillium_cache_reasoning.sh"
