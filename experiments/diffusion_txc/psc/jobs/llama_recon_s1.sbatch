#!/bin/bash
#SBATCH --account=cis240096p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --job-name=dtxc-llama_recon_s1
#SBATCH --output=/ocean/projects/cis240096p/manningc/dtxc/logs/%x-%j.out
set -euo pipefail
source $HOME/.dtxc_env
export HF_HOME=/ocean/projects/cis240096p/manningc/dtxc/hf_home
export PYTHONUNBUFFERED=1
# shared venv, mkdir-mutex bootstrap
VENV=/ocean/projects/cis240096p/manningc/dtxc/venv
if mkdir "$VENV.lock" 2>/dev/null; then
  if [ ! -f "$VENV/ready" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q torch numpy transformers==4.46.2 datasets==3.1.0 zstandard sentencepiece accelerate huggingface_hub
    touch "$VENV/ready"
  fi
  rmdir "$VENV.lock"
else
  for i in $(seq 1 240); do [ -f "$VENV/ready" ] && break; sleep 10; done
  [ -f "$VENV/ready" ] || { echo "venv never became ready"; exit 1; }
fi
RUN=/ocean/projects/cis240096p/manningc/dtxc/runs/llama_recon_s1
mkdir -p "$RUN"
"$VENV/bin/python" /ocean/projects/cis240096p/manningc/dtxc/psc_train_sae.py \
  --model NousResearch/Meta-Llama-3.1-8B --hook ln110 --arm recon --seed 1 \
  --steps 20000 --k 64 --dataset fineweb \
  --out "$RUN" --hf-repo dmanningcoe/diffusion-topk-saes --hf-subdir llama31-8b-ln1L10-20k
