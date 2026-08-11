#!/bin/bash
#SBATCH --account=cis240096p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH --cpus-per-task=4
#SBATCH --time=5:30:00
#SBATCH --job-name=dtxc-gemma_recon_s2
#SBATCH --output=/ocean/projects/cis240096p/manningc/dtxc/logs/%x-%j.out
set -euo pipefail
source $HOME/.dtxc_env
export HF_HOME=/ocean/projects/cis240096p/manningc/dtxc/hf_home
export PYTHONUNBUFFERED=1
# shared venv, mkdir-mutex bootstrap
VENV=/ocean/projects/cis240096p/manningc/dtxc/venv
if mkdir "$VENV.lock" 2>/dev/null; then
  if [ ! -f "$VENV/ready" ]; then
    /usr/bin/python3.11 -m venv "$VENV"
    "$VENV/bin/pip" install -q torch numpy transformers==4.46.2 datasets==3.1.0 zstandard sentencepiece accelerate huggingface_hub
    touch "$VENV/ready"
  fi
  rmdir "$VENV.lock"
else
  for i in $(seq 1 240); do [ -f "$VENV/ready" ] && break; sleep 10; done
  [ -f "$VENV/ready" ] || { echo "venv never became ready"; exit 1; }
fi
RUN=/ocean/projects/cis240096p/manningc/dtxc/runs/gemma_recon_s2
mkdir -p "$RUN"
"$VENV/bin/python" /ocean/projects/cis240096p/manningc/dtxc/psc_train_sae.py \
  --model google/gemma-2-2b --hook resid12 --arm recon --seed 2 \
  --steps 24000 --k 40 --dataset pile \
  --out "$RUN" --hf-repo dmanningcoe/diffusion-topk-saes --hf-subdir gemma2-2b-l12-100M
