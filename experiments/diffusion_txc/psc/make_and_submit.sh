#!/usr/bin/env bash
# Generate sbatch files for the 14 overnight SAE jobs, ship them to PSC, submit.
# Run locally: bash make_and_submit.sh
set -euo pipefail

SOCK="$HOME/.ssh/cm/psc"
HOST="manningc@bridges2.psc.edu"
PROJ="/ocean/projects/cis240096p/manningc"
BASE="$PROJ/dtxc"
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS="$HERE/jobs"
mkdir -p "$JOBS"

# job spec: name model hook dataset k steps walltime
SPECS=(
  "gemma_recon_s0   google/gemma-2-2b                recon      0 resid12 pile    40 24000 12:00:00 gemma2-2b-l12-100M"
  "gemma_recon_s1   google/gemma-2-2b                recon      1 resid12 pile    40 24000 12:00:00 gemma2-2b-l12-100M"
  "gemma_dsm_s0     google/gemma-2-2b                dsm        0 resid12 pile    40 24000 12:00:00 gemma2-2b-l12-100M"
  "gemma_dsm_s1     google/gemma-2-2b                dsm        1 resid12 pile    40 24000 12:00:00 gemma2-2b-l12-100M"
  "gemma_dsmann_s0  google/gemma-2-2b                dsm_anneal 0 resid12 pile    40 24000 12:00:00 gemma2-2b-l12-100M"
  "gemma_dsmann_s1  google/gemma-2-2b                dsm_anneal 1 resid12 pile    40 24000 12:00:00 gemma2-2b-l12-100M"
  "llama_recon_s0   NousResearch/Meta-Llama-3.1-8B   recon      0 ln110   fineweb 64 20000 20:00:00 llama31-8b-ln1L10-20k"
  "llama_recon_s1   NousResearch/Meta-Llama-3.1-8B   recon      1 ln110   fineweb 64 20000 20:00:00 llama31-8b-ln1L10-20k"
  "llama_dsm_s0     NousResearch/Meta-Llama-3.1-8B   dsm        0 ln110   fineweb 64 20000 20:00:00 llama31-8b-ln1L10-20k"
  "llama_dsm_s1     NousResearch/Meta-Llama-3.1-8B   dsm        1 ln110   fineweb 64 20000 20:00:00 llama31-8b-ln1L10-20k"
  "qwen_recon_s0    Qwen/Qwen2.5-14B-Instruct        recon      0 ln124   fineweb 64 12000 30:00:00 qwen25-14b-ln1L24-12k"
  "qwen_recon_s1    Qwen/Qwen2.5-14B-Instruct        recon      1 ln124   fineweb 64 12000 30:00:00 qwen25-14b-ln1L24-12k"
  "qwen_dsm_s0      Qwen/Qwen2.5-14B-Instruct        dsm        0 ln124   fineweb 64 12000 30:00:00 qwen25-14b-ln1L24-12k"
  "qwen_dsm_s1     Qwen/Qwen2.5-14B-Instruct         dsm        1 ln124   fineweb 64 12000 30:00:00 qwen25-14b-ln1L24-12k"
)

for spec in "${SPECS[@]}"; do
  read -r name model arm seed hook dataset k steps wall subdir <<<"$spec"
  cat > "$JOBS/$name.sbatch" <<EOF
#!/bin/bash
#SBATCH --account=cis240096p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH --cpus-per-task=4
#SBATCH --time=$wall
#SBATCH --job-name=dtxc-$name
#SBATCH --output=$BASE/logs/%x-%j.out
set -euo pipefail
source \$HOME/.dtxc_env
export HF_HOME=$BASE/hf_home
export PYTHONUNBUFFERED=1
# shared venv, mkdir-mutex bootstrap
VENV=$BASE/venv
if mkdir "\$VENV.lock" 2>/dev/null; then
  if [ ! -f "\$VENV/ready" ]; then
    python3 -m venv "\$VENV"
    "\$VENV/bin/pip" install -q torch numpy transformers==4.46.2 datasets==3.1.0 zstandard sentencepiece accelerate huggingface_hub
    touch "\$VENV/ready"
  fi
  rmdir "\$VENV.lock"
else
  for i in \$(seq 1 240); do [ -f "\$VENV/ready" ] && break; sleep 10; done
  [ -f "\$VENV/ready" ] || { echo "venv never became ready"; exit 1; }
fi
RUN=$BASE/runs/$name
mkdir -p "\$RUN"
"\$VENV/bin/python" $BASE/psc_train_sae.py \\
  --model $model --hook $hook --arm $arm --seed $seed \\
  --steps $steps --k $k --dataset $dataset \\
  --out "\$RUN" --hf-repo dmanningcoe/diffusion-topk-saes --hf-subdir $subdir
EOF
done
echo "generated $(ls "$JOBS" | wc -l) sbatch files"

# ship: env file (tokens), training script, job files
ssh -o ControlPath="$SOCK" "$HOST" "mkdir -p $BASE/logs $BASE/runs $BASE/jobs"
printf 'export HF_TOKEN=%s\nexport HF_WRITE_TOKEN=%s\n' "${HF_TOKEN}" "${HF_WRITE_TOKEN:-$HF_TOKEN}" | \
  ssh -o ControlPath="$SOCK" "$HOST" "cat > ~/.dtxc_env && chmod 600 ~/.dtxc_env"
scp -o ControlPath="$SOCK" -q "$HERE/psc_train_sae.py" "$HOST:$BASE/psc_train_sae.py"
scp -o ControlPath="$SOCK" -q "$JOBS"/*.sbatch "$HOST:$BASE/jobs/"

# submit all, record IDs
: > "$HERE/submitted_jobs.txt"
for spec in "${SPECS[@]}"; do
  read -r name _ <<<"$spec"
  jid=$(ssh -o ControlPath="$SOCK" "$HOST" "sbatch --parsable $BASE/jobs/$name.sbatch")
  echo "$name $jid" | tee -a "$HERE/submitted_jobs.txt"
done
echo "ALL SUBMITTED"
