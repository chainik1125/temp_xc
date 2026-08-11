#!/usr/bin/env bash
# Re-seed the PSC gemma cohort to seeds 2/3 so it doesn't collide with the
# Modal gemma cohort (seeds 0/1). Writes 6 new sbatch files, ships them, and
# submits. Cancelling the old seed-0/1 gemma jobs is a separate deliberate
# step -- see the caller.
#
# Two fixes vs the original generator:
#   - /usr/bin/python3.11 for the venv (the default python3 is 3.6.8, which
#     cannot install transformers==4.46.2 at all);
#   - 5:30 walltime instead of 12:00, which backfills far more easily and is
#     still ~3.5x the measured 1.6 h runtime on the same l40s-48 GPU.
set -euo pipefail

SOCK="$HOME/.ssh/cm/psc"
HOST="manningc@bridges2.psc.edu"
BASE="/ocean/projects/cis240096p/manningc/dtxc"
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS="$HERE/jobs"
WALL="5:30:00"
SUBDIR="gemma2-2b-l12-100M"

SPECS=(
  "gemma_recon_s2   recon      2"
  "gemma_recon_s3   recon      3"
  "gemma_dsm_s2     dsm        2"
  "gemma_dsm_s3     dsm        3"
  "gemma_dsmann_s2  dsm_anneal 2"
  "gemma_dsmann_s3  dsm_anneal 3"
)

for spec in "${SPECS[@]}"; do
  read -r name arm seed <<<"$spec"
  cat > "$JOBS/$name.sbatch" <<EOF
#!/bin/bash
#SBATCH --account=cis240096p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH --cpus-per-task=4
#SBATCH --time=$WALL
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
    /usr/bin/python3.11 -m venv "\$VENV"
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
  --model google/gemma-2-2b --hook resid12 --arm $arm --seed $seed \\
  --steps 24000 --k 40 --dataset pile \\
  --out "\$RUN" --hf-repo dmanningcoe/diffusion-topk-saes --hf-subdir $SUBDIR
EOF
  # scp to the login node is blocked; pipe through ssh instead
  ssh -o ControlPath="$SOCK" "$HOST" "cat > $BASE/jobs/$name.sbatch" < "$JOBS/$name.sbatch"
done
echo "shipped 6 sbatch files"

: > "$HERE/submitted_gemma_s23.txt"
for spec in "${SPECS[@]}"; do
  read -r name _ <<<"$spec"
  jid=$(ssh -o ControlPath="$SOCK" "$HOST" "sbatch --parsable $BASE/jobs/$name.sbatch")
  echo "$name $jid" | tee -a "$HERE/submitted_gemma_s23.txt"
done
echo "SUBMITTED"
