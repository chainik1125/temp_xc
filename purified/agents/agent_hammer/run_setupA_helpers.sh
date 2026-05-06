#!/usr/bin/env bash
# Phase 2.5 — Helper procs on freed GPUs 3, 4 for setupA tsae latter halves.
# Existing procs on GPUs 0,1,2 will skip these cells once helpers cache them.
#
# Helpers target k_poses that existing procs haven't reached yet:
#   chunk1 latter: k=4, 5, 6   (existing GPU 0,1,2 chunk1 procs are at k=1-3)
#   chunk2 latter: k=15, 17, 20 (existing chunk2 procs are at k=8-12)
set -e
cd /workspace/temp_xc/purified
mkdir -p logs

launch() {
  local gpu=$1; shift
  local arch=$1; shift
  local seed=$1; shift
  local kvals="$1"; shift
  local label=$1; shift
  local logfile="logs/hammer_${label}.log"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env AGENT_NAME=agent_hammer TEMP_BENCH_POD_MODE=ephemeral OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
    .venv/bin/python -m experiments.c2_synthetic_coupled.run_baselines \
      --arch "${arch}" --seed "${seed}" --k-poses ${kvals} \
    < /dev/null > "${logfile}" 2>&1
}

K1_LATTER="4 5 6"
K2_LATTER="15 17 20"

# GPU 3: chunk1 latter halves (3 procs at 3-tenant)
launch 3 tsae_paper 1  "$K1_LATTER" helper_setupA_s1_k1latter
launch 3 tsae_paper 2  "$K1_LATTER" helper_setupA_s2_k1latter
launch 3 tsae_paper 42 "$K1_LATTER" helper_setupA_s42_k1latter

# GPU 4: chunk2 latter halves (3 procs at 3-tenant)
launch 4 tsae_paper 1  "$K2_LATTER" helper_setupA_s1_k2latter
launch 4 tsae_paper 2  "$K2_LATTER" helper_setupA_s2_k2latter
launch 4 tsae_paper 42 "$K2_LATTER" helper_setupA_s42_k2latter

echo "[hammer] launched 6 helper procs on GPUs 3, 4"
sleep 2
pgrep -af "experiments.c2_.*\.run_baselines" | wc -l
echo "[hammer] tail logs/hammer_helper_*.log to monitor"
