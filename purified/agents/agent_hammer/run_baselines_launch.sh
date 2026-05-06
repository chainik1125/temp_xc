#!/usr/bin/env bash
# agent_hammer launcher: 108 baseline cells across 5× RTX PRO 6000.
#
# Sharding: 9 (arch, seed, setup) shards × 2 k_pos chunks = 18 sub-shards.
# Distribute 4 procs/GPU on GPUs 0-3 + 2 procs on GPU 4 = 18 total parallel.
# Each sub-shard: 6 cells × ~30-60 sec = 3-6 min wall (assuming light GPU
# contention thanks to small toy d_sae=40 cells).
#
# Sub-shards (18):
#   Setup A (c2): tsae_paper × 3 seeds × 2 chunks    →  6 sub-shards
#   Setup B (c1_noisy): tsae_paper × 3 seeds × 2 ch  →  6 sub-shards
#   Setup B (c1_noisy): topk_sae   × 3 seeds × 2 ch  →  6 sub-shards
#
# Layout:
#   GPU 0: A_tsae_s1_*  + B_tsae_s1_*    (4 procs, all tsae_paper)
#   GPU 1: A_tsae_s2_*  + B_tsae_s2_*    (4 procs)
#   GPU 2: A_tsae_s42_* + B_tsae_s42_*   (4 procs)
#   GPU 3: B_topk_s1_*  + B_topk_s2_*    (4 procs, topk_sae)
#   GPU 4: B_topk_s42_*                  (2 procs)
set -e
cd /workspace/temp_xc/purified
mkdir -p logs

K1="1 2 3 4 5 6"
K2="8 10 12 15 17 20"

launch() {
  local gpu=$1; shift
  local driver=$1; shift
  local arch=$1; shift
  local seed=$1; shift
  local kvals="$1"; shift
  local label=$1; shift
  local logfile="logs/hammer_${label}.log"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env AGENT_NAME=agent_hammer TEMP_BENCH_POD_MODE=ephemeral OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
    .venv/bin/python -m "${driver}" \
      --arch "${arch}" --seed "${seed}" --k-poses ${kvals} \
    < /dev/null > "${logfile}" 2>&1
}

# --- GPU 0: Setup A + Setup B tsae_paper seed 1 (4 procs) ---
launch 0 experiments.c2_synthetic_coupled.run_baselines tsae_paper 1  "$K1" setupA_tsae_s1_k1
launch 0 experiments.c2_synthetic_coupled.run_baselines tsae_paper 1  "$K2" setupA_tsae_s1_k2
launch 0 experiments.c1_noisy_filler.run_baselines       tsae_paper 1  "$K1" setupB_tsae_s1_k1
launch 0 experiments.c1_noisy_filler.run_baselines       tsae_paper 1  "$K2" setupB_tsae_s1_k2

# --- GPU 1: Setup A + Setup B tsae_paper seed 2 (4 procs) ---
launch 1 experiments.c2_synthetic_coupled.run_baselines tsae_paper 2  "$K1" setupA_tsae_s2_k1
launch 1 experiments.c2_synthetic_coupled.run_baselines tsae_paper 2  "$K2" setupA_tsae_s2_k2
launch 1 experiments.c1_noisy_filler.run_baselines       tsae_paper 2  "$K1" setupB_tsae_s2_k1
launch 1 experiments.c1_noisy_filler.run_baselines       tsae_paper 2  "$K2" setupB_tsae_s2_k2

# --- GPU 2: Setup A + Setup B tsae_paper seed 42 (4 procs) ---
launch 2 experiments.c2_synthetic_coupled.run_baselines tsae_paper 42 "$K1" setupA_tsae_s42_k1
launch 2 experiments.c2_synthetic_coupled.run_baselines tsae_paper 42 "$K2" setupA_tsae_s42_k2
launch 2 experiments.c1_noisy_filler.run_baselines       tsae_paper 42 "$K1" setupB_tsae_s42_k1
launch 2 experiments.c1_noisy_filler.run_baselines       tsae_paper 42 "$K2" setupB_tsae_s42_k2

# --- GPU 3: Setup B topk_sae seeds 1, 2 (4 procs) ---
launch 3 experiments.c1_noisy_filler.run_baselines       topk_sae   1  "$K1" setupB_topk_s1_k1
launch 3 experiments.c1_noisy_filler.run_baselines       topk_sae   1  "$K2" setupB_topk_s1_k2
launch 3 experiments.c1_noisy_filler.run_baselines       topk_sae   2  "$K1" setupB_topk_s2_k1
launch 3 experiments.c1_noisy_filler.run_baselines       topk_sae   2  "$K2" setupB_topk_s2_k2

# --- GPU 4: Setup B topk_sae seed 42 (2 procs) ---
launch 4 experiments.c1_noisy_filler.run_baselines       topk_sae   42 "$K1" setupB_topk_s42_k1
launch 4 experiments.c1_noisy_filler.run_baselines       topk_sae   42 "$K2" setupB_topk_s42_k2

echo "[hammer] launched 18 sub-shards"
sleep 3
pgrep -af "experiments.c[12]_.*\.run_baselines" | head -25
echo "[hammer] use: tail -f logs/hammer_*.log  to monitor"
