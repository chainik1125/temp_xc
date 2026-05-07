#!/usr/bin/env bash
# Setup F + G baseline backfill — 180 cells.
#
# F: 3 datasources (sigma=0.5, 1.0, 2.0) × 4 archs × 3 seeds × 3 k_pos
# G: 2 datasources (sigma=1.0, 2.0)      × 4 archs × 3 seeds × 3 k_pos
#   Total = (3+2) × 4 × 3 × 3 = 180 cells.
#
# Each shard = one (arch, datasource, seed, T) tuple iterating 3 k_pos
# values via the driver. 60 shards total = 4 archs × 5 ds × 3 seeds.
# (stacked_sae T=2 and T=5 count as different archs for sharding.)
#
# Layout: 12 shards/GPU on 5 GPUs (60 / 5). All in parallel — d_sae=40
# means each cell ≈ 1-2 GB VRAM, plenty of headroom on 96 GB cards.
# At 12-tenant per GPU, throughput is GPU-bound → wall ≈ 15 min.
set -e
cd /workspace/temp_xc/purified
mkdir -p logs

DATASOURCES_F=(
  "toy_coupled_obs_noise_K10_M20_d256_sigma0p5"
  "toy_coupled_obs_noise_K10_M20_d256_sigma1p0"
  "toy_coupled_obs_noise_K10_M20_d256_sigma2p0"
)
DATASOURCES_G=(
  "toy_hierarchical_Kg10_Kl30_d256_sigma1p0"
  "toy_hierarchical_Kg10_Kl30_d256_sigma2p0"
)
ALL_DS=("${DATASOURCES_F[@]}" "${DATASOURCES_G[@]}")
SEEDS=(1 2 42)

# (arch_label, --arch, --T) — stacked_sae T=2 and T=5 each count as a shard.
ARCH_TUPLES=(
  "tfa_pos      tfa_pos      "        # no T
  "stacked_T2   stacked_sae  2"
  "stacked_T5   stacked_sae  5"
  "tsae_paper   tsae_paper   "         # no T (uses train_window_size=2 inside driver)
)

# Build full shard list.
shards=()
for ds in "${ALL_DS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for tuple in "${ARCH_TUPLES[@]}"; do
      read -r label arch T <<< "$tuple"
      t_arg=""
      [[ -n "$T" ]] && t_arg="--T $T"
      shards+=("$ds|$seed|$arch|$label|$t_arg")
    done
  done
done

echo "[fg_baselines] total shards: ${#shards[@]}"

# Round-robin to 5 GPUs.
n_gpus=5
i=0
for shard in "${shards[@]}"; do
  gpu=$((i % n_gpus))
  IFS="|" read -r ds seed arch label t_arg <<< "$shard"
  log="logs/hammer_fg_${label}_${ds##*_}_seed${seed}.log"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env AGENT_NAME=agent_hammer TEMP_BENCH_POD_MODE=ephemeral \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
    .venv/bin/python -m agents.agent_hammer.run_setup_fg_baselines \
      --datasource "${ds}" --arch "${arch}" --seed "${seed}" \
      --k-poses 1 2 3 ${t_arg} \
    < /dev/null > "${log}" 2>&1
  i=$((i + 1))
done

echo "[fg_baselines] launched ${#shards[@]} shards across ${n_gpus} GPUs"
sleep 3
n_alive=$(pgrep -af "run_setup_fg_baselines" | grep -v grep | wc -l)
echo "[fg_baselines] alive python procs: ${n_alive}"
echo "[fg_baselines] tail logs/hammer_fg_*.log to monitor"
