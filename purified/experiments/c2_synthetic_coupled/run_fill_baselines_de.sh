#!/usr/bin/env bash
# C2 baseline backfill — Setup D (pB05_np5, pB05_np10) + Setup E.
# 18 (datasource, arch, seed) shards round-robin on 8 GPUs.
#
# Each shard: 1 arch × 1 seed × 7 k_pos = 7 cells, ~3-5 min wall.
# 18 shards / 8 GPUs ≈ 2-3 shards per GPU = ~10-15 min total wall.
#
# TEMP_BENCH_POD_MODE=persistent disables HF auto-push (rate-limit safe).
# Manual push at end via scripts/push_synth_ckpts_to_hf.py.
set -e
cd "$(dirname "$0")/../.."   # purified/

mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

DATASOURCES=(
  "toy_coupled_noisy_K10_M20_d256_pB05_np5"
  "toy_coupled_noisy_K10_M20_d256_pB05_np10"
  "toy_hierarchical_Kg10_Kl30_d256"
)
ARCHS=("tsae_paper" "tfa_pos")
SEEDS=(1 2 42)

# Build shard list, round-robin GPU assignment.
job_idx=0
for ds in "${DATASOURCES[@]}"; do
  for arch in "${ARCHS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      gpu=$((job_idx % 8))
      label="$(echo "${ds}" | sed 's/toy_//;s/_K10_M20_d256_/_/;s/_Kg10_Kl30_d256/_E/')_${arch}_s${seed}"
      log="logs/fill_de_${label}_gpu${gpu}.log"
      echo "[launch] gpu=${gpu} ds=${ds} arch=${arch} seed=${seed} → ${log}"
      setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
        env AGENT_NAME=agent_synth TEMP_BENCH_POD_MODE=persistent \
            OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
        .venv/bin/python -m experiments.c2_synthetic_coupled.fill_baselines \
          --datasource "${ds}" --arch "${arch}" --seed "${seed}" \
          --k-poses 1 2 3 4 5 6 8 \
          --n-steps 8000 \
        < /dev/null > "${log}" 2>&1
      job_idx=$((job_idx + 1))
    done
  done
done

echo
echo "[done] launched ${job_idx} shards across 8 GPUs"
sleep 3
pgrep -af "experiments.c2_synthetic_coupled.fill_baselines" | wc -l
