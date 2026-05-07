#!/usr/bin/env bash
# Setup K + L — agent_pro mission 2026-05-07.
# 7× RTX 5090 launcher. Two datasources, full baseline coverage:
#   topk_sae, stacked_sae T=2/T=5, tsae_paper, tfa_pos, txc_base T-sweep.
# 12 archs × 3 seeds × 2 datasources = 72 shards, round-robin on 7 GPUs.
# Wall ~25-40 min depending on contention.
set -e
cd "$(dirname "$0")/../.."   # purified/

mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

SEEDS=(1 2 42)

declare -a ARCH_TS=(
  "topk_sae:"
  "stacked_sae:2"
  "stacked_sae:5"
  "tsae_paper:"
  "tfa_pos:"
  "txc_base:2"
  "txc_base:4"
  "txc_base:5"
  "txc_base:6"
  "txc_base:8"
  "txc_base:10"
  "txc_base:12"
)

DATASOURCES=(
  "toy_anticorrelated_Kg10_Kl30_d256"   # Setup K
  "toy_magmod_Kg10_Kl30_d256_alpha1"    # Setup L
)

job_idx=0
for ds in "${DATASOURCES[@]}"; do
  for at in "${ARCH_TS[@]}"; do
    IFS=":" read -r arch T_arg <<< "$at"
    for seed in "${SEEDS[@]}"; do
      gpu=$((job_idx % 7))
      ds_short=$(echo "${ds}" | sed 's/toy_//')
      label="kl_${ds_short}_${arch}_T${T_arg:-X}_s${seed}"
      log="logs/${label}_gpu${gpu}.log"

      # k_poses depend on T (k_pos × T ≤ d_sae=40 for windowed archs).
      if [[ -n "$T_arg" && "$arch" == "txc_base" ]]; then
        case "$T_arg" in
          2)  KP="1 2 3 4 5 6 8" ;;
          4)  KP="1 2 3 4 5 6 8" ;;
          5)  KP="1 2 3 4 5 6 8" ;;
          6)  KP="1 2 3 4 6" ;;
          8)  KP="1 2 3 4 5" ;;
          10) KP="1 2 3 4" ;;
          12) KP="1 2 3" ;;
          *)  KP="1 2 3 4 5 6 8" ;;
        esac
      elif [[ -n "$T_arg" && "$arch" == "stacked_sae" ]]; then
        # stacked T=5: k×5 ≤ 40 → k ≤ 8. T=2: k ≤ 8 globally.
        KP="1 2 3 4 5 6 8"
      else
        KP="1 2 3 4 5 6 8"
      fi

      T_FLAG=""
      [[ -n "$T_arg" ]] && T_FLAG="--T $T_arg"

      echo "[launch] gpu=${gpu} ds=${ds_short} arch=${arch} T=${T_arg:-default} seed=${seed} → ${log}"
      setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
        env AGENT_NAME=agent_pro TEMP_BENCH_POD_MODE=persistent \
            OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
        .venv/bin/python -m experiments.c2_synthetic_coupled.fill_baselines \
          --datasource "${ds}" --arch "${arch}" ${T_FLAG} --seed "${seed}" \
          --k-poses ${KP} --n-steps 8000 \
        < /dev/null > "${log}" 2>&1
      job_idx=$((job_idx + 1))
    done
  done
done

echo
echo "[done] launched ${job_idx} K+L shards across 7 GPUs"
sleep 3
pgrep -af "experiments.c2_synthetic_coupled.fill_baselines" | wc -l
