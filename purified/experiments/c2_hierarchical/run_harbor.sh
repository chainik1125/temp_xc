#!/usr/bin/env bash
# Setup harbor — high-π weak-magnitude globals (per-token noise floor).
set -e
cd "$(dirname "$0")/../.."

mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

DS="toy_hierarchical_Kg10_Kl30_d256_harbor"
SEEDS=(1 2 42)

declare -a ARCH_TS=(
  "topk_sae:" "stacked_sae:2" "stacked_sae:5"
  "tsae_paper:" "tfa_pos:"
  "txc_base:2" "txc_base:4" "txc_base:5"
  "txc_base:6" "txc_base:8" "txc_base:10" "txc_base:12"
)

job_idx=0
for at in "${ARCH_TS[@]}"; do
  IFS=":" read -r arch T_arg <<< "$at"
  for seed in "${SEEDS[@]}"; do
    gpu=$((job_idx % 8))
    label="harbor_${arch}_T${T_arg:-X}_s${seed}"
    log="logs/${label}_gpu${gpu}.log"

    if [[ "$T_arg" == "6"  ]]; then KP="1 2 3 4 6"
    elif [[ "$T_arg" == "8"  ]]; then KP="1 2 3 4 5"
    elif [[ "$T_arg" == "10" ]]; then KP="1 2 3 4"
    elif [[ "$T_arg" == "12" ]]; then KP="1 2 3"
    else                              KP="1 2 3 4 5 6 8"
    fi

    T_FLAG=""
    [[ -n "$T_arg" ]] && T_FLAG="--T $T_arg"

    echo "[launch harbor] gpu=${gpu} arch=${arch} T=${T_arg:-default} seed=${seed} → ${log}"
    setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
      env AGENT_NAME=agent_synth TEMP_BENCH_POD_MODE=persistent \
          OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
      .venv/bin/python -m experiments.c2_synthetic_coupled.fill_baselines \
        --datasource "${DS}" --arch "${arch}" ${T_FLAG} --seed "${seed}" \
        --k-poses ${KP} --n-steps 8000 \
      < /dev/null > "${log}" 2>&1
    job_idx=$((job_idx + 1))
  done
done

echo "[done] launched ${job_idx} harbor shards"
