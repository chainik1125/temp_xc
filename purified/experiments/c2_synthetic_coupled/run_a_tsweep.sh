#!/usr/bin/env bash
# Setup A T-sweep — fill missing T ∈ {2, 4, 6, 8, 10, 12} for txc_base.
# Filler launched T=5 only. 18 shards on 8 GPUs. Runs cache-hit if
# filler's cells already exist on origin/final.
set -e
cd "$(dirname "$0")/../.."

mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

DS="toy_coupled_K10_M20_d256"
SEEDS=(1 2 42)
T_VALUES=(2 4 6 8 10 12)

job_idx=0
for T in "${T_VALUES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    gpu=$((job_idx % 8))
    label="A_tsweep_T${T}_s${seed}"
    log="logs/${label}_gpu${gpu}.log"

    if [[ "$T" == "6"  ]]; then KP="1 2 3 4 6"
    elif [[ "$T" == "8"  ]]; then KP="1 2 3 4 5"
    elif [[ "$T" == "10" ]]; then KP="1 2 3 4"
    elif [[ "$T" == "12" ]]; then KP="1 2 3"
    else                          KP="1 2 3 4 5 6 8"
    fi

    echo "[launch A-tsweep] gpu=${gpu} T=${T} seed=${seed} → ${log}"
    setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
      env AGENT_NAME=agent_synth TEMP_BENCH_POD_MODE=persistent \
          OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
      .venv/bin/python -m experiments.c2_synthetic_coupled.fill_baselines \
        --datasource "${DS}" --arch txc_base --T "${T}" --seed "${seed}" \
        --k-poses ${KP} --n-steps 8000 \
      < /dev/null > "${log}" 2>&1
    job_idx=$((job_idx + 1))
  done
done

echo "[done] launched ${job_idx} A-tsweep shards"
