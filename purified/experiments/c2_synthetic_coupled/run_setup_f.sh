#!/usr/bin/env bash
# Setup F launcher: txc_base T-sweep + topk_sae baseline on the coupled +
# observation-noise generator at one σ value (default σ=1.0).
#
# Cells: 7 T × 3 seeds × 3 k_pos = 63 (txc_base) + 1 T × 3 seeds × 3
# k_pos = 9 (topk_sae) = 72. ~5-10 min wall on 8 GPUs.
set -u
cd "$(git rev-parse --show-toplevel)/purified"

DS="${1:-toy_coupled_obs_noise_K10_M20_d256_sigma1p0}"
LOG_DIR="experiments/c2_synthetic_coupled/setupf_logs"
mkdir -p "$LOG_DIR"
N_STEPS="${N_STEPS:-8000}"
T_VALUES=(2 4 5 6 8 10 12)
SEEDS=(1 2 42)
K_POSES="1 2 3"

PIDS=()
job_idx=0

# txc_base T-sweep
for T in "${T_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu=$((job_idx % 8))
        log="$LOG_DIR/setupF_${DS#toy_coupled_obs_noise_K10_M20_d256_}_T${T}_seed${seed}.log"
        TQDM_DISABLE=1 AGENT_NAME=agent_synth TEMP_BENCH_POD_MODE=persistent \
            bash scripts/run_on_gpu.sh "$gpu" -- \
            .venv/bin/python -m experiments.c2_synthetic_coupled.run_setup_f \
                --datasource "$DS" --T "$T" --seed "$seed" \
                --k-poses $K_POSES --n-steps "$N_STEPS" --archs txc_base \
            > "$log" 2>&1 &
        PIDS+=($!)
        job_idx=$((job_idx + 1))
    done
done

# topk_sae baseline (single T value to keep eval_cfg consistent; topk_sae
# ignores T anyway).
for seed in "${SEEDS[@]}"; do
    gpu=$((job_idx % 8))
    log="$LOG_DIR/setupF_topk_${DS#toy_coupled_obs_noise_K10_M20_d256_}_seed${seed}.log"
    TQDM_DISABLE=1 AGENT_NAME=agent_synth TEMP_BENCH_POD_MODE=persistent \
        bash scripts/run_on_gpu.sh "$gpu" -- \
        .venv/bin/python -m experiments.c2_synthetic_coupled.run_setup_f \
            --datasource "$DS" --T 5 --seed "$seed" \
            --k-poses $K_POSES --n-steps "$N_STEPS" --archs topk_sae \
        > "$log" 2>&1 &
    PIDS+=($!)
    job_idx=$((job_idx + 1))
done

echo "Launched ${#PIDS[@]} Setup F jobs"
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then FAIL=$((FAIL+1)); fi
done
echo "Setup F done; $FAIL fails"
