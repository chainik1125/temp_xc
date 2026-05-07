#!/usr/bin/env bash
# Setup G launcher: pinned to GPUs 0-3 only (4 GPUs).
# 7 T × 3 seeds + 3 SAE seeds = 24 jobs round-robin on 4 GPUs.
set -u
cd "$(git rev-parse --show-toplevel)/purified"

DS="${1:-toy_hierarchical_Kg10_Kl30_d256_sigma1p0}"
LOG_DIR="experiments/c2_hierarchical/setupg_logs"
mkdir -p "$LOG_DIR"
N_STEPS="${N_STEPS:-8000}"
T_VALUES=(2 4 5 6 8 10 12)
SEEDS=(1 2 42)
K_POSES="1 2 3"
N_GPUS="${N_GPUS:-4}"        # 4 GPUs by default
GPU_OFFSET="${GPU_OFFSET:-0}" # start at GPU 0 by default

PIDS=()
job_idx=0
for T in "${T_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu=$((GPU_OFFSET + job_idx % N_GPUS))
        log="$LOG_DIR/setupG_${DS#toy_hierarchical_}_T${T}_seed${seed}.log"
        TQDM_DISABLE=1 AGENT_NAME=agent_synth TEMP_BENCH_POD_MODE=persistent \
            bash scripts/run_on_gpu.sh "$gpu" -- \
            .venv/bin/python -m experiments.c2_hierarchical.run_setup_g \
                --datasource "$DS" --T "$T" --seed "$seed" \
                --k-poses $K_POSES --n-steps "$N_STEPS" --archs txc_base \
            > "$log" 2>&1 &
        PIDS+=($!)
        job_idx=$((job_idx + 1))
    done
done
for seed in "${SEEDS[@]}"; do
    gpu=$((GPU_OFFSET + job_idx % N_GPUS))
    log="$LOG_DIR/setupG_topk_${DS#toy_hierarchical_}_seed${seed}.log"
    TQDM_DISABLE=1 AGENT_NAME=agent_synth TEMP_BENCH_POD_MODE=persistent \
        bash scripts/run_on_gpu.sh "$gpu" -- \
        .venv/bin/python -m experiments.c2_hierarchical.run_setup_g \
            --datasource "$DS" --T 5 --seed "$seed" \
            --k-poses $K_POSES --n-steps "$N_STEPS" --archs topk_sae \
        > "$log" 2>&1 &
    PIDS+=($!)
    job_idx=$((job_idx + 1))
done
echo "Launched ${#PIDS[@]} Setup G jobs on GPUs 0-${N_GPUS} only"
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then FAIL=$((FAIL+1)); fi
done
echo "Setup G done; $FAIL fails"
