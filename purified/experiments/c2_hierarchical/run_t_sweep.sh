#!/usr/bin/env bash
# Setup E (hierarchical) txc_base T-sweep launcher.
# 21 jobs (T × seed) round-robin on 8 GPUs, ~5-10 min wall.

set -u
cd "$(git rev-parse --show-toplevel)/purified"

DS="${1:-toy_hierarchical_Kg10_Kl30_d256}"
LOG_DIR="experiments/c2_hierarchical/tsweep_logs"
mkdir -p "$LOG_DIR"
N_STEPS="${N_STEPS:-8000}"
T_VALUES=(2 4 5 6 8 10 12)
SEEDS=(1 2 42)
K_POSES="1 2 3"

PIDS=()
job_idx=0
for T in "${T_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu=$((job_idx % 8))
        log="$LOG_DIR/tsweep_hier_${DS#toy_hierarchical_}_T${T}_seed${seed}.log"
        # Auto-push DISABLED (HF rate limit).
        TQDM_DISABLE=1 AGENT_NAME=agent_synth TEMP_BENCH_POD_MODE=persistent \
            bash scripts/run_on_gpu.sh "$gpu" -- \
            .venv/bin/python -m experiments.c2_hierarchical.run_t_sweep \
                --datasource "$DS" \
                --T "$T" --seed "$seed" \
                --k-poses $K_POSES \
                --n-steps "$N_STEPS" \
            > "$log" 2>&1 &
        PIDS+=($!)
        job_idx=$((job_idx + 1))
    done
done
echo "Launched ${#PIDS[@]} hier T-sweep jobs"
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then FAIL=$((FAIL+1)); fi
done
echo "Hier T-sweep done; $FAIL fails"
