#!/usr/bin/env bash
# T-sweep launcher: txc_base × T ∈ {2,4,5,6,8,10,12} × seed ∈ {1,2,42}
# at k_pos ∈ {1, 2, 3} (max k_pos that fits all T values at d_sae=40).
#
# 21 (T, seed) jobs round-robin across 8 GPUs = 2-3 jobs/GPU, ~4-5 min wall.

set -u

cd "$(git rev-parse --show-toplevel)/purified"

DS="${1:-toy_coupled_noisy_K10_M20_d256_pB05_np10}"
LOG_DIR="experiments/c2_synthetic_coupled/tsweep_logs"
mkdir -p "$LOG_DIR"
N_STEPS="${N_STEPS:-8000}"
T_VALUES=(2 4 5 6 8 10 12)
SEEDS=(1 2 42)
K_POSES="1 2 3"

echo "=== C2 T-sweep launcher ==="
echo "  datasource: $DS"
echo "  T values:   ${T_VALUES[*]}"
echo "  k_poses:    $K_POSES (fits all T at d_sae=40)"
echo "  n_steps:    $N_STEPS"
echo ""

PIDS=()
job_idx=0
for T in "${T_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu=$((job_idx % 8))
        log="$LOG_DIR/tsweep_${DS#toy_coupled_noisy_K10_M20_d256_}_T${T}_seed${seed}.log"
        echo ">>> job $job_idx → GPU $gpu (T=$T seed=$seed)"
        # Auto-push DISABLED during this run — HF API rate limit
        # (256 commits/hr) is already saturated by the backfill. Cells
        # are pushed manually later via push_synth_ckpts_to_hf.py.
        TQDM_DISABLE=1 AGENT_NAME=agent_synth TEMP_BENCH_POD_MODE=persistent \
            bash scripts/run_on_gpu.sh "$gpu" -- \
            .venv/bin/python -m experiments.c2_synthetic_coupled.run_t_sweep \
                --datasource "$DS" \
                --T "$T" --seed "$seed" \
                --k-poses $K_POSES \
                --n-steps "$N_STEPS" \
            > "$log" 2>&1 &
        PIDS+=($!)
        job_idx=$((job_idx + 1))
    done
done

echo ""
echo "Launched ${#PIDS[@]} subprocesses round-robin across 8 GPUs."
echo "Waiting…"

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[FAIL] PID $pid exited non-zero"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
if [ $FAIL -eq 0 ]; then
    echo "=== T-sweep complete: ${#PIDS[@]} jobs finished ==="
else
    echo "=== T-sweep finished with $FAIL failed job(s); check logs ==="
    exit 1
fi
