#!/usr/bin/env bash
# run_sharded.sh — Phase 3 ENGINEER fan-out: 18 (arch_t, seed) tuples
# across 8 GPUs, each tuple processed by ONE subprocess via --arch-t-idx.
# No duplication this time.
#
# Usage (run from purified/):
#   bash experiments/c2_hierarchical/run_sharded.sh
#
# Each tuple = 8 k_pos cells. Round-robin = 2-3 tuples per GPU.
# Wall time at 1-2 min/cell × 24 cells/GPU peak = ~30-50 min.

set -u

cd "$(git rev-parse --show-toplevel)/purified"

LOG_DIR="experiments/c2_hierarchical/run_logs"
mkdir -p "$LOG_DIR"
PRIMARY_DS="${PRIMARY_DS:-toy_hierarchical_Kg10_Kl30_d256}"
N_STEPS="${N_STEPS:-20000}"
N_ARCHTS=6
SEEDS=(1 2 42)

echo "=== C2 Phase 3 ENGINEER (sharded) ==="
echo "  primary datasource: $PRIMARY_DS"
echo "  n_steps:            $N_STEPS"
echo ""

PIDS=()
job_idx=0
for arch_t_idx in $(seq 0 $((N_ARCHTS - 1))); do
    for seed in "${SEEDS[@]}"; do
        gpu=$((job_idx % 8))
        log="$LOG_DIR/sharded_job${job_idx}_gpu${gpu}_archt${arch_t_idx}_seed${seed}.log"
        echo ">>> job $job_idx → GPU $gpu (arch_t=$arch_t_idx seed=$seed)"
        TQDM_DISABLE=1 AGENT_NAME=agent_synth \
            bash scripts/run_on_gpu.sh "$gpu" -- \
            .venv/bin/python -m experiments.c2_hierarchical.run \
                --datasource "$PRIMARY_DS" \
                --arch-t-idx "$arch_t_idx" \
                --seeds "$seed" \
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
    echo "=== Phase 3 ENGINEER complete: all ${#PIDS[@]} jobs finished ==="
else
    echo "=== Phase 3 finished with $FAIL failed job(s); check logs ==="
    exit 1
fi
