#!/usr/bin/env bash
# run_zoom.sh — Phase 2 ZOOM launcher for agent_synth on 8× H100.
#
# Takes the winning (p_B, n_parents) datasource as arg (or reads it
# from hunt_summary.json) and dense-sweeps:
#   - 6 archs × 3 seeds × 12 k_pos = 216 cells
#   - n_steps = 30_000
#   - 18 (arch_t, seed) tuples sharded across 8 GPUs round-robin
#     → 2-3 tuples sequentially per GPU = ~70-90 min wall total.
#
# Usage:
#   bash experiments/c2_synthetic_coupled/run_zoom.sh \
#        toy_coupled_noisy_K10_M20_d256_pB05_np5
# or auto-pick from hunt_summary.json:
#   bash experiments/c2_synthetic_coupled/run_zoom.sh

set -u

cd "$(git rev-parse --show-toplevel)/purified"

DS="${1:-}"
if [ -z "$DS" ] && [ -f experiments/c2_synthetic_coupled/hunt_summary.json ]; then
    DS=$(.venv/bin/python -c "import json; print(json.load(open('experiments/c2_synthetic_coupled/hunt_summary.json'))['overall_winner_datasource'])")
fi

if [ -z "$DS" ]; then
    echo "ERROR: no datasource provided and hunt_summary.json missing." >&2
    echo "Usage: $0 <datasource>" >&2
    exit 1
fi

LOG_DIR="experiments/c2_synthetic_coupled/zoom_logs"
mkdir -p "$LOG_DIR"
N_STEPS="${N_STEPS:-30000}"
SEEDS=(1 2 42)
N_ARCHTS=6  # Number of entries in ZOOM_ARCH_TS

echo "=== C2 Phase 2 ZOOM launcher (8× H100) ==="
echo "  winner datasource: $DS"
echo "  n_steps:           $N_STEPS"
echo "  log_dir:           $LOG_DIR"
echo ""

PIDS=()
job_idx=0
for arch_t_idx in $(seq 0 $((N_ARCHTS - 1))); do
    for seed in "${SEEDS[@]}"; do
        gpu=$((job_idx % 8))
        log="$LOG_DIR/job${job_idx}_gpu${gpu}_archt${arch_t_idx}_seed${seed}.log"
        echo ">>> job $job_idx → GPU $gpu (arch_t=$arch_t_idx seed=$seed)"
        TQDM_DISABLE=1 AGENT_NAME=agent_synth \
            bash scripts/run_on_gpu.sh "$gpu" -- \
            .venv/bin/python -m experiments.c2_synthetic_coupled.run_hunt \
                --datasource "$DS" \
                --phase zoom \
                --arch-t-idx "$arch_t_idx" \
                --seeds "$seed" \
                --n-steps "$N_STEPS" \
            > "$log" 2>&1 &
        PIDS+=($!)
        job_idx=$((job_idx + 1))
    done
done

echo ""
echo "Launched ${#PIDS[@]} subprocesses (round-robin across 8 GPUs)."
echo "Each job runs 12 k_pos cells. Each GPU runs 2-3 jobs concurrently."
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
    echo "=== ZOOM complete: all ${#PIDS[@]} jobs finished cleanly ==="
else
    echo "=== ZOOM finished with $FAIL failed job(s); check logs ==="
    exit 1
fi
