#!/usr/bin/env bash
# run_hunt.sh — Phase 1 HUNT launcher for agent_synth on 8× H100.
#
# Fans out 8 parallel subprocesses, each pinned to one GPU, each
# processing one (p_B, n_parents) datasource. Together = 288 cells.
#
# Usage (run from purified/):
#   bash experiments/c2_synthetic_coupled/run_hunt.sh
#
# Logs go to experiments/c2_synthetic_coupled/hunt_logs/<datasource>.log

set -u

cd "$(git rev-parse --show-toplevel)/purified"

LOG_DIR="experiments/c2_synthetic_coupled/hunt_logs"
mkdir -p "$LOG_DIR"

# (gpu_idx, datasource) — see briefing for the rationale of each cell.
DATASOURCES=(
    "0:toy_coupled_noisy_K10_M20_d256_pB05_np2"
    "1:toy_coupled_noisy_K10_M20_d256_pB05_np5"
    "2:toy_coupled_noisy_K10_M20_d256_pB03_np5"
    "3:toy_coupled_noisy_K10_M20_d256_pB03_np8"
    "4:toy_coupled_noisy_K10_M20_d256_pB02_np8"
    "5:toy_coupled_noisy_K10_M20_d256_pB05_np8"
    "6:toy_coupled_noisy_K10_M20_d256_pB01_np5"
    "7:toy_coupled_noisy_K10_M20_d256_pB05_np10"
)

PHASE="${PHASE:-hunt}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=== C2 HUNT launcher (8× H100) ==="
echo "  phase=$PHASE  log_dir=$LOG_DIR  extra=$EXTRA_ARGS"
echo ""

PIDS=()
for entry in "${DATASOURCES[@]}"; do
    gpu="${entry%%:*}"
    ds="${entry#*:}"
    log="$LOG_DIR/${ds}.log"
    echo ">>> GPU $gpu  →  $ds  (log: $log)"
    TQDM_DISABLE=1 AGENT_NAME=agent_synth \
        bash scripts/run_on_gpu.sh "$gpu" -- \
        .venv/bin/python -m experiments.c2_synthetic_coupled.run_hunt \
            --datasource "$ds" --phase "$PHASE" $EXTRA_ARGS \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} subprocesses: ${PIDS[*]}"
echo "Waiting for all to finish… (tail $LOG_DIR/*.log to monitor)"
echo ""

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[FAIL] PID $pid exited non-zero"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
if [ $FAIL -eq 0 ]; then
    echo "=== HUNT complete: all 8 shards finished cleanly ==="
else
    echo "=== HUNT finished with $FAIL failed shard(s); check logs ==="
    exit 1
fi
