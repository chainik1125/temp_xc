#!/usr/bin/env bash
# run.sh — Phase 3 ENGINEER launcher for the hierarchical bench.
#
# Fans out 6 archs × 1 primary datasource on GPUs 0-5, plus 2 sanity-
# check datasources on GPUs 6-7.
#
# Usage (run from purified/):
#   bash experiments/c2_hierarchical/run.sh
#
# Logs go to experiments/c2_hierarchical/run_logs/<gpu>_<arch>.log

set -u

cd "$(git rev-parse --show-toplevel)/purified"

LOG_DIR="experiments/c2_hierarchical/run_logs"
mkdir -p "$LOG_DIR"

PRIMARY_DS="${PRIMARY_DS:-toy_hierarchical_Kg10_Kl30_d256}"
SECONDARY_DS_A="${SECONDARY_DS_A:-toy_hierarchical_Kg10_Kl50_d256}"
SECONDARY_DS_B="${SECONDARY_DS_B:-toy_hierarchical_Kg10_Kl30_d256_np2}"
N_STEPS="${N_STEPS:-20000}"

# (gpu, arch_label, --archs, t_label_filter)
# arch labels: topk (no T), stk2 (T=2), stk5 (T=5 default), txcb (T=5 default),
# txp2 (T_max=2), txp5 (T_max=5).
PRIMARY_JOBS=(
    "0:topk_sae:default"
    "1:stacked_sae:T=2"
    "2:stacked_sae:default"
    "3:txc_base:default"
    "4:txc_pro:T=2"
    "5:txc_pro:T=5"
)

echo "=== C2 Phase 3 ENGINEER launcher ==="
echo "  primary datasource:   $PRIMARY_DS"
echo "  secondary GPU 6:      $SECONDARY_DS_A (top-3 archs only)"
echo "  secondary GPU 7:      $SECONDARY_DS_B (top-3 archs only)"
echo "  n_steps:              $N_STEPS"
echo ""

PIDS=()

# Primary jobs: 6 archs on GPUs 0-5, full sweep on PRIMARY_DS.
for entry in "${PRIMARY_JOBS[@]}"; do
    gpu="${entry%%:*}"
    rest="${entry#*:}"
    arch="${rest%%:*}"
    tlabel="${rest#*:}"
    log="$LOG_DIR/gpu${gpu}_${arch}_${tlabel//=/}.log"
    echo ">>> GPU $gpu  →  $arch ($tlabel) on $PRIMARY_DS  (log: $log)"
    TQDM_DISABLE=1 AGENT_NAME=agent_synth \
        bash scripts/run_on_gpu.sh "$gpu" -- \
        .venv/bin/python -m experiments.c2_hierarchical.run \
            --datasource "$PRIMARY_DS" \
            --archs "$arch" \
            --n-steps "$N_STEPS" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

# Secondary jobs: GPUs 6-7 run TXC vs SAE on ablation datasources to
# confirm the win is robust to K_local and n_global_parents.
for sec in "6:$SECONDARY_DS_A" "7:$SECONDARY_DS_B"; do
    gpu="${sec%%:*}"
    ds="${sec#*:}"
    log="$LOG_DIR/gpu${gpu}_${ds}.log"
    echo ">>> GPU $gpu  →  3 archs (topk_sae, txc_base, txc_pro) on $ds  (log: $log)"
    TQDM_DISABLE=1 AGENT_NAME=agent_synth \
        bash scripts/run_on_gpu.sh "$gpu" -- \
        .venv/bin/python -m experiments.c2_hierarchical.run \
            --datasource "$ds" \
            --archs topk_sae txc_base txc_pro \
            --n-steps "$N_STEPS" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} subprocesses: ${PIDS[*]}"
echo "Waiting for all to finish… (tail $LOG_DIR/*.log to monitor)"

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "[FAIL] PID $pid exited non-zero"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
if [ $FAIL -eq 0 ]; then
    echo "=== Phase 3 complete: all ${#PIDS[@]} jobs finished cleanly ==="
else
    echo "=== Phase 3 finished with $FAIL failed job(s); check logs ==="
    exit 1
fi
