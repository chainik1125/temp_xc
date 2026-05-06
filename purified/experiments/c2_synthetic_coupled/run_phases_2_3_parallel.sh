#!/usr/bin/env bash
# run_phases_2_3_parallel.sh — fan out Phase 2 ZOOM + Phase 3 ENGINEER
# concurrently across 8 GPUs once Phase 1 HUNT has identified a winner.
#
# Phase 2 (ZOOM): 18 (arch_t × seed) jobs on the winner datasource.
# Phase 3 (ENGINEER): 8 jobs (6 archs primary + 2 secondary datasources).
# Together: 26 jobs distributed round-robin across 8 GPUs (~3 jobs/GPU
# concurrently). H100 80GB easily fits 3 small models per GPU.
#
# Usage (run from purified/, after run_hunt.sh + hunt_analysis.py):
#   bash experiments/c2_synthetic_coupled/run_phases_2_3_parallel.sh
#
# Reads winner from experiments/c2_synthetic_coupled/hunt_summary.json.

set -u

cd "$(git rev-parse --show-toplevel)/purified"

ZOOM_LOG_DIR="experiments/c2_synthetic_coupled/zoom_logs"
HIER_LOG_DIR="experiments/c2_hierarchical/run_logs"
mkdir -p "$ZOOM_LOG_DIR" "$HIER_LOG_DIR"

# 1. Identify the winner datasource.
SUMMARY=experiments/c2_synthetic_coupled/hunt_summary.json
if [ ! -f "$SUMMARY" ]; then
    echo "ERROR: $SUMMARY missing — run hunt_analysis first." >&2
    exit 1
fi
WINNER=$(.venv/bin/python -c "import json; print(json.load(open('$SUMMARY'))['overall_winner_datasource'])")
echo "=== Winner datasource: $WINNER ==="

ZOOM_N_STEPS="${ZOOM_N_STEPS:-30000}"
HIER_N_STEPS="${HIER_N_STEPS:-20000}"
SEEDS=(1 2 42)
N_ARCHTS=6

PIDS=()
job_idx=0

# 2. Phase 2 ZOOM jobs (18 (arch_t, seed) tuples → round-robin GPU 0..7).
echo ""
echo "--- Phase 2 ZOOM jobs ---"
for arch_t_idx in $(seq 0 $((N_ARCHTS - 1))); do
    for seed in "${SEEDS[@]}"; do
        gpu=$((job_idx % 8))
        log="$ZOOM_LOG_DIR/zoom_job${job_idx}_gpu${gpu}_archt${arch_t_idx}_seed${seed}.log"
        echo ">>> zoom job $job_idx → GPU $gpu (arch_t=$arch_t_idx seed=$seed)"
        TQDM_DISABLE=1 AGENT_NAME=agent_synth \
            bash scripts/run_on_gpu.sh "$gpu" -- \
            .venv/bin/python -m experiments.c2_synthetic_coupled.run_hunt \
                --datasource "$WINNER" \
                --phase zoom \
                --arch-t-idx "$arch_t_idx" \
                --seeds "$seed" \
                --n-steps "$ZOOM_N_STEPS" \
            > "$log" 2>&1 &
        PIDS+=($!)
        job_idx=$((job_idx + 1))
    done
done

# 3. Phase 3 ENGINEER jobs (6 primary archs + 2 secondary datasources).
echo ""
echo "--- Phase 3 ENGINEER jobs ---"
PRIMARY_DS="toy_hierarchical_Kg10_Kl30_d256"
SECONDARY_DS_A="toy_hierarchical_Kg10_Kl50_d256"
SECONDARY_DS_B="toy_hierarchical_Kg10_Kl30_d256_np2"

# 6 primary archs on GPUs 0-5 (each runs all 3 seeds × 8 k_pos).
PRIMARY_JOBS=(
    "topk_sae:default"
    "stacked_sae:T=2"
    "stacked_sae:default"
    "txc_base:default"
    "txc_pro:T=2"
    "txc_pro:T=5"
)
for entry in "${PRIMARY_JOBS[@]}"; do
    arch="${entry%%:*}"
    tlabel="${entry#*:}"
    gpu=$((job_idx % 8))
    log="$HIER_LOG_DIR/hier_job${job_idx}_gpu${gpu}_${arch}_${tlabel//=/}.log"
    echo ">>> hier primary $arch ($tlabel) → GPU $gpu"
    TQDM_DISABLE=1 AGENT_NAME=agent_synth \
        bash scripts/run_on_gpu.sh "$gpu" -- \
        .venv/bin/python -m experiments.c2_hierarchical.run \
            --datasource "$PRIMARY_DS" \
            --archs "$arch" \
            --n-steps "$HIER_N_STEPS" \
        > "$log" 2>&1 &
    PIDS+=($!)
    job_idx=$((job_idx + 1))
done

# 2 secondary datasources × 3 archs (topk_sae, txc_base, txc_pro).
for ds in "$SECONDARY_DS_A" "$SECONDARY_DS_B"; do
    gpu=$((job_idx % 8))
    log="$HIER_LOG_DIR/hier_secondary_gpu${gpu}_${ds}.log"
    echo ">>> hier secondary ds=$ds → GPU $gpu (3 archs)"
    TQDM_DISABLE=1 AGENT_NAME=agent_synth \
        bash scripts/run_on_gpu.sh "$gpu" -- \
        .venv/bin/python -m experiments.c2_hierarchical.run \
            --datasource "$ds" \
            --archs topk_sae txc_base txc_pro \
            --n-steps "$HIER_N_STEPS" \
        > "$log" 2>&1 &
    PIDS+=($!)
    job_idx=$((job_idx + 1))
done

echo ""
echo "Launched ${#PIDS[@]} subprocesses (round-robin across 8 GPUs)."
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
    echo "=== Phases 2 + 3 complete: all ${#PIDS[@]} jobs finished cleanly ==="
else
    echo "=== Phases 2 + 3 finished with $FAIL failed job(s); check logs ==="
    exit 1
fi
