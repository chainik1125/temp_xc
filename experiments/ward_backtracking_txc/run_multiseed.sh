#!/usr/bin/env bash
# Multi-seed verification: TXC vs H13 close call on resid_L10 k=16.
#
# Re-trains txc and txc_h13 at resid_L10 k=16 with 3 additional seeds
# {7, 11, 23}. Uses the inline-Sonnet-grading evaluate_cell so each cell
# emits both a per-cell B1 and a Sonnet-floor metric in one pass.
#
# 2-GPU parallelism: GPU 0 runs the TXC seeds, GPU 1 runs the H13 seeds.
# Each cell takes ~10-15 min total with the parallelized B1 (commit
# 95a27989).

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="experiments.ward_backtracking_txc"

SEEDS=(7 11 23)
GPU0_LOG=/tmp/multiseed_gpu0.log
GPU1_LOG=/tmp/multiseed_gpu1.log
: > "$GPU0_LOG"
: > "$GPU1_LOG"

run_seed() {
    local arch=$1 seed=$2 gpu=$3 logfile=$4
    local cell="${arch}__resid_L10__k16__s${seed}"
    echo "[multiseed] launching $cell on cuda:$gpu" >> "$logfile"
    CUDA_VISIBLE_DEVICES="$gpu" python -m $ROOT.evaluate_cell --cell "$cell" >> "$logfile" 2>&1
    echo "[multiseed] $cell DONE" >> "$logfile"
}

# GPU 0 → TXC seeds (sequential within GPU)
(
  for s in "${SEEDS[@]}"; do
    run_seed txc "$s" 0 "$GPU0_LOG"
  done
) &
PID_GPU0=$!

# GPU 1 → H13 seeds (sequential within GPU)
(
  for s in "${SEEDS[@]}"; do
    run_seed txc_h13 "$s" 1 "$GPU1_LOG"
  done
) &
PID_GPU1=$!

echo "[multiseed] GPU 0 driver PID=$PID_GPU0 (TXC seeds 7/11/23 → $GPU0_LOG)"
echo "[multiseed] GPU 1 driver PID=$PID_GPU1 (H13 seeds 7/11/23 → $GPU1_LOG)"
wait $PID_GPU0 $PID_GPU1
echo "[multiseed] all 6 cells complete."

# Refresh global metrics from the new cell_metrics files (no API needed —
# inline grading already wrote Sonnet metrics during evaluate_cell).
python -m $ROOT.regrade_cells --concurrency 12

echo "[multiseed] DONE."
