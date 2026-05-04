#!/usr/bin/env bash
# C7 sweep on pool GPU (default: GPU 2). Launched alongside the primary
# sweep on GPU 1 to parallelise the remaining 4 archs.
#
# Usage:
#   GPU=2 ARCHS="txc_pro tfa mlc stacked_sae" bash scripts/c7_run_sweep_pool.sh
#
# NOTE: claim_gpu via fcntl flock is process-scoped and would release
# when this wrapper script exits. Until temp_bench.utils.gpu_locks
# grows a long-lived "leased" mode, we rely on the per-process
# CUDA_VISIBLE_DEVICES pin + manual coordination.

set -eu
cd "$(dirname "$0")/.."

GPU="${GPU:-2}"
ARCHS="${ARCHS:-txc_pro tfa mlc stacked_sae}"
SEEDS="${SEEDS:-42}"
LOG="logs/c7_sweep_seed${SEEDS// /_}_gpu${GPU}.log"

mkdir -p logs

echo "[c7-sweep-pool] starting — gpu=$GPU archs='$ARCHS' seeds='$SEEDS' log=$LOG" >&2

CUDA_VISIBLE_DEVICES=$GPU TQDM_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m experiments.c7_backtracking.run \
    --archs $ARCHS --seeds $SEEDS \
    > "$LOG" 2>&1 &
PID=$!
echo "[c7-sweep-pool] background PID=$PID" >&2
echo "$PID"
