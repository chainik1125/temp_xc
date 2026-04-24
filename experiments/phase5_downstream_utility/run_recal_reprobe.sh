#!/usr/bin/env bash
# Recalibrate BatchTopK threshold for T15/T20 and re-probe.
# Original ckpts finished training with negative `threshold` buffers
# (see batchtopk_threshold_audit.json). The recalibration loads each
# ckpt, runs 200 fineweb batches in train() mode, and saves a new ckpt
# with suffix `_recal`. Then we re-probe at both aggregations.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/recal_reprobe_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

ARCHS=(
    txcdr_t15_batchtopk
    txcdr_t20_batchtopk
)

say "=== Recalibrating BatchTopK thresholds ==="
.venv/bin/python -u experiments/phase5_downstream_utility/analysis/recalibrate_batchtopk_threshold.py \
    "${ARCHS[@]}" 2>&1 | tee "$LOG_DIR/recal_run.log"

say "=== Re-probing recalibrated ckpts ==="
RIDS=()
for a in "${ARCHS[@]}"; do
    RIDS+=("${a}_recal__seed42")
done

for AGG in last_position mean_pool; do
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "$AGG" --skip-baselines \
        --run-ids "${RIDS[@]}" \
        > "$LOG_DIR/probe_recal_${AGG}.log" 2>&1
    say "  probe $AGG exit=$? in $(( $(date +%s) - t0 ))s"
done
say "=== recal_reprobe done ==="
