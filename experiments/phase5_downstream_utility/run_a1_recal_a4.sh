#!/usr/bin/env bash
# Post-probe orchestration: A1 concat-probe, then recalibrate T15/T20
# thresholds, re-probe at both aggregations, then regenerate the Δ
# table and 4-panel plot for A4.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/a1_recal_a4_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

say "=== A1: concat-latent probe ==="
t0=$(date +%s)
.venv/bin/python -u experiments/phase5_downstream_utility/analysis/concat_probe.py \
    > "$LOG_DIR/concat_probe.log" 2>&1
say "  concat_probe exit=$? in $(( $(date +%s) - t0 ))s"

say "=== Recalibrating T15/T20 BatchTopK thresholds ==="
t0=$(date +%s)
.venv/bin/python -u experiments/phase5_downstream_utility/analysis/recalibrate_batchtopk_threshold.py \
    txcdr_t15_batchtopk txcdr_t20_batchtopk \
    > "$LOG_DIR/recal_run.log" 2>&1
say "  recalibrate exit=$? in $(( $(date +%s) - t0 ))s"

say "=== Re-probing recalibrated ckpts (both aggregations) ==="
for AGG in last_position mean_pool; do
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "$AGG" --skip-baselines \
        --run-ids txcdr_t15_batchtopk_recal__seed42 txcdr_t20_batchtopk_recal__seed42 \
        > "$LOG_DIR/probe_recal_${AGG}.log" 2>&1
    say "  probe recal $AGG exit=$? in $(( $(date +%s) - t0 ))s"
done

say "=== Rebuilding A4 Δ table + 4-panel plot ==="
.venv/bin/python -u experiments/phase5_downstream_utility/analysis/batchtopk_delta_table.py \
    > "$LOG_DIR/delta_table.log" 2>&1
say "  delta_table exit=$?"
.venv/bin/python -u experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep_4panel.py \
    > "$LOG_DIR/4panel_plot.log" 2>&1
say "  4panel_plot exit=$?"
.venv/bin/python -u experiments/phase5_downstream_utility/plots/make_batchtopk_plot.py \
    > "$LOG_DIR/batchtopk_plot.log" 2>&1
say "  batchtopk_plot exit=$?"

say "=== a1_recal_a4 complete ==="
