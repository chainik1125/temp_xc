#!/usr/bin/env bash
# Train + probe mlc_contrastive (Ye et al. 2025 port to MLC base).
# Waits for the mean_pool orchestrator to finish before starting, since GPU
# and RAM are shared. Probes at last_position + mean_pool (the two relevant
# aggregations — full_window is not requested for this variant).
#
# Single new checkpoint ~850 MB; two probing passes at ~10-15 min each.
set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export UV_LINK_MODE=copy
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/mlc_contrastive_orchestrator.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

say "=== MLC_CONTRASTIVE ORCHESTRATOR START ==="

# Wait for the mean_pool orchestrator to finish (GPU + RAM contention).
say "waiting for mean_pool orchestrator (run_mean_pool_probing.sh) to finish..."
while pgrep -f "run_mean_pool_probing.sh" > /dev/null; do
    sleep 30
done
say "mean_pool orchestrator done — proceeding"

# ── Disk quota guard ──────────────────────────────────────────────
# User flagged 60 GB remaining on the moosefs quota. A new ckpt is
# ~850 MB + a training log ~10 KB, so well within budget, but we
# abort if free space drops under 5 GB during the run.
QUOTA_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results
free_gb() { df -BG "$1" | awk 'NR==2{gsub(/G/,"",$4); print $4}'; }
pre=$(free_gb "$QUOTA_DIR")
say "pre-run free space on $QUOTA_DIR: ${pre} GB"
if [ "${pre:-0}" -lt 5 ]; then
    say "FATAL: < 5 GB free — aborting before training to avoid quota fail"
    exit 1
fi

# ── STEP 1: train mlc_contrastive (single arch) ────────────────────
say "STEP 1: training mlc_contrastive (d_sae=18432, k=100, α=0.1, h=9216)"
t0=$(date +%s)
.venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[42], max_steps=25000, archs=['mlc_contrastive'])
" > "$LOG_DIR/train_mlc_contrastive.log" 2>&1
ec=$?
dt=$(( $(date +%s) - t0 ))
say "STEP 1 done in ${dt}s (exit=$ec)"
if [ $ec -ne 0 ]; then
    say "training FAILED — aborting, see $LOG_DIR/train_mlc_contrastive.log"
    exit 1
fi

# ── STEP 2: probe at last_position + mean_pool ──────────────────────
for agg in last_position mean_pool; do
    say "STEP 2: probing mlc_contrastive__seed42 @ ${agg}"
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "$agg" --skip-baselines \
        --run-ids mlc_contrastive__seed42 \
        > "$LOG_DIR/probe_mlc_contrastive_${agg}.log" 2>&1
    dt=$(( $(date +%s) - t0 ))
    say "  done (${dt}s)"
done

# ── STEP 3: regenerate all plots (now includes mlc_contrastive bars) ─
say "STEP 3: regenerating plots"
.venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py \
    > "$LOG_DIR/plots_after_mlc_contrastive.log" 2>&1

post=$(free_gb "$QUOTA_DIR")
say "post-run free space: ${post} GB (Δ=-$((pre - post)) GB)"
say "=== MLC_CONTRASTIVE ORCHESTRATOR END ==="
