#!/usr/bin/env bash
# Mean-pool probing: waits for the T-sweep orchestrator to finish, then
# probes all 24 archs (19 original + 5 T-sweep) at aggregation=mean_pool.
# Matches SAEBench / Kantamneni's canonical aggregation — averages the
# K per-slide d_sae vectors produced by full_window into one vector per
# example. Reuses existing caches and model checkpoints; one extra
# numpy reshape+mean per arch over what full_window already computes.
set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export UV_LINK_MODE=copy
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/mean_pool_orchestrator.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

say "=== MEAN_POOL ORCHESTRATOR START ==="

# Wait for the T-sweep orchestrator to finish before starting (GPU + RAM).
say "waiting for T-sweep orchestrator (run_fw_tsweep.sh) to finish..."
while pgrep -f "run_fw_tsweep.sh" > /dev/null; do
    sleep 30
done
say "T-sweep orchestrator done — starting mean_pool probing"

ARCHS=(
    "topk_sae__seed42"
    "mlc__seed42"
    "txcdr_t5__seed42"
    "txcdr_t20__seed42"
    "stacked_t5__seed42"
    "stacked_t20__seed42"
    "matryoshka_t5__seed42"
    "txcdr_shared_dec_t5__seed42"
    "txcdr_shared_enc_t5__seed42"
    "txcdr_tied_t5__seed42"
    "txcdr_pos_t5__seed42"
    "txcdr_causal_t5__seed42"
    "txcdr_block_sparse_t5__seed42"
    "txcdr_lowrank_dec_t5__seed42"
    "txcdr_rank_k_dec_t5__seed42"
    "temporal_contrastive__seed42"
    "tfa_small__seed42"
    "tfa_pos_small__seed42"
    "time_layer_crosscoder_t5__seed42"
    "txcdr_t2__seed42"
    "txcdr_t3__seed42"
    "txcdr_t8__seed42"
    "txcdr_t10__seed42"
    "txcdr_t15__seed42"
)

# First run: ONE python invocation probes all 24 archs + emits baselines
# (36/36 coverage at aggregation=mean_pool tag) in a single pass. The
# per-task cache streams; peak RAM matches the full_window run since
# mean_pool delegates to that path internally.
say "probing all 24 archs + baselines at mean_pool"
t0=$(date +%s)
.venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
    --aggregation mean_pool \
    --run-ids "${ARCHS[@]}" \
    > "$LOG_DIR/probe_mean_pool.log" 2>&1
ec=$?
dt=$(( $(date +%s) - t0 ))
say "  mean_pool probing finished in ${dt}s (exit=$ec)"

# Regenerate plots — the plotter now iterates over 3 aggregations,
# producing 8 new mean_pool plots (4 bar + 4 per-task × 2 task sets × 2 metrics).
say "regenerating plots (now including mean_pool)"
.venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py \
    >> "$LOG_DIR/plots_after_mean_pool.log" 2>&1
.venv/bin/python experiments/phase5_downstream_utility/plots/plot_training_curves.py \
    >> "$LOG_DIR/plots_after_mean_pool.log" 2>&1

say "=== MEAN_POOL ORCHESTRATOR END ==="
