#!/usr/bin/env bash
# Orchestrates: wait for current probing → train T-sweep → probe T-sweep →
# regenerate plots. Runs in background; polling-based handoff between steps.
set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export UV_LINK_MODE=copy
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/fw_tsweep_orchestrator.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

say "=== ORCHESTRATOR START ==="

# ─── STEP A: wait for the MLC + time_layer full_window probing to finish. ───
say "waiting for STEP A (run_probing.py for mlc + time_layer) to finish..."
while pgrep -f "run_probing.py --aggregation full_window --run-ids mlc__seed42" > /dev/null; do
    sleep 30
done
say "STEP A done"

# ─── STEP B: train txcdr_t{2,3,8,10,15} sequentially. ───
say "STEP B: training 5 new TXCDR T-sweep archs"
t0=$(date +%s)
.venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[42], max_steps=25000,
        archs=['txcdr_t2','txcdr_t3','txcdr_t8','txcdr_t10','txcdr_t15'])
" > "$LOG_DIR/train_t_sweep.log" 2>&1
ec=$?
dt=$(( $(date +%s) - t0 ))
say "STEP B: training finished in ${dt}s (exit=$ec)"
if [ $ec -ne 0 ]; then
    say "STEP B FAILED — aborting"
    exit 1
fi

# ─── STEP C: probe each new arch at both aggregations. ───
say "STEP C: probing 5 new archs × 2 aggregations"
for arch in txcdr_t2 txcdr_t3 txcdr_t8 txcdr_t10 txcdr_t15; do
    for agg in last_position full_window; do
        say "  probing ${arch}__seed42 @ ${agg}"
        t0=$(date +%s)
        .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
            --aggregation "$agg" --skip-baselines --run-ids "${arch}__seed42" \
            > "$LOG_DIR/probe_${arch}_${agg}.log" 2>&1
        dt=$(( $(date +%s) - t0 ))
        say "    done (${dt}s)"
    done
done

# ─── STEP D: regenerate plots (headline + T-sweep). ───
say "STEP D: plots"
bash experiments/phase5_downstream_utility/regenerate_plots.sh > "$LOG_DIR/plots_after_tsweep.log" 2>&1
.venv/bin/python -u experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep.py \
    >> "$LOG_DIR/plots_after_tsweep.log" 2>&1

say "=== ORCHESTRATOR END ==="
