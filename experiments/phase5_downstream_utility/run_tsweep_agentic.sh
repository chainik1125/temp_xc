#!/usr/bin/env bash
# Phase 5.7 experiment (iii): T-sweep on agentic_txc_02.
#
# Trains agentic_txc_02_t{2,3,8,10,15,20} with the cycle-02 recipe
# (γ=0.5, n_contr_scales=min(3,T), α=1.0). Probes each at both
# aggregations (test-set). Answers: does cycle-02's T=5 peak shift
# under multi-scale contrastive?

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/tsweep_agentic_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

# Order T ascending — shorter T values train fastest.
CANDS=(
    agentic_txc_02_t2
    agentic_txc_02_t3
    agentic_txc_02_t8
    agentic_txc_02_t10
    agentic_txc_02_t15
    agentic_txc_02_t20
)

for CAND in "${CANDS[@]}"; do
    CKPT="$CKPT_DIR/${CAND}__seed42.pt"
    if [ -f "$CKPT" ]; then
        say "ckpt exists -> skip training: $CKPT"
    else
        say "=== training $CAND (seed=42, 25k max steps) ==="
        t0=$(date +%s)
        .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[42], max_steps=25000, archs=['$CAND'])
" > "$LOG_DIR/train_${CAND}.log" 2>&1
        ec=$?
        say "  train $CAND exit=$ec in $(( $(date +%s) - t0 ))s"
        if [ $ec -ne 0 ]; then
            say "  TRAIN FAILED for $CAND"
            continue
        fi
    fi
done

say "=== probing all 6 T-sweep archs @ last_position + mean_pool ==="
RIDS=()
for CAND in "${CANDS[@]}"; do RIDS+=("${CAND}__seed42"); done
for AGG in last_position mean_pool; do
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "$AGG" --skip-baselines \
        --run-ids "${RIDS[@]}" \
        > "$LOG_DIR/probe_tsweep_agentic_${AGG}.log" 2>&1
    say "  probe $AGG exit=$? in $(( $(date +%s) - t0 ))s"
done
say "=== tsweep agentic run complete ==="
