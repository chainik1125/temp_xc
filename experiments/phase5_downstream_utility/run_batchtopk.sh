#!/usr/bin/env bash
# Phase 5.7 experiment (ii): BatchTopK apples-to-apples.
#
# Trains 4 BatchTopK variants: txcdr_t5_batchtopk, mlc_batchtopk,
# agentic_txc_02_batchtopk, agentic_mlc_08_batchtopk. Probes each at
# both aggregations (test-set, not val). Comparison vs the TopK
# baselines tells us whether the multi-scale recipe composes with
# BatchTopK sparsity.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/batchtopk_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

CANDS=(
    txcdr_t5_batchtopk
    mlc_batchtopk
    agentic_txc_02_batchtopk
    agentic_mlc_08_batchtopk
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

say "=== probing all 4 BatchTopK archs @ last_position + mean_pool ==="
for AGG in last_position mean_pool; do
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "$AGG" --skip-baselines \
        --run-ids txcdr_t5_batchtopk__seed42 mlc_batchtopk__seed42 \
                 agentic_txc_02_batchtopk__seed42 agentic_mlc_08_batchtopk__seed42 \
        > "$LOG_DIR/probe_batchtopk_${AGG}.log" 2>&1
    say "  probe $AGG exit=$? in $(( $(date +%s) - t0 ))s"
done
say "=== batchtopk run complete ==="
