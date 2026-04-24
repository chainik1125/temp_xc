#!/usr/bin/env bash
# Part B cycle 1 — Hypothesis H1: translation-invariant conv encoder.
# Train ConvTXCDR at T ∈ {5, 10, 15, 20, 30} seed=42 and probe at
# last_position. Score via t_scaling_score.py.
#
# Targets: monotonicity ≥ 0.8, Δ(T=30 − T=5) > +0.02.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/partB_h1_conv_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

# T-sweep at matched per-window k = 100·T.
T_VALUES=(5 10 15 20 30)

for T in "${T_VALUES[@]}"; do
    ARCH="conv_txcdr_t${T}"
    CKPT="$CKPT_DIR/${ARCH}__seed42.pt"
    if [ -f "$CKPT" ]; then
        say "ckpt exists -> skip: $CKPT"
    else
        say "=== training $ARCH ==="
        t0=$(date +%s)
        .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[42], max_steps=25000, archs=['$ARCH'])
" > "$LOG_DIR/train_${ARCH}.log" 2>&1
        say "  train $ARCH exit=$? in $(( $(date +%s) - t0 ))s"
    fi
done

say "=== probing ConvTXCDR T-sweep at last_position ==="
RIDS=()
for T in "${T_VALUES[@]}"; do
    rid="conv_txcdr_t${T}__seed42"
    if [ -f "$CKPT_DIR/${rid}.pt" ]; then
        RIDS+=("$rid")
    fi
done
t0=$(date +%s)
.venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
    --aggregation last_position --skip-baselines \
    --run-ids "${RIDS[@]}" \
    > "$LOG_DIR/probe_partB_h1_last_position.log" 2>&1
say "  probe lp exit=$? in $(( $(date +%s) - t0 ))s"

# Probe at mean_pool for T <= 20 (T > 20 infeasible — tail-20 cache).
RIDS_MP=()
for T in 5 10 15 20; do
    rid="conv_txcdr_t${T}__seed42"
    if [ -f "$CKPT_DIR/${rid}.pt" ]; then
        RIDS_MP+=("$rid")
    fi
done
if [ ${#RIDS_MP[@]} -gt 0 ]; then
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation mean_pool --skip-baselines \
        --run-ids "${RIDS_MP[@]}" \
        > "$LOG_DIR/probe_partB_h1_mean_pool.log" 2>&1
    say "  probe mp exit=$? in $(( $(date +%s) - t0 ))s"
fi

say "=== scoring H1 T-scaling ==="
.venv/bin/python -u -c "
import sys
sys.path.insert(0, '/workspace/temp_xc')
from experiments.phase5_downstream_utility.analysis.t_scaling_score import score_arch_family
for agg in ('last_position', 'mean_pool'):
    r = score_arch_family('conv_txcdr_t', [5, 10, 15, 20, 30], aggregation=agg)
    print(f'H1 conv_txcdr | {agg}: {r}')
" 2>&1 | tee -a "$MAIN"

say "=== partB_h1 complete ==="
