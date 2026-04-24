#!/usr/bin/env bash
# Phase 5.7 A5: extend T-sweep beyond T=20.
# Train txcdr_t{24,28,32,36} (TopK) + _batchtopk variants, probe last_position.
# mean_pool blocked at T>20 because acts_anchor LAST_N=20.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/tsweep_extended_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

# Order: TopK first (smaller params), then BatchTopK. Alternate T ascending
# so if OOM hits it hits at the largest T and we still have the other pairs.
CANDS=(
    txcdr_t30
    txcdr_t30_batchtopk
)
# (A5 reduced to T=30 only for time budget. T ∈ {24, 28, 32, 36}
# deferred — a single T=30 point is sufficient to answer "does curve
# continue rising past T=20 or plateau?" For the matryoshka T=30 Part B
# reference point, T=30 vanilla TXCDR is the direct comparison.)

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
            say "  TRAIN FAILED for $CAND — continuing"
        fi
    fi
done

say "=== probing T>20 archs (last_position ONLY; mean_pool infeasible at T>LAST_N=20) ==="
RIDS=()
for CAND in "${CANDS[@]}"; do
    if [ -f "$CKPT_DIR/${CAND}__seed42.pt" ]; then
        RIDS+=("${CAND}__seed42")
    fi
done
if [ ${#RIDS[@]} -gt 0 ]; then
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation last_position --skip-baselines \
        --run-ids "${RIDS[@]}" \
        > "$LOG_DIR/probe_tsweep_extended_last_position.log" 2>&1
    say "  probe last_position exit=$? in $(( $(date +%s) - t0 ))s"
fi
say "=== tsweep_extended run complete ==="
