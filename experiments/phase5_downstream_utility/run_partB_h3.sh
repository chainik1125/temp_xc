#!/usr/bin/env bash
# Part B cycle 2 — Hypothesis H3: log-scale matryoshka.
# Escapes the O(T³·d_in) decoder OOM that stopped PositionMatryoshkaTXCDR
# at T≥10. Uses log-spaced scales {1, 2, 4, 8, 16, 32} — truncated to ≤T.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/partB_h3_logmatryoshka_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

T_VALUES=(5 10 15 20 30)

for T in "${T_VALUES[@]}"; do
    ARCH="log_matryoshka_t${T}"
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

say "=== probing H3 T-sweep at last_position ==="
RIDS=()
for T in "${T_VALUES[@]}"; do
    rid="log_matryoshka_t${T}__seed42"
    [ -f "$CKPT_DIR/${rid}.pt" ] && RIDS+=("$rid")
done
t0=$(date +%s)
.venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
    --aggregation last_position --skip-baselines \
    --run-ids "${RIDS[@]}" \
    > "$LOG_DIR/probe_partB_h3_last_position.log" 2>&1
say "  probe lp exit=$? in $(( $(date +%s) - t0 ))s"

# mean_pool for T<=20
RIDS_MP=()
for T in 5 10 15 20; do
    rid="log_matryoshka_t${T}__seed42"
    [ -f "$CKPT_DIR/${rid}.pt" ] && RIDS_MP+=("$rid")
done
if [ ${#RIDS_MP[@]} -gt 0 ]; then
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation mean_pool --skip-baselines \
        --run-ids "${RIDS_MP[@]}" \
        > "$LOG_DIR/probe_partB_h3_mean_pool.log" 2>&1
    say "  probe mp exit=$? in $(( $(date +%s) - t0 ))s"
fi

say "=== scoring H3 T-scaling ==="
.venv/bin/python -u -c "
import sys
sys.path.insert(0, '/workspace/temp_xc')
from experiments.phase5_downstream_utility.analysis.t_scaling_score import score_arch_family
for agg in ('last_position', 'mean_pool'):
    r = score_arch_family('log_matryoshka_t', [5, 10, 15, 20, 30], aggregation=agg)
    print(f'H3 log_matryoshka | {agg}: {r}')
" 2>&1 | tee -a "$MAIN"

say "=== partB_h3 complete ==="
