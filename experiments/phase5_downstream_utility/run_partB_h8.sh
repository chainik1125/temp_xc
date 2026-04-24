#!/usr/bin/env bash
# Part B H8: bare TXC + anti-dead + matryoshka + MULTI-DISTANCE InfoNCE.
# Contrasts at shift={1, 2} (vs H7's single shift=1).
# Addresses user q: "extend contrastive window length beyond 2".

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/partB_h8_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

ARCH="phase57_partB_h8_bare_multidistance"

for SEED in 42 1 2; do
    RID="${ARCH}__seed${SEED}"
    CKPT="$CKPT_DIR/${RID}.pt"
    if [ -f "$CKPT" ]; then
        say "ckpt exists -> skip: $CKPT"
    else
        say "=== training $RID ==="
        t0=$(date +%s)
        .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[$SEED], max_steps=25000, archs=['$ARCH'])
" > "$LOG_DIR/train_${RID}.log" 2>&1
        say "  train $RID exit=$? in $(( $(date +%s) - t0 ))s"
    fi
done

say "=== probing H8 at both aggregations ==="
RIDS=()
for SEED in 42 1 2; do
    rid="${ARCH}__seed${SEED}"
    [ -f "$CKPT_DIR/${rid}.pt" ] && RIDS+=("$rid")
done
for AGG in last_position mean_pool; do
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "$AGG" --skip-baselines \
        --run-ids "${RIDS[@]}" \
        > "$LOG_DIR/probe_partB_h8_${AGG}.log" 2>&1
    say "  probe $AGG exit=$? in $(( $(date +%s) - t0 ))s"
done

say "=== partB_h8 complete ==="
