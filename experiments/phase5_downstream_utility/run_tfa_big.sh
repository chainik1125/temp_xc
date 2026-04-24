#!/usr/bin/env bash
# Phase 5.7 experiment (i): full-size TFA training + probing.
#
# Trains tfa_big + tfa_pos_big at d_sae=18432, seq_len=128, 25k max steps,
# plateau-stop. Probes both at last_position + mean_pool. Also probes
# the `_full` dual-probing variants (z_novel + z_pred) via ckpt symlinks.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/tfa_big_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

for CAND in tfa_big tfa_pos_big; do
    CKPT="$CKPT_DIR/${CAND}__seed42.pt"
    if [ -f "$CKPT" ]; then
        say "ckpt exists -> skip training: $CKPT"
    else
        say "=== training $CAND (seed=42, 25k max steps) ==="
        t0=$(date +%s)
        .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[42], max_steps=8000, archs=['$CAND'])
" > "$LOG_DIR/train_${CAND}.log" 2>&1
        ec=$?
        say "  train $CAND exit=$ec in $(( $(date +%s) - t0 ))s"
        if [ $ec -ne 0 ]; then
            say "  TRAIN FAILED for $CAND — skipping"
            continue
        fi
    fi
    # Create symlink for dual probing variant
    if [ ! -e "${CKPT%.*}_full.pt" ]; then
        ln -sf "${CAND}__seed42.pt" "$CKPT_DIR/${CAND}_full__seed42.pt"
        say "  symlinked ${CAND}_full__seed42.pt -> ${CAND}__seed42.pt"
    fi
done

say "=== probing tfa_big / tfa_pos_big @ last_position + mean_pool ==="
for AGG in last_position mean_pool; do
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "$AGG" --skip-baselines \
        --run-ids tfa_big__seed42 tfa_big_full__seed42 \
                 tfa_pos_big__seed42 tfa_pos_big_full__seed42 \
        > "$LOG_DIR/probe_tfa_big_${AGG}.log" 2>&1
    say "  probe $AGG exit=$? in $(( $(date +%s) - t0 ))s"
done
say "=== tfa_big run complete ==="
