#!/usr/bin/env bash
# Phase 5.7 experiment (ii) extended-scope: BatchTopK for remaining 17 archs.
# Briefing:
#   docs/han/research_logs/phase5_downstream_utility/
#   2026-04-23-handover-batchtopk-extend.md
#
# Order: small/fast archs first (topk_sae, stacked, small-T txcdr) so any
# OOM or wiring bug fails fast. Large-T TXCDR + T-sweep agentic last.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/batchtopk_extend_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

# Ordered small/fast first.
CANDS=(
    topk_sae_batchtopk
    mlc_contrastive_batchtopk
    mlc_contrastive_alpha100_batchtopk
    stacked_t5_batchtopk
    txcdr_t2_batchtopk
    txcdr_t3_batchtopk
    matryoshka_t5_batchtopk
    matryoshka_txcdr_contrastive_t5_alpha100_batchtopk
    agentic_txc_02_t2_batchtopk
    agentic_txc_02_t3_batchtopk
    time_layer_crosscoder_t5_batchtopk
    txcdr_t8_batchtopk
    agentic_txc_02_t8_batchtopk
    stacked_t20_batchtopk
    txcdr_t10_batchtopk
    txcdr_t15_batchtopk
    txcdr_t20_batchtopk
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
            say "  TRAIN FAILED for $CAND — continuing"
        fi
    fi
done

say "=== probing all trained BatchTopK extended archs ==="
RIDS=()
for CAND in "${CANDS[@]}"; do
    if [ -f "$CKPT_DIR/${CAND}__seed42.pt" ]; then
        RIDS+=("${CAND}__seed42")
    fi
done
for AGG in last_position mean_pool; do
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "$AGG" --skip-baselines \
        --run-ids "${RIDS[@]}" \
        > "$LOG_DIR/probe_batchtopk_extend_${AGG}.log" 2>&1
    say "  probe $AGG exit=$? in $(( $(date +%s) - t0 ))s"
done
say "=== batchtopk_extend run complete ==="
