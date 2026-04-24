#!/usr/bin/env bash
# Part B H10 + H12: encoder-variant ablations on vanilla TXC at T=5.
# Tests whether TXC's per-position W_enc (T, d_in, d_sae) is load-bearing
# or redundant given simpler shared-encoder + positional-prior setups.
#
# H10a: shared W_enc + pos embed + ReLU per pos + sum over T
# H10b: H10 without pos embed (control)
# H12:  shared W_1 + pos embed → concat T outputs → W_2 → TopK
#
# All at T=5, seed=42. Probe at both aggregations.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/partB_h10_h12_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

CANDS=(
    txc_shared_relu_sum_pos_t5
    txc_shared_relu_sum_nopos_t5
    txc_shared_concat_two_layer_t5
)

for ARCH in "${CANDS[@]}"; do
    RID="${ARCH}__seed42"
    CKPT="$CKPT_DIR/${RID}.pt"
    if [ -f "$CKPT" ]; then
        say "ckpt exists -> skip: $CKPT"
    else
        say "=== training $RID ==="
        t0=$(date +%s)
        .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[42], max_steps=25000, archs=['$ARCH'])
" > "$LOG_DIR/train_${RID}.log" 2>&1
        say "  train $RID exit=$? in $(( $(date +%s) - t0 ))s"
    fi
done

say "=== probing H10/H12 at both aggregations ==="
RIDS=()
for ARCH in "${CANDS[@]}"; do
    rid="${ARCH}__seed42"
    [ -f "$CKPT_DIR/${rid}.pt" ] && RIDS+=("$rid")
done
for AGG in last_position mean_pool; do
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "$AGG" --skip-baselines \
        --run-ids "${RIDS[@]}" \
        > "$LOG_DIR/probe_partB_h10_h12_${AGG}.log" 2>&1
    say "  probe $AGG exit=$? in $(( $(date +%s) - t0 ))s"
done

say "=== partB_h10_h12 complete ==="
