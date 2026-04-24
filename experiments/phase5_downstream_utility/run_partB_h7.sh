#!/usr/bin/env bash
# Part B H7: bare TXC + anti-dead + matryoshka + MULTI-scale InfoNCE.
# Fuses Phase 6.2 Track 2 + agentic_txc_02 cycle 02 recipe.
# Target: top Fig 1 benchmark (0.8124 lp current top).

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/partB_h7_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

ARCH="phase57_partB_h7_bare_multiscale"

# Train seed=42 first; if beats baseline, add seeds 1, 2.
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

say "=== probing H7 at both aggregations ==="
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
        > "$LOG_DIR/probe_partB_h7_${AGG}.log" 2>&1
    say "  probe $AGG exit=$? in $(( $(date +%s) - t0 ))s"
done

say "=== scoring H7 vs agentic_txc_02 + Fig 1 top ==="
.venv/bin/python -u -c "
import json, statistics as st
from collections import defaultdict
FLIP = {'winogrande_correct_completion', 'wsc_coreference'}
by_arch = defaultdict(list)
with open('/workspace/temp_xc/experiments/phase5_downstream_utility/results/probing_results.jsonl') as f:
    for line in f:
        r = json.loads(line)
        rid = r.get('run_id', '')
        if r.get('k_feat') != 5: continue
        if '$ARCH' not in rid and rid not in (
            'agentic_txc_02__seed42', 'agentic_txc_02__seed1', 'agentic_txc_02__seed2',
            'mlc_contrastive_alpha100_batchtopk__seed42'): continue
        agg = r.get('aggregation')
        auc = r.get('test_auc')
        if auc is None or agg not in ('last_position', 'mean_pool'): continue
        v = float(auc)
        if r['task_name'] in FLIP: v = max(v, 1-v)
        by_arch[(rid, agg)].append(v)
for k, vals in sorted(by_arch.items()):
    print(f'{k[0]:<55s} {k[1]:<15s} mean={st.mean(vals):.4f}  n={len(vals)}')
" 2>&1 | tee -a "$MAIN"

say "=== partB_h7 complete ==="
