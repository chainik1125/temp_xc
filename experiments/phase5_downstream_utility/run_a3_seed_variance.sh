#!/usr/bin/env bash
# A3: 3-seed variance on baselines + headline winners (handover 2026-04-23).
# Current state (from probing_results.jsonl):
#   ✓ agentic_{txc_02,mlc_08} seeds 1,2 — fully probed
#   △ mlc seeds 1,2 — last_position only
#   △ txcdr_t5 seed 1 — last_position only; seed 2 — nothing
#   ✗ matryoshka_t5 seeds 1,2 — missing
#   ✗ mlc_contrastive seeds 1,2 — missing
#   ✗ agentic_{txc_02,mlc_08}_batchtopk seeds 1,2 — missing
#
# Tier 1 (essential for headline σ): txcdr_t5 seed 2, mlc + mlc_contrastive
#   seeds 1,2. Probe all at both aggregations.
# Tier 2 (if time): matryoshka_t5, agentic_*_batchtopk seeds 1,2.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/a3_seed_variance_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

TIER1_TRAINS=(
    "txcdr_t5:2"
    "mlc_contrastive:1"
    "mlc_contrastive:2"
    "matryoshka_t5:1"
    "matryoshka_t5:2"
)
TIER2_TRAINS=(
    "agentic_txc_02_batchtopk:1"
    "agentic_txc_02_batchtopk:2"
    "agentic_mlc_08_batchtopk:1"
    "agentic_mlc_08_batchtopk:2"
)

train_one() {
    local arch=$1 seed=$2
    local rid="${arch}__seed${seed}"
    local ckpt="$CKPT_DIR/${rid}.pt"
    if [ -f "$ckpt" ]; then
        say "  ckpt exists: $ckpt"
        return 0
    fi
    say "=== training $rid ==="
    t0=$(date +%s)
    .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[$seed], max_steps=25000, archs=['$arch'])
" > "$LOG_DIR/train_${rid}.log" 2>&1
    ec=$?
    say "  train $rid exit=$ec in $(( $(date +%s) - t0 ))s"
}

# Run Tier 1 first, then Tier 2.
for spec in "${TIER1_TRAINS[@]}" "${TIER2_TRAINS[@]}"; do
    arch=${spec%%:*}; seed=${spec##*:}
    train_one "$arch" "$seed"
done

say "=== probing all seed-variance ckpts at both aggregations ==="
# Build RID list of all seed-1/2 ckpts we care about
RIDS=()
for spec in "${TIER1_TRAINS[@]}" "${TIER2_TRAINS[@]}"; do
    arch=${spec%%:*}; seed=${spec##*:}
    rid="${arch}__seed${seed}"
    if [ -f "$CKPT_DIR/${rid}.pt" ]; then
        RIDS+=("$rid")
    fi
done
# Also re-probe baselines at mean_pool where missing
for rid in mlc__seed1 mlc__seed2 txcdr_t5__seed1; do
    if [ -f "$CKPT_DIR/${rid}.pt" ]; then
        RIDS+=("$rid")
    fi
done

for AGG in last_position mean_pool; do
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "$AGG" --skip-baselines \
        --run-ids "${RIDS[@]}" \
        > "$LOG_DIR/probe_a3_seed_variance_${AGG}.log" 2>&1
    say "  probe $AGG exit=$? in $(( $(date +%s) - t0 ))s"
done
say "=== A3 seed variance run complete ==="
