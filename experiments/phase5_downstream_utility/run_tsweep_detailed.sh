#!/usr/bin/env bash
# Detailed T-sweep: fill in the low-T gap (T=6,7) and high-T range (T=24,28).
# T=32, T=36 tentative — same OOM envelope as T=30 which failed.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/tsweep_detailed_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

# Order: small fast first (T=6,7 and A3 Tier 2) so OOMs (if any) happen late.
# T=30, 32, 36 skipped — proven to OOM at d_sae=18432 on A40.
CANDS=(
    # Low-T fill-in (user ask, handover granularity)
    txcdr_t6
    txcdr_t7
    txcdr_t6_batchtopk
    txcdr_t7_batchtopk
    agentic_txc_02_t6
    agentic_txc_02_t7
    # A3 Tier 2 — agentic BatchTopK seed variance
    # (handled below in separate seed loop)
    # High-T extension (A5; T=30,32,36 OOM — skip)
    txcdr_t24
    txcdr_t24_batchtopk
    txcdr_t28
    txcdr_t28_batchtopk
)

# A3 Tier 2: agentic_*_batchtopk seeds 1, 2 (handover's full A3 scope)
A3_TIER2=(
    "agentic_txc_02_batchtopk:1"
    "agentic_txc_02_batchtopk:2"
    "agentic_mlc_08_batchtopk:1"
    "agentic_mlc_08_batchtopk:2"
)

for CAND in "${CANDS[@]}"; do
    CKPT="$CKPT_DIR/${CAND}__seed42.pt"
    if [ -f "$CKPT" ]; then
        say "ckpt exists -> skip: $CKPT"
    else
        say "=== training $CAND ==="
        t0=$(date +%s)
        .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[42], max_steps=25000, archs=['$CAND'])
" > "$LOG_DIR/train_${CAND}.log" 2>&1
        ec=$?
        say "  train $CAND exit=$ec in $(( $(date +%s) - t0 ))s"
    fi
done

# A3 Tier 2 trainings
for spec in "${A3_TIER2[@]}"; do
    arch=${spec%%:*}; seed=${spec##*:}
    rid="${arch}__seed${seed}"
    CKPT="$CKPT_DIR/${rid}.pt"
    if [ -f "$CKPT" ]; then
        say "ckpt exists -> skip: $CKPT"
    else
        say "=== training $rid (A3 Tier 2) ==="
        t0=$(date +%s)
        .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[$seed], max_steps=25000, archs=['$arch'])
" > "$LOG_DIR/train_${rid}.log" 2>&1
        say "  train $rid exit=$? in $(( $(date +%s) - t0 ))s"
    fi
done

# Probe: last_position for T>20 (mean_pool infeasible); both for T<=20.
say "=== probing detailed T-sweep at last_position ==="
RIDS_LP=()
for CAND in "${CANDS[@]}"; do
    [ -f "$CKPT_DIR/${CAND}__seed42.pt" ] && RIDS_LP+=("${CAND}__seed42")
done
# Add A3 Tier 2 to lp probing
for spec in "${A3_TIER2[@]}"; do
    arch=${spec%%:*}; seed=${spec##*:}
    rid="${arch}__seed${seed}"
    [ -f "$CKPT_DIR/${rid}.pt" ] && RIDS_LP+=("$rid")
done
if [ ${#RIDS_LP[@]} -gt 0 ]; then
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation last_position --skip-baselines \
        --run-ids "${RIDS_LP[@]}" \
        > "$LOG_DIR/probe_tsweep_detailed_last_position.log" 2>&1
    say "  probe lp exit=$? in $(( $(date +%s) - t0 ))s"
fi

say "=== probing detailed T-sweep at mean_pool (T<=20 only) ==="
RIDS_MP=()
for CAND in txcdr_t6 txcdr_t7 txcdr_t6_batchtopk txcdr_t7_batchtopk \
           agentic_txc_02_t6 agentic_txc_02_t7; do
    [ -f "$CKPT_DIR/${CAND}__seed42.pt" ] && RIDS_MP+=("${CAND}__seed42")
done
# A3 Tier 2 at mean_pool as well
for spec in "${A3_TIER2[@]}"; do
    arch=${spec%%:*}; seed=${spec##*:}
    rid="${arch}__seed${seed}"
    [ -f "$CKPT_DIR/${rid}.pt" ] && RIDS_MP+=("$rid")
done
if [ ${#RIDS_MP[@]} -gt 0 ]; then
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation mean_pool --skip-baselines \
        --run-ids "${RIDS_MP[@]}" \
        > "$LOG_DIR/probe_tsweep_detailed_mean_pool.log" 2>&1
    say "  probe mp exit=$? in $(( $(date +%s) - t0 ))s"
fi

# Regen the 4-panel plot with full data.
.venv/bin/python -u experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep_4panel.py > /dev/null 2>&1

say "=== tsweep_detailed complete ==="
