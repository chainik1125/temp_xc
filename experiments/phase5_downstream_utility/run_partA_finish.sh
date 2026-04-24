#!/usr/bin/env bash
# Finish Part A: A5 (T=30 extension) + A3 (3-seed variance Tier 1) + plots.
# Runs everything serially. Total ~4-5 hr wall-clock.

set -u
cd /workspace/temp_xc

export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1

LOG_DIR=/workspace/temp_xc/logs/overnight
CKPT_DIR=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts
mkdir -p "$LOG_DIR"
MAIN="$LOG_DIR/partA_finish_run.log"
say() { echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$MAIN"; }

# ═══════════════════════════════════════════════════════════
# A5: T=30 extension (last_position only — mean_pool infeasible at T>20)
# ═══════════════════════════════════════════════════════════
say "=== A5: train + probe T=30 TXCDR variants ==="
for ARCH in txcdr_t30 txcdr_t30_batchtopk; do
    CKPT="$CKPT_DIR/${ARCH}__seed42.pt"
    if [ -f "$CKPT" ]; then
        say "  ckpt exists -> skip: $CKPT"
    else
        say "  training $ARCH"
        t0=$(date +%s)
        .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[42], max_steps=25000, archs=['$ARCH'])
" > "$LOG_DIR/train_${ARCH}.log" 2>&1
        say "    exit=$? in $(( $(date +%s) - t0 ))s"
    fi
done

say "  probing T=30 archs at last_position"
RIDS=(txcdr_t30__seed42 txcdr_t30_batchtopk__seed42)
RIDS_EXIST=()
for rid in "${RIDS[@]}"; do
    [ -f "$CKPT_DIR/${rid}.pt" ] && RIDS_EXIST+=("$rid")
done
if [ ${#RIDS_EXIST[@]} -gt 0 ]; then
    t0=$(date +%s)
    .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation last_position --skip-baselines \
        --run-ids "${RIDS_EXIST[@]}" \
        > "$LOG_DIR/probe_a5_t30_last_position.log" 2>&1
    say "    probe a5 lp exit=$? in $(( $(date +%s) - t0 ))s"
fi

# ═══════════════════════════════════════════════════════════
# A3: 3-seed variance Tier 1 (core archs only; Tier 2 deferred)
# ═══════════════════════════════════════════════════════════
say "=== A3 Tier 1: seed variance training ==="
# Only train missing ckpts (some already exist from previous work)
declare -a A3_SPEC=(
    "txcdr_t5:2"
    "mlc_contrastive:1"
    "mlc_contrastive:2"
    "matryoshka_t5:1"
    "matryoshka_t5:2"
)
for spec in "${A3_SPEC[@]}"; do
    arch=${spec%%:*}; seed=${spec##*:}
    rid="${arch}__seed${seed}"
    ckpt="$CKPT_DIR/${rid}.pt"
    if [ -f "$ckpt" ]; then
        say "  ckpt exists -> skip: $ckpt"
    else
        say "  training $rid"
        t0=$(date +%s)
        .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[$seed], max_steps=25000, archs=['$arch'])
" > "$LOG_DIR/train_${rid}.log" 2>&1
        say "    exit=$? in $(( $(date +%s) - t0 ))s"
    fi
done

say "  probing A3 ckpts (both aggregations)"
A3_RIDS=()
# Seeds we want probed at both aggs (fill in any missing)
for seed in 1 2; do
    for arch in txcdr_t5 mlc mlc_contrastive matryoshka_t5; do
        rid="${arch}__seed${seed}"
        [ -f "$CKPT_DIR/${rid}.pt" ] && A3_RIDS+=("$rid")
    done
done
for AGG in last_position mean_pool; do
    if [ ${#A3_RIDS[@]} -gt 0 ]; then
        t0=$(date +%s)
        .venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
            --aggregation "$AGG" --skip-baselines \
            --run-ids "${A3_RIDS[@]}" \
            > "$LOG_DIR/probe_a3_${AGG}.log" 2>&1
        say "    probe a3 $AGG exit=$? in $(( $(date +%s) - t0 ))s"
    fi
done

# ═══════════════════════════════════════════════════════════
# A7: plot regen + HF sync
# ═══════════════════════════════════════════════════════════
say "=== A7: regenerating plots + delta table ==="
.venv/bin/python -u experiments/phase5_downstream_utility/plots/make_headline_plot.py > "$LOG_DIR/plot_headline.log" 2>&1
say "  headline plot exit=$?"
.venv/bin/python -u experiments/phase5_downstream_utility/plots/make_batchtopk_plot.py > "$LOG_DIR/plot_batchtopk.log" 2>&1
say "  batchtopk plot exit=$?"
.venv/bin/python -u experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep_4panel.py > "$LOG_DIR/plot_4panel.log" 2>&1
say "  4panel plot exit=$?"
.venv/bin/python -u experiments/phase5_downstream_utility/analysis/batchtopk_delta_table.py > "$LOG_DIR/delta_table.log" 2>&1
say "  delta table exit=$?"

say "=== partA_finish complete ==="
