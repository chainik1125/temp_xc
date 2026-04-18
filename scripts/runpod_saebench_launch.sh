#!/bin/bash
# runpod_saebench_launch.sh — single-entry launcher for the full
# long-training SAEBench sparse-probing benchmark.
#
# What this does (chain of phases, all resumable):
#   1. Gemma activation caching (skips if cache exists)
#   2. Preflight sanity check (self-cleans its ckpt, see B10)
#   3. Long training with plateau early-stop for 10 cells
#        (SAE×2 + MLC×2 + TempXC×6) — runs until loss plateaus or
#        until STEPS upper bound, whichever comes first.
#   4. Probing eval with ordered + shuffled pair per cell, via the
#        new run_eval.py CLI. Per-example predictions persisted for
#        confusion-matrix analysis.
#   5. Summary tables + analysis script output.
#
# Pod requirements:
#   - H100 80GB (any SXM variant) for training speed
#   - Disk: 250 GB persistent volume + 40 GB root disk
#     (see docs/aniket/bench_harness/disk_sizing.md)
#   - HF token + ANTHROPIC_API_KEY + WANDB_API_KEY set in .env
#
# Usage:
#   bash scripts/runpod_saebench_launch.sh
#
# Environment overrides:
#   STEPS=50000             hard upper bound per cell (default 30000)
#   PLATEAU_PCT=0.005       early-stop threshold (default 0.005 = 0.5%)
#   PLATEAU_MIN_STEPS=5000  floor before plateau kicks in (default 5000)
#   T_SWEEP_MAX=20          cap T for TempXC sweep (default 20; 40 doubles compute)
#   SKIP_PREFLIGHT=1        skip the preflight sanity check
#   SKIP_CACHE=1            skip the Gemma activation caching phase
#
# Time budget (H100, default settings):
#   cache:     ~1 h       (6000 × 128 × 5-layer Gemma forward)
#   preflight: ~15 min
#   train:     ~6-10 h    (plateau-dependent; SAE/MLC exit fast, TempXC longer)
#   eval:      ~12 h      (ordered + shuffled × 10 cells × 4 aggs × 8 tasks)
#   TOTAL:     ~20-24 h

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

# Tag the W&B group so all runs from this launch cluster together.
export WANDB_GROUP="${WANDB_GROUP:-saebench-long-$(date +%Y%m%d-%H%M)}"

STEPS="${STEPS:-30000}"
PLATEAU_PCT="${PLATEAU_PCT:-0.005}"
PLATEAU_MIN_STEPS="${PLATEAU_MIN_STEPS:-5000}"
T_SWEEP_MAX="${T_SWEEP_MAX:-20}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SKIP_CACHE="${SKIP_CACHE:-0}"
NUM_SEQS="${NUM_SEQS:-6000}"
SEQ_LEN="${SEQ_LEN:-128}"

export STEPS PLATEAU_PCT PLATEAU_MIN_STEPS

echo "════════════════════════════════════════════════════════════"
echo " SAEBench long-training launch"
echo " W&B group:       $WANDB_GROUP"
echo " STEPS (max):     $STEPS"
echo " plateau pct:     $PLATEAU_PCT"
echo " plateau floor:   $PLATEAU_MIN_STEPS"
echo " T_SWEEP_MAX:     $T_SWEEP_MAX"
echo " skip cache:      $SKIP_CACHE"
echo " skip preflight:  $SKIP_PREFLIGHT"
echo " start:           $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "════════════════════════════════════════════════════════════"

# Pre-flight regression gate (B1-B16 + item 8 hook). Fails fast if a
# known bug has been reintroduced. ~0.1 s.
echo ""
echo ">> regression gate (src/bench/regressions.py)"
python -m src.bench.regressions

# Cell enumeration — same as the old orchestrator.
ARCHS_AT_T5=(sae mlc tempxc)
PROTOCOLS=(A B)
T_SWEEP_VALUES=()
for t in 10 20 40; do
    [ "$t" -le "$T_SWEEP_MAX" ] && T_SWEEP_VALUES+=("$t")
done

phase() {
    echo ""
    echo "──── PHASE: $1 ──── start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# ─── CACHE ───────────────────────────────────────────────────────────
if [ "$SKIP_CACHE" != "1" ]; then
    phase "cache multi-layer activations"
    if ls data/cached_activations/gemma-2-2b/fineweb/resid_L1[0-4].npy >/dev/null 2>&1; then
        n_found=$(ls data/cached_activations/gemma-2-2b/fineweb/resid_L1[0-4].npy 2>/dev/null | wc -l | tr -d ' ')
        if [ "$n_found" -eq 5 ]; then
            echo ">> SKIP — all 5 layer caches already present"
        else
            NUM_SEQS="$NUM_SEQS" SEQ_LEN="$SEQ_LEN" bash scripts/runpod_saebench_cache.sh
        fi
    else
        NUM_SEQS="$NUM_SEQS" SEQ_LEN="$SEQ_LEN" bash scripts/runpod_saebench_cache.sh
    fi
fi

# ─── PREFLIGHT ───────────────────────────────────────────────────────
if [ "$SKIP_PREFLIGHT" != "1" ]; then
    phase "preflight"
    if [ -f results/saebench/preflight/preflight.jsonl ]; then
        echo ">> SKIP — preflight already passed (rm results/saebench/preflight/ to redo)"
    else
        bash scripts/runpod_saebench_preflight.sh
    fi
fi

# ─── TRAIN (10 cells, plateau early-stop) ─────────────────────────────
phase "train 10 checkpoints (long, plateau-detecting)"
t0_train=$(date +%s)

for arch in "${ARCHS_AT_T5[@]}"; do
    for proto in "${PROTOCOLS[@]}"; do
        echo ""
        echo ">> $(date -u +%H:%M:%SZ) | train: arch=$arch protocol=$proto T=5"
        STEPS="$STEPS" PLATEAU_PCT="$PLATEAU_PCT" PLATEAU_MIN_STEPS="$PLATEAU_MIN_STEPS" \
            bash scripts/runpod_saebench_train.sh --arch "$arch" --protocol "$proto" --t 5
    done
done

for t in "${T_SWEEP_VALUES[@]}"; do
    for proto in "${PROTOCOLS[@]}"; do
        echo ""
        echo ">> $(date -u +%H:%M:%SZ) | train: arch=tempxc protocol=$proto T=$t"
        STEPS="$STEPS" PLATEAU_PCT="$PLATEAU_PCT" PLATEAU_MIN_STEPS="$PLATEAU_MIN_STEPS" \
            bash scripts/runpod_saebench_train.sh --arch tempxc --protocol "$proto" --t "$t"
    done
done

t1_train=$(date +%s)
echo ""
echo ">> train phase wall time: $((t1_train - t0_train)) sec"
ls -lh results/saebench/ckpts/ | head -20

# ─── EVAL (ordered + shuffled via run_eval.py) ────────────────────────
phase "probing eval (ordered + shuffled pair)"
t0_eval=$(date +%s)

for arch in "${ARCHS_AT_T5[@]}"; do
    for proto in "${PROTOCOLS[@]}"; do
        echo ""
        echo ">> $(date -u +%H:%M:%SZ) | eval: arch=$arch protocol=$proto T=5"
        bash scripts/runpod_saebench_run_eval.sh --arch "$arch" --protocol "$proto" --t 5 --aggregation all
    done
done

for t in "${T_SWEEP_VALUES[@]}"; do
    for proto in "${PROTOCOLS[@]}"; do
        echo ""
        echo ">> $(date -u +%H:%M:%SZ) | eval: arch=tempxc protocol=$proto T=$t"
        bash scripts/runpod_saebench_run_eval.sh --arch tempxc --protocol "$proto" --t "$t" --aggregation all
    done
done

t1_eval=$(date +%s)
echo ""
echo ">> eval phase wall time: $((t1_eval - t0_eval)) sec"

# ─── SUMMARY ─────────────────────────────────────────────────────────
phase "summary + plots"
python scripts/analyze_saebench.py || echo "analyze_saebench errored (non-fatal)"
python scripts/plot_training_curves.py || echo "plot_training_curves errored (non-fatal)"

echo ""
echo "════════════════════════════════════════════════════════════"
echo " launch done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Push results to aniket branch:"
echo "  bash scripts/runpod_push_results.sh"
