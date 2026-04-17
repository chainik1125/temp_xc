#!/bin/bash
# runpod_saebench_orchestrator.sh — launch-and-leave driver for the
# full SAEBench sparse-probing experiment. Chains caching → pre-flight
# → training (10 checkpoints) → probing (all aggregations) → summary.
#
# Designed to run overnight on a single H100 80GB. Expect ~12-16 hours
# for the full sweep with default settings.
#
# **Resumable**: every step checks for existing outputs before re-running.
# Safe to Ctrl+C and re-launch; resumes from the next missing piece.
#
# Usage:
#   # Full launch (default: all phases, all architectures, T ∈ {5,10,20}):
#   bash scripts/runpod_saebench_orchestrator.sh full
#
#   # Phase-by-phase:
#   bash scripts/runpod_saebench_orchestrator.sh cache      # only multi-layer cache
#   bash scripts/runpod_saebench_orchestrator.sh preflight  # only the 30-min sanity check
#   bash scripts/runpod_saebench_orchestrator.sh train      # only training (all 10 ckpts)
#   bash scripts/runpod_saebench_orchestrator.sh eval       # only probing evals
#   bash scripts/runpod_saebench_orchestrator.sh summary    # only the final JSONL aggregation
#
# Env overrides:
#   T_SWEEP_MAX=20        cap T-sweep at this value (default 20; drops T=40)
#   STEPS=5000            training steps per checkpoint (default 5000)
#   SKIP_PREFLIGHT=1      skip the 30-min pre-flight (don't do this first time)
#   NUM_SEQS=6000         FineWeb sequences to cache (default 6000 at seq_len=128)
#   SEQ_LEN=128           token sequence length (default 128 — matches probing)
#
# Time budget (default settings, H100 80GB):
#   cache:     ~1 h       (6000 × 128 × 5-layer Gemma forward)
#   preflight: ~30 min
#   train:     ~5 h       (SAE×2 + MLC×2 + TempXC×{5,10,20}×2 = 10 ckpts)
#   eval:      ~7 h       (10 ckpts × 4 aggregations, Gemma-cache amortized)
#   ------------------
#   TOTAL:     ~13-14 h

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

PHASE="${1:-full}"
T_SWEEP_MAX="${T_SWEEP_MAX:-20}"
STEPS="${STEPS:-5000}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
NUM_SEQS="${NUM_SEQS:-6000}"
SEQ_LEN="${SEQ_LEN:-128}"

echo "════════════════════════════════════════════════════════════"
echo " SAEBench sparse-probing orchestrator"
echo " phase:         $PHASE"
echo " T_sweep max:   $T_SWEEP_MAX"
echo " training steps:$STEPS"
echo " skip preflight:$SKIP_PREFLIGHT"
echo " start:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "════════════════════════════════════════════════════════════"

# Derive the checkpoint list once so training/eval agree.
# 10 total at T_SWEEP_MAX=20: (SAE×2) + (MLC×2) + (TempXC × {5,10,20} × 2 protocols = 6).
ARCHS_AT_T5=(sae mlc tempxc)
PROTOCOLS=(A B)
T_SWEEP_VALUES=()
for t in 10 20 40; do
    [ "$t" -le "$T_SWEEP_MAX" ] && T_SWEEP_VALUES+=("$t")
done

log_phase() {
    echo ""
    echo "──── PHASE: $1 ──── start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# ─── CACHE ───────────────────────────────────────────────────────────
run_cache() {
    log_phase "cache multi-layer activations"
    if ls data/cached_activations/gemma-2-2b/fineweb/resid_L1[0-4].npy >/dev/null 2>&1; then
        local n_found
        n_found=$(ls data/cached_activations/gemma-2-2b/fineweb/resid_L1[0-4].npy 2>/dev/null | wc -l | tr -d ' ')
        if [ "$n_found" -eq 5 ]; then
            echo ">> SKIP — all 5 layer caches already present"
            return 0
        fi
    fi
    NUM_SEQS="$NUM_SEQS" SEQ_LEN="$SEQ_LEN" bash scripts/runpod_saebench_cache.sh
}

# ─── PRE-FLIGHT ──────────────────────────────────────────────────────
run_preflight() {
    log_phase "pre-flight validation"
    if [ "$SKIP_PREFLIGHT" = "1" ]; then
        echo ">> SKIP — SKIP_PREFLIGHT=1"
        return 0
    fi
    if [ -f results/saebench/preflight/preflight.jsonl ]; then
        echo ">> SKIP — pre-flight already passed (remove results/saebench/preflight/ to re-run)"
        return 0
    fi
    bash scripts/runpod_saebench_preflight.sh
}

# ─── TRAINING ────────────────────────────────────────────────────────
train_all() {
    log_phase "train 10 checkpoints"
    local total_start=$(date +%s)

    # Base grid at T=5: SAE×2, MLC×2, TempXC-T5×2
    for arch in "${ARCHS_AT_T5[@]}"; do
        for proto in "${PROTOCOLS[@]}"; do
            echo ""
            echo ">> $(date -u +%H:%M:%SZ) | train: arch=$arch protocol=$proto T=5"
            STEPS="$STEPS" bash scripts/runpod_saebench_train.sh \
                --arch "$arch" --protocol "$proto" --t 5
        done
    done

    # T-sweep: TempXC only at T ∈ T_SWEEP_VALUES
    for t in "${T_SWEEP_VALUES[@]}"; do
        for proto in "${PROTOCOLS[@]}"; do
            echo ""
            echo ">> $(date -u +%H:%M:%SZ) | train: arch=tempxc protocol=$proto T=$t"
            STEPS="$STEPS" bash scripts/runpod_saebench_train.sh \
                --arch tempxc --protocol "$proto" --t "$t"
        done
    done

    local total_end=$(date +%s)
    echo ""
    echo ">> training phase wall time: $((total_end - total_start)) sec"
    ls -lh results/saebench/ckpts/ | head -20
}

# ─── EVAL ────────────────────────────────────────────────────────────
eval_all() {
    log_phase "probing evals"
    local total_start=$(date +%s)

    # Base grid evals
    for arch in "${ARCHS_AT_T5[@]}"; do
        for proto in "${PROTOCOLS[@]}"; do
            echo ""
            echo ">> $(date -u +%H:%M:%SZ) | eval: arch=$arch protocol=$proto T=5 (all 4 aggregations)"
            bash scripts/runpod_saebench_run_eval.sh \
                --arch "$arch" --protocol "$proto" --t 5 --aggregation all
        done
    done

    # T-sweep evals
    for t in "${T_SWEEP_VALUES[@]}"; do
        for proto in "${PROTOCOLS[@]}"; do
            echo ""
            echo ">> $(date -u +%H:%M:%SZ) | eval: arch=tempxc protocol=$proto T=$t"
            bash scripts/runpod_saebench_run_eval.sh \
                --arch tempxc --protocol "$proto" --t "$t" --aggregation all
        done
    done

    local total_end=$(date +%s)
    echo ""
    echo ">> eval phase wall time: $((total_end - total_start)) sec"
}

# ─── SUMMARY ─────────────────────────────────────────────────────────
summarize() {
    log_phase "summary"
    python - <<'PY'
import json
import glob
import os
import statistics
from collections import defaultdict

jsonl_paths = sorted(glob.glob("results/saebench/results/*.jsonl"))
if not jsonl_paths:
    print("No JSONL files found. Did eval run?")
    raise SystemExit(1)

records = []
for p in jsonl_paths:
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

print(f"Total records across {len(jsonl_paths)} JSONL files: {len(records)}")

# Headline: protocol A, aggregation full_window, k=5, averaged across tasks
headline = [r for r in records
            if r["matching_protocol"] == "A"
            and r["aggregation"] == "full_window"
            and r["k"] == 5]
by_arch_t = defaultdict(list)
for r in headline:
    by_arch_t[(r["architecture"], r["t"])].append(r["accuracy"])

print()
print("=== Headline: protocol A × full_window × k=5 ===")
print(f"{'arch':<8} {'T':<4} {'mean_acc':<10} {'n_tasks':<8}")
for (arch, t), accs in sorted(by_arch_t.items()):
    m = statistics.mean(accs)
    print(f"{arch:<8} {t:<4} {m:<10.4f} {len(accs):<8}")

# Aggregation ablation at T=5, protocol A
print()
print("=== Aggregation ablation (T=5, protocol A, k=5) ===")
by_arch_agg = defaultdict(list)
for r in records:
    if r["matching_protocol"] == "A" and r["t"] == 5 and r["k"] == 5:
        by_arch_agg[(r["architecture"], r["aggregation"])].append(r["accuracy"])
print(f"{'arch':<8} {'aggregation':<14} {'mean_acc':<10}")
for (arch, agg), accs in sorted(by_arch_agg.items()):
    m = statistics.mean(accs)
    print(f"{arch:<8} {agg:<14} {m:<10.4f}")

# T-sweep: TempXC only
print()
print("=== T-sweep (TempXC, protocol A, full_window, k=5) ===")
tsweep = [r for r in records
          if r["architecture"] == "tempxc"
          and r["matching_protocol"] == "A"
          and r["aggregation"] == "full_window"
          and r["k"] == 5]
by_t = defaultdict(list)
for r in tsweep:
    by_t[r["t"]].append(r["accuracy"])
print(f"{'T':<4} {'mean_acc':<10} {'n_tasks':<8}")
for t, accs in sorted(by_t.items()):
    m = statistics.mean(accs)
    print(f"{t:<4} {m:<10.4f} {len(accs):<8}")

print()
print(f"Full records: results/saebench/results/*.jsonl")
print(f"Raw SAEBench outputs: results/saebench/results/saebench_json/")
PY
}

# ─── MAIN DISPATCH ───────────────────────────────────────────────────
case "$PHASE" in
    cache)      run_cache ;;
    preflight)  run_preflight ;;
    train)      train_all ;;
    eval)       eval_all ;;
    summary)    summarize ;;
    full)
        run_cache
        run_preflight
        train_all
        eval_all
        summarize
        ;;
    *)
        echo "usage: $0 {full|cache|preflight|train|eval|summary}"
        exit 2
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════"
echo " orchestrator phase '$PHASE' done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "════════════════════════════════════════════════════════════"
