#!/bin/bash
# runpod_saebench_orchestrator.sh — end-to-end driver for the full
# sparse-probing experiment grid.
#
# Trains the 6 base checkpoints (3 archs × 2 protocols at T=5) plus
# the T-sweep TempXC checkpoints (T ∈ {10, 20, 40} × 2 protocols),
# then runs SAEBench sparse_probing on each with all 4 aggregations.
#
# Resumable: every step skips if output already exists. Run this
# multiple times; each invocation only runs the missing pieces.
#
# Split into two phases because of the SAEBench deps-isolation:
#   Phase 1 (main venv): train all checkpoints.
#   Phase 2 (saebench venv): run probing evals.
#
# Usage:
#   bash scripts/runpod_saebench_orchestrator.sh train     # phase 1
#   bash scripts/runpod_saebench_orchestrator.sh eval      # phase 2
#   bash scripts/runpod_saebench_orchestrator.sh both      # both (default)
#
# Env overrides:
#   SKIP_MLC=1           skip MLC runs (MLC ArchSpec pending)
#   SKIP_TSWEEP=1        skip T-sweep (only T=5)
#   MAX_T=40             cap T-sweep at this value
#
# Before launching, profile VRAM at the intended T_max:
#   bash scripts/runpod_saebench_profile_vram.sh --t 40

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

PHASE="${1:-both}"
SKIP_MLC="${SKIP_MLC:-1}"        # default: skip MLC until its ArchSpec lands
SKIP_TSWEEP="${SKIP_TSWEEP:-0}"
MAX_T="${MAX_T:-40}"

BASE_ARCHS=(sae tempxc)
[ "$SKIP_MLC" != "1" ] && BASE_ARCHS+=(mlc)
PROTOCOLS=(A B)
BASE_T=5
TSWEEP_T_VALUES=(10 20 40)

echo "=== saebench orchestrator ==="
echo "  phase:       $PHASE"
echo "  archs:       ${BASE_ARCHS[*]}"
echo "  protocols:   ${PROTOCOLS[*]}"
echo "  base T:      $BASE_T"
[ "$SKIP_TSWEEP" = "1" ] && echo "  T-sweep:     SKIPPED" \
                         || echo "  T-sweep:     TempXC at T ∈ {${TSWEEP_T_VALUES[*]}}  (cap=$MAX_T)"
echo ""

train_one() {
    local arch="$1" protocol="$2" t="$3"
    echo ""
    echo ">> train: arch=$arch protocol=$protocol T=$t"
    bash scripts/runpod_saebench_train.sh \
        --arch "$arch" --protocol "$protocol" --t "$t"
}

eval_one() {
    local arch="$1" protocol="$2" t="$3"
    echo ""
    echo ">> eval: arch=$arch protocol=$protocol T=$t (all 4 aggregations)"
    bash scripts/runpod_saebench_run_eval.sh \
        --arch "$arch" --protocol "$protocol" --t "$t" \
        --aggregation all
}

# ─── Phase 1: training ──────────────────────────────────────────────
if [ "$PHASE" = "train" ] || [ "$PHASE" = "both" ]; then
    echo "=== phase 1: training ==="

    # base grid: 3 archs × 2 protocols at T=5
    for arch in "${BASE_ARCHS[@]}"; do
        for proto in "${PROTOCOLS[@]}"; do
            train_one "$arch" "$proto" "$BASE_T"
        done
    done

    # T-sweep: TempXC only
    if [ "$SKIP_TSWEEP" != "1" ]; then
        for t in "${TSWEEP_T_VALUES[@]}"; do
            if [ "$t" -gt "$MAX_T" ]; then
                echo ">> SKIP T=$t (> MAX_T=$MAX_T)"
                continue
            fi
            for proto in "${PROTOCOLS[@]}"; do
                train_one "tempxc" "$proto" "$t"
            done
        done
    fi

    echo ""
    echo "=== training phase done ==="
    ls -lh results/saebench/ckpts/ 2>/dev/null | tail -20
fi

# ─── Phase 2: probing evals ─────────────────────────────────────────
if [ "$PHASE" = "eval" ] || [ "$PHASE" = "both" ]; then
    echo ""
    echo "=== phase 2: probing evals ==="
    echo "  NOTE: SAEBench must be installed in the current env."
    echo "        If deps conflict, activate the saebench sidecar env first."
    echo ""

    for arch in "${BASE_ARCHS[@]}"; do
        for proto in "${PROTOCOLS[@]}"; do
            eval_one "$arch" "$proto" "$BASE_T"
        done
    done

    if [ "$SKIP_TSWEEP" != "1" ]; then
        for t in "${TSWEEP_T_VALUES[@]}"; do
            if [ "$t" -gt "$MAX_T" ]; then
                continue
            fi
            for proto in "${PROTOCOLS[@]}"; do
                eval_one "tempxc" "$proto" "$t"
            done
        done
    fi

    echo ""
    echo "=== eval phase done ==="
    echo ""
    echo "Aggregate JSONL files:"
    ls -lh results/saebench/results/*.jsonl 2>/dev/null | sed 's|^|  |'
    echo ""
    echo "Total records across all runs:"
    cat results/saebench/results/*.jsonl 2>/dev/null | wc -l
fi
