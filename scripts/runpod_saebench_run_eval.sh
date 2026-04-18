#!/bin/bash
# runpod_saebench_run_eval.sh — run SAEBench sparse_probing on one
# trained checkpoint across all 4 aggregation strategies. One JSONL
# record per (task, k) emitted to results/saebench/results/<run_id>.jsonl.
#
# SAEBench has strict pinned deps (numpy<2, datasets<4, sae_lens^6.22)
# that conflict with our main env. This script assumes SAEBench is
# installed in a sidecar env; activate it before invoking, e.g.:
#
#   source /workspace/saebench_env/bin/activate
#   bash scripts/runpod_saebench_run_eval.sh --arch tempxc --protocol A --t 5
#
# See docs/aniket/experiments/sparse_probing/saebench_notes § 7.
#
# Usage:
#   bash scripts/runpod_saebench_run_eval.sh --arch sae --protocol A
#   bash scripts/runpod_saebench_run_eval.sh --arch tempxc --protocol B --t 10
#
# With --aggregation, run only that one strategy (default: all four).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

ARCH=""
PROTOCOL=""
T=5
SEED=42
AGGREGATION="all"  # all | last | mean | max | full_window

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)        ARCH="$2";        shift 2;;
        --protocol)    PROTOCOL="$2";    shift 2;;
        --t)           T="$2";           shift 2;;
        --seed)        SEED="$2";        shift 2;;
        --aggregation) AGGREGATION="$2"; shift 2;;
        *) echo "unknown flag: $1"; exit 2;;
    esac
done

for req in ARCH PROTOCOL; do
    if [ -z "${!req}" ]; then
        echo "FAIL: --${req,,} is required"
        exit 2
    fi
done

# Resolve the checkpoint path + output JSONL location
eval "$(python - <<PY
from src.bench.saebench.configs import ckpt_name, CKPT_DIR, RESULTS_DIR
name = ckpt_name("$ARCH", "$PROTOCOL", t=$T, seed=$SEED)
print(f'CKPT_PATH="{CKPT_DIR}/{name}"')
print(f'RESULTS_DIR="{RESULTS_DIR}"')
PY
)"

if [ ! -f "$CKPT_PATH" ]; then
    echo "FAIL: checkpoint not found: $CKPT_PATH"
    echo "  did you run runpod_saebench_train.sh for this config first?"
    exit 3
fi

OUT_JSONL="${RESULTS_DIR}/${ARCH}_prot${PROTOCOL}_T${T}.jsonl"
mkdir -p "$RESULTS_DIR"

case "$AGGREGATION" in
    all) AGGREGATIONS=(last mean max full_window);;
    last|mean|max|full_window) AGGREGATIONS=("$AGGREGATION");;
    *) echo "FAIL: --aggregation must be all, last, mean, max, or full_window"; exit 2;;
esac

echo "=== saebench sparse_probing eval ==="
echo "  arch:           $ARCH"
echo "  protocol:       $PROTOCOL"
echo "  T:              $T"
echo "  ckpt:           $CKPT_PATH"
echo "  aggregations:   ${AGGREGATIONS[*]}"
echo "  output JSONL:   $OUT_JSONL"
echo ""

for agg in "${AGGREGATIONS[@]}"; do
    echo ""
    echo ">> running probing eval for aggregation=$agg (ordered + shuffled pair)"
    # Uses the new run_eval.py CLI which invokes run_probing twice:
    # once ordered, once shuffled (seed=42). Per-example predictions are
    # persisted via the shared probe_fit utility (see item 8). Regression
    # gate runs at startup; --skip-regressions bypasses it if needed.
    python -m src.bench.run_eval \
        --architecture "$ARCH" \
        --protocol "$PROTOCOL" \
        --t "$T" \
        --aggregation "$agg" \
        --ckpt "$CKPT_PATH" \
        --output "$OUT_JSONL" \
        --seed "$SEED" \
        --device "cuda:0" \
        --both-ordered-shuffled
done

echo ""
echo "=== done ==="
echo "  total records in $OUT_JSONL: $(wc -l < "$OUT_JSONL")"
