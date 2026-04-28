#!/bin/bash
# run_overnight_grid.sh — Phase-3-only paper-budget grid sweep, runs the
# real (coef × token-window) grid for each arch (sae/tempxc/mlc) at
# n=500 by default, wiping Venhoff's shipped rolling jsonl between
# arches so hybrid_token.py doesn't early-exit on its 140 author rows.
#
# Why this exists separately from the orchestrator:
#   - Phase 0/1/2 are already done; vectors live at
#     vendor/.../train-vectors/results/vars/optimized_vectors/_<arch>.pt
#   - Earlier "grid sweep" was a single-cell run (--coefficients 0.5
#     --token_windows 0); the *_grid.json files were stale shipped data
#     from rolling not being wiped. See docs/aniket/experiments/
#     venhoff_eval/summary.md banner for context.
#
# Defaults are paper App C.1 budget. Override via env.
#
# Usage on the pod, from repo root:
#   nohup bash experiments/venhoff_paper_run/run_overnight_grid.sh \
#       > logs/hybrid_per_arch/overnight.log 2>&1 &
#
# Override: ARCHES, N_TASKS, COEFFICIENTS, TOKEN_WINDOWS

set -u
cd "$(git rev-parse --show-toplevel)"

ARCHES="${ARCHES:-sae tempxc mlc}"
N_TASKS="${N_TASKS:-500}"
COEFFICIENTS="${COEFFICIENTS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
TOKEN_WINDOWS="${TOKEN_WINDOWS:--1 -15 -50 -100 0}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B}"
THINKING_MODEL="${THINKING_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
STEERING_LAYER="${STEERING_LAYER:-12}"
SAE_LAYER="${SAE_LAYER:-6}"
N_CLUSTERS="${N_CLUSTERS:-15}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2000}"
MAX_THINKING_TOKENS="${MAX_THINKING_TOKENS:-2000}"
DATASET="${DATASET:-math500}"
GPU="${GPU:-0}"

VENDOR_ROOT="${VENDOR_ROOT:-/workspace/spar-temporal-crosscoders/vendor/thinking-llms-interp}"
HYBRID_RESULTS_DIR="$VENDOR_ROOT/hybrid/results"
ROLLING_DIR="$HYBRID_RESULTS_DIR/rolling"

base_short="$(echo "$BASE_MODEL" | awk -F'/' '{print tolower($NF)}')"
mkdir -p logs/hybrid_per_arch

echo "=== overnight grid sweep ==="
echo "  arches:        $ARCHES"
echo "  n_tasks:       $N_TASKS"
echo "  coefficients:  $COEFFICIENTS"
echo "  token_windows: $TOKEN_WINDOWS"
echo "  gpu:           $GPU"
echo "  vendor_root:   $VENDOR_ROOT"
echo ""

for arch in $ARCHES; do
    log="logs/hybrid_per_arch/${arch}_overnight.log"
    out_json="$HYBRID_RESULTS_DIR/benchmark_results_${base_short}_${DATASET}_${arch}_grid.json"

    echo ">>> [$arch] wiping rolling jsonl + flat benchmark JSON"
    rm -f "$ROLLING_DIR"/rolling_${base_short}_${DATASET}*.* 2>/dev/null || true
    rm -f "$HYBRID_RESULTS_DIR/benchmark_results_${base_short}_${DATASET}.json" 2>/dev/null || true

    echo ">>> [$arch] launching hybrid grid (logs → $log)"
    CUDA_VISIBLE_DEVICES="$GPU" \
        python -m src.bench.venhoff.run_hybrid \
            --arch "$arch" \
            --venhoff-root "$VENDOR_ROOT" \
            --base-model "$BASE_MODEL" \
            --thinking-model "$THINKING_MODEL" \
            --dataset "$DATASET" \
            --steering-layer "$STEERING_LAYER" \
            --sae-layer "$SAE_LAYER" \
            --n-clusters "$N_CLUSTERS" \
            --n-tasks "$N_TASKS" \
            --coefficients $COEFFICIENTS \
            --token-windows $TOKEN_WINDOWS \
            > "$log" 2>&1 || echo "    (run_hybrid wrapper raised — likely the cosmetic FileNotFoundError on missing per-arch results dir; checking flat JSON)"

    flat_json="$HYBRID_RESULTS_DIR/benchmark_results_${base_short}_${DATASET}.json"
    if [[ -f "$flat_json" ]]; then
        echo ">>> [$arch] copying flat JSON → $out_json"
        cp "$flat_json" "$out_json"
    else
        echo "!!! [$arch] flat JSON missing at $flat_json — run probably failed, see $log"
    fi

    echo ">>> [$arch] sanity check: did the run actually use the full grid?"
    grep -E "Resume:|--coefficients|--token_windows|n_tasks=" "$log" | head -10 || true
    echo ""
done

echo "=== overnight grid sweep done ==="
echo "  per-arch JSONs:"
ls -la "$HYBRID_RESULTS_DIR"/benchmark_results_${base_short}_${DATASET}_*_grid.json 2>/dev/null || true
echo ""
echo "Next: bash experiments/venhoff_paper_run/run_analysis.sh"
