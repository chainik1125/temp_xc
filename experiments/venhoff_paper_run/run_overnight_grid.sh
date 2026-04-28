#!/bin/bash
# run_overnight_grid.sh — Phase-3-only paper-budget grid sweep.
# Runs the real (coef × token-window) grid for each arch
# (sae/tempxc/mlc) at n=500, 10 coefs × 5 windows by default.
#
# Why this script exists:
#   - Phase 0/1/2 are already done; Phase 2 vectors live at
#     vendor/.../train-vectors/results/vars/optimized_vectors/
#     (SAE: bare {model}_{tag}.pt = Venhoff's shipped;
#      TempXC: {model}_{tag}_tempxc.pt = our trained;
#      MLC: {model}_{tag}_mlc.pt = our trained).
#   - The earlier "grid sweep" (--coefficients 0.5 --token_windows 0)
#     was a single-cell run, and the *_grid.json files were stale
#     shipped data because rolling jsonl wasn't wiped per arch.
#
# Parallelism:
#   - 3 archs run in parallel, one per GPU (CUDA_VISIBLE_DEVICES=0/1/2).
#   - Each arch gets its own hardlinked copy of the vendor tree
#     (`cp -al`) so their rolling jsonl + benchmark JSON paths don't
#     collide. All read-only files (models, vectors, code) are
#     hardlinks back to the canonical vendor dir — so disk cost is
#     negligible and steering vectors resolve to the same on-disk
#     bytes for all three runs.
#   - 4th GPU is intentionally idle; sharding within a single arch
#     (HF auto-shard or task-slicing) requires vendor patches we
#     haven't validated, so left for a later pass.
#
# OOM risk on Phase 3: ~zero. base+thinking ≈ 16GB+16GB = 32GB on a
# single H100 80GB, no big optimizer states. We export
# expandable_segments anyway as belt-and-suspenders.
#
# Usage on the pod, from repo root:
#   nohup bash experiments/venhoff_paper_run/run_overnight_grid.sh \
#       > logs/hybrid_per_arch/overnight.log 2>&1 &
#
# Override (env): ARCHES, N_TASKS, COEFFICIENTS, TOKEN_WINDOWS,
#                 GPUS (space-separated, len must equal len(ARCHES))

set -u
cd "$(git rev-parse --show-toplevel)"

ARCHES="${ARCHES:-sae tempxc mlc}"
GPUS="${GPUS:-0 1 2}"
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

VENDOR_REAL="${VENDOR_REAL:-/workspace/spar-temporal-crosscoders/vendor/thinking-llms-interp}"
PER_ARCH_ROOT="${PER_ARCH_ROOT:-/workspace/vendor_runs}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

base_short="$(echo "$BASE_MODEL" | awk -F'/' '{print tolower($NF)}')"
mkdir -p logs/hybrid_per_arch

# Validate len(GPUS) == len(ARCHES).
read -ra ARCH_ARR <<< "$ARCHES"
read -ra GPU_ARR <<< "$GPUS"
if [[ ${#ARCH_ARR[@]} -ne ${#GPU_ARR[@]} ]]; then
    echo "FAIL: ARCHES has ${#ARCH_ARR[@]} entries but GPUS has ${#GPU_ARR[@]}; they must match"
    exit 2
fi

echo "=== overnight grid sweep (parallel) ==="
echo "  arches:        $ARCHES"
echo "  gpus:          $GPUS"
echo "  n_tasks:       $N_TASKS"
echo "  coefficients:  $COEFFICIENTS  (n=$(echo $COEFFICIENTS | wc -w))"
echo "  token_windows: $TOKEN_WINDOWS  (n=$(echo $TOKEN_WINDOWS | wc -w))"
echo "  vendor_real:   $VENDOR_REAL"
echo "  per_arch_root: $PER_ARCH_ROOT"
echo "  alloc_conf:    $PYTORCH_CUDA_ALLOC_CONF"
echo ""

# ── Step 1: per-arch hardlinked vendor copies.
echo ">>> staging per-arch vendor copies"
mkdir -p "$PER_ARCH_ROOT"
for arch in "${ARCH_ARR[@]}"; do
    target="$PER_ARCH_ROOT/${arch}"
    if [[ -d "$target" ]]; then
        echo "    [info] removing stale $target"
        rm -rf "$target"
    fi
    echo "    [info] cp -al  $VENDOR_REAL  →  $target"
    cp -al "$VENDOR_REAL" "$target"
    # Replace hybrid/results with a private writable copy (not hardlinks)
    # so rolling jsonl + benchmark JSON writes don't clobber the canonical
    # tree or sibling archs.
    rm -rf "$target/hybrid/results"
    cp -r "$VENDOR_REAL/hybrid/results" "$target/hybrid/results"
    # Wipe rolling + any flat benchmark JSON leftover from earlier runs so
    # hybrid_token.py doesn't early-exit on shipped author results.
    rm -f "$target"/hybrid/results/rolling/rolling_${base_short}_${DATASET}*.* 2>/dev/null || true
    rm -f "$target"/hybrid/results/benchmark_results_${base_short}_${DATASET}*.json 2>/dev/null || true
done
echo ""

# ── Step 2: parallel launch.
echo ">>> launching $((${#ARCH_ARR[@]})) arches in parallel"
pids=()
for i in "${!ARCH_ARR[@]}"; do
    arch="${ARCH_ARR[$i]}"
    gpu="${GPU_ARR[$i]}"
    target="$PER_ARCH_ROOT/${arch}"
    log="logs/hybrid_per_arch/${arch}_overnight.log"
    echo "    [info] arch=$arch gpu=$gpu venhoff_root=$target log=$log"

    CUDA_VISIBLE_DEVICES="$gpu" \
        python -m src.bench.venhoff.run_hybrid \
            --arch "$arch" \
            --venhoff-root "$target" \
            --base-model "$BASE_MODEL" \
            --thinking-model "$THINKING_MODEL" \
            --dataset "$DATASET" \
            --steering-layer "$STEERING_LAYER" \
            --sae-layer "$SAE_LAYER" \
            --n-clusters "$N_CLUSTERS" \
            --n-tasks "$N_TASKS" \
            --coefficients $COEFFICIENTS \
            --token-windows $TOKEN_WINDOWS \
            > "$log" 2>&1 &
    pids+=($!)
done
echo "    pids: ${pids[*]}"
echo ""

# ── Step 3: wait per-pid so a single failure doesn't silently eat
# the others' output.
echo ">>> waiting on archs"
any_failed=0
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    arch="${ARCH_ARR[$i]}"
    if wait "$pid"; then
        echo "    [done] arch=$arch status=ok"
    else
        rc=$?
        echo "    [fail] arch=$arch rc=$rc log=logs/hybrid_per_arch/${arch}_overnight.log"
        any_failed=1
    fi
done
echo ""

# ── Step 4: collect benchmark JSONs into the canonical vendor dir
# under arch-suffixed names so analyze_grid.py can find them.
echo ">>> collecting per-arch benchmark JSONs"
for arch in "${ARCH_ARR[@]}"; do
    target="$PER_ARCH_ROOT/${arch}"
    src_json="$target/hybrid/results/benchmark_results_${base_short}_${DATASET}.json"
    out_json="$VENDOR_REAL/hybrid/results/benchmark_results_${base_short}_${DATASET}_${arch}_grid.json"
    if [[ -f "$src_json" ]]; then
        cp "$src_json" "$out_json"
        echo "    [ok] $arch → $out_json"
    else
        echo "    [warn] $arch missing flat JSON at $src_json — see log"
        any_failed=1
    fi
done
echo ""

# ── Step 5: sanity-check each arch's run actually used the full grid.
echo ">>> sanity check — full grid used? (look for matching --coefficients/--token_windows + no Resume early-exit)"
for arch in "${ARCH_ARR[@]}"; do
    log="logs/hybrid_per_arch/${arch}_overnight.log"
    echo "  --- $arch ---"
    grep -E "Resume:|--coefficients|--token_windows|n_tasks=|cmd=" "$log" 2>/dev/null | head -8 || true
done
echo ""

if [[ $any_failed -eq 1 ]]; then
    echo "=== overnight grid sweep finished with failures — inspect logs above ==="
    exit 1
fi

echo "=== overnight grid sweep done ==="
ls -la "$VENDOR_REAL"/hybrid/results/benchmark_results_${base_short}_${DATASET}_*_grid.json 2>/dev/null || true
echo ""
echo "Next: bash experiments/venhoff_paper_run/run_analysis.sh"
