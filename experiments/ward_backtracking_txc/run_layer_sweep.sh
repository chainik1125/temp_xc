#!/usr/bin/env bash
# Layer sweep: TXC k=16 at residual layers {5, 15, 20, 25} (L10 already
# done as the main run's winner). Tests Dmitry's question — does the
# TXC-vs-SAE-family lead hold across layers, or is it L10-specific?
#
# Pipeline per layer:
#   1. Cache base Llama-3.1-8B activations at resid_L<N> (~6-7 min on H100,
#      writes results/.../activations/resid_L<N>.npy)
#   2. Train txc__resid_L<N>__k16__s42 (~10 min)
#   3. Mine + B1 + Sonnet grade + metric (evaluate_cell, ~15 min with the
#      parallelized B1 from commit 95a27989 and inline grading from #15)
#
# 2-GPU schedule: pair layers across the two GPUs so caching + train + B1
# overlap. GPU 0 runs L5 then L15. GPU 1 runs L20 then L25. Caching for the
# next-layer is sequential on its own GPU because a single forward pass
# saturates the GPU; training overlaps with mining/B1 of the prior cell
# only if we kick them off in separate processes (we don't here — single
# evaluate_cell process per layer).

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="experiments.ward_backtracking_txc"

LAYERS_GPU0=(5 15)
LAYERS_GPU1=(20 25)
LOG_GPU0=/tmp/layer_sweep_gpu0.log
LOG_GPU1=/tmp/layer_sweep_gpu1.log
: > "$LOG_GPU0"
: > "$LOG_GPU1"

run_layer() {
    local layer=$1 gpu=$2 logfile=$3
    local cell="txc__resid_L${layer}__k16__s42"
    echo "=== [$cell] cache + train + B1 on cuda:$gpu ===" >> "$logfile"

    # Step 1: cache (skip if file exists)
    local cache_file="results/ward_backtracking_txc/activations/resid_L${layer}.npy"
    if [ ! -f "$cache_file" ]; then
        CUDA_VISIBLE_DEVICES="$gpu" python -m $ROOT.cache_activations \
            --override-hookpoint "resid_L${layer}:${layer}:resid" >> "$logfile" 2>&1 \
            || { echo "[FAIL] cache resid_L${layer}" >> "$logfile"; return 1; }
    else
        echo "[skip-cache] $cache_file exists" >> "$logfile"
    fi

    # Step 2: full pipeline (train + mine + B1 + inline grade + metric)
    CUDA_VISIBLE_DEVICES="$gpu" python -m $ROOT.evaluate_cell --cell "$cell" >> "$logfile" 2>&1 \
        || { echo "[FAIL] evaluate $cell" >> "$logfile"; return 1; }
    echo "[$cell] DONE" >> "$logfile"
}

(
    for L in "${LAYERS_GPU0[@]}"; do
        run_layer "$L" 0 "$LOG_GPU0" || break
    done
) &
PID0=$!

(
    for L in "${LAYERS_GPU1[@]}"; do
        run_layer "$L" 1 "$LOG_GPU1" || break
    done
) &
PID1=$!

echo "[layer-sweep] GPU 0 driver PID=$PID0 (L5, L15 → $LOG_GPU0)"
echo "[layer-sweep] GPU 1 driver PID=$PID1 (L20, L25 → $LOG_GPU1)"
wait $PID0 $PID1
echo "[layer-sweep] all 4 layers complete."

# Refresh global metrics (already inline-graded per cell, but this also
# refreshes bootstrap CIs which depend on the full row set).
python -m $ROOT.regrade_cells --concurrency 12

echo "[layer-sweep] DONE."
