#!/usr/bin/env bash
# Stage B paper-budget grid orchestrator — fans cells across N GPUs.
#
# Phase 1 (cache):       single GPU, one model, one corpus pass. Cheap.
# Phases 2-5 (train,
#   mine, B1, B2):       parallelize (arch, hookpoint) cells across the
#                        N GPUs visible to the process by spawning N
#                        sub-processes with CUDA_VISIBLE_DEVICES=k each
#                        and `wait`-ing on every batch of N.
#
# Phase 6 (plots):       single CPU process at the end.
#
# Defaults to N = $(nvidia-smi -L | wc -l) so it auto-scales to whatever
# GPU count the pod provides; explicit override via NUM_GPUS env var.
#
# USAGE:
#   bash experiments/ward_backtracking_txc/run_grid_2gpu.sh
#     [NUM_GPUS=2]           # default: detected from nvidia-smi
#     [SKIP_PHASE1=1]        # default: 0 (run Phase 1 if cache missing)
#     [SKIP_TRAIN=1]         # default: 0 (skip training; use existing ckpts)
#     [ARCH_LIST="txc tsae"] # default: read from config.yaml
#
# All sub-processes inherit the env, so HF_TOKEN / ANTHROPIC_API_KEY
# from `source scripts/runpod_activate.sh` flow through.

set -euo pipefail

cd "$(dirname "$0")/../.."

ROOT="experiments.ward_backtracking_txc"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "${NUM_GPUS}" -lt 1 ]; then
    echo "[fatal] no GPUs detected (nvidia-smi returned 0). Aborting." >&2
    exit 1
fi
echo "[orchestrator] detected NUM_GPUS=${NUM_GPUS}"

# Read arch_list + hookpoint list from config.yaml (no python dep here;
# uses python -c inline so we don't need yq).
read_cfg() {
    python -c "
import yaml
cfg = yaml.safe_load(open('experiments/ward_backtracking_txc/config.yaml'))
if '$1' == 'arch_list':
    print(' '.join(cfg['txc'].get('arch_list', ['txc'])))
elif '$1' == 'hookpoints':
    print(' '.join(hp['key'] for hp in cfg['hookpoints'] if hp.get('enabled', True)))
"
}

ARCH_LIST="${ARCH_LIST:-$(read_cfg arch_list)}"
HOOKPOINTS="$(read_cfg hookpoints)"
echo "[orchestrator] arch_list='${ARCH_LIST}' hookpoints='${HOOKPOINTS}'"

# Build the (arch, hookpoint) cell list. Phase scripts already skip cells
# whose checkpoint / output already exists, so re-running is safe.
CELLS=()
for arch in $ARCH_LIST; do
    for hp in $HOOKPOINTS; do
        CELLS+=("${arch}|${hp}")
    done
done
N_CELLS=${#CELLS[@]}
echo "[orchestrator] ${N_CELLS} cells × phases will fan across ${NUM_GPUS} GPUs"

# ---- Phase 1: cache activations (single GPU, all hookpoints in one pass) ----
if [ "${SKIP_PHASE1:-0}" != "1" ]; then
    echo
    echo "=== Phase 1: cache activations on cuda:0 ==="
    CUDA_VISIBLE_DEVICES=0 python -m $ROOT.cache_activations
fi

# ---- Phase-runner helper: round-robin cells across N GPUs ----
# Usage: run_phase <phase_name> <python_module> [extra_args...]
run_phase() {
    local phase="$1"; shift
    local module="$1"; shift
    local extra=("$@")
    echo
    echo "=== Phase: ${phase} (cells × ${NUM_GPUS} GPUs in parallel) ==="
    local pids=()
    local i=0
    for cell in "${CELLS[@]}"; do
        local arch="${cell%|*}"; local hp="${cell#*|}"
        local gpu=$(( i % NUM_GPUS ))
        local logf="/tmp/${phase}_${arch}_${hp}_gpu${gpu}.log"
        echo "  [launch] ${arch}/${hp} on cuda:${gpu} → ${logf}"
        CUDA_VISIBLE_DEVICES=${gpu} python -m $module \
            --arch "${arch}" --only "${hp}" "${extra[@]}" \
            >"${logf}" 2>&1 &
        pids+=($!)
        i=$(( i + 1 ))
        # When the in-flight pool fills, wait for the whole batch and
        # report any failures before launching the next round.
        if (( ${#pids[@]} == NUM_GPUS )); then
            local failed=0
            for pid in "${pids[@]}"; do
                if ! wait "$pid"; then
                    echo "  [FAIL] pid=$pid in phase=${phase}"
                    failed=$(( failed + 1 ))
                fi
            done
            pids=()
            if (( failed > 0 )); then
                echo "  [phase ${phase}] ${failed} of ${NUM_GPUS} cells in this batch failed; "
                echo "  check the matching .log files and re-run after fixing."
                exit 1
            fi
        fi
    done
    # Drain any remaining pids in the partial batch.
    for pid in "${pids[@]}"; do
        wait "$pid" || { echo "  [FAIL] pid=$pid in phase=${phase}"; exit 1; }
    done
    echo "  [done] phase=${phase} all ${N_CELLS} cells completed"
}

# ---- Phase 2: train TXC × all archs × all hookpoints ----
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    run_phase train ${ROOT}.train_txc
fi

# ---- Phase 3: mine features ----
run_phase mine ${ROOT}.mine_features

# ---- Phase 4: B1 single-feature steering eval ----
# Sharded across GPUs via --source-shard k/N. Each shard loads the
# reasoning model on its GPU (~16 GB bf16) and evaluates its slice of
# the source list. DoM baselines are evaluated by EVERY shard so each
# shard's output JSON is self-contained; we dedupe DoM rows on merge.
echo
echo "=== Phase 4: B1 steering eval (sharded across ${NUM_GPUS} GPUs) ==="
b1_pids=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    suffix="__shard${gpu}of${NUM_GPUS}"
    logf="/tmp/b1_shard${gpu}of${NUM_GPUS}.log"
    echo "  [launch] B1 shard ${gpu}/${NUM_GPUS} on cuda:${gpu} → ${logf}"
    CUDA_VISIBLE_DEVICES=${gpu} python -m ${ROOT}.b1_steer_eval \
        --source-shard "${gpu}/${NUM_GPUS}" --out-suffix "${suffix}" \
        >"${logf}" 2>&1 &
    b1_pids+=($!)
done
for pid in "${b1_pids[@]}"; do
    wait "$pid" || { echo "  [FAIL] B1 shard pid=$pid"; exit 1; }
done

# Merge shard JSONs into the canonical results file. DoM rows appear
# in every shard; we dedupe by (source, magnitude, prompt_id).
echo "  [merge] combining B1 shards into b1_steering_results.json"
python -c "
import json
from pathlib import Path
import yaml
cfg = yaml.safe_load(open('experiments/ward_backtracking_txc/config.yaml'))
out_path = Path(cfg['paths']['steering'])
N = ${NUM_GPUS}
seen = set()
rows = []
meta = {}
for k in range(N):
    p = out_path.with_name(out_path.stem + f'__shard{k}of{N}' + out_path.suffix)
    if not p.exists(): continue
    obj = json.loads(p.read_text())
    meta = obj.get('meta', meta)
    for r in obj['rows']:
        key = (r['source'], r['magnitude'], r['prompt_id'])
        if key in seen: continue
        seen.add(key); rows.append(r)
out_path.write_text(json.dumps({'rows': rows, 'meta': meta}, indent=2))
print(f'merged {len(rows)} unique rows from {N} shards into {out_path}')
"

# ---- Phase 5: B2 cross-model ----
run_phase b2 ${ROOT}.b2_cross_model

# ---- Phase 6: plotting (CPU only) ----
echo
echo "=== Phase 6: plots (CPU) ==="
python -m ${ROOT}.plot.training_curves
python -m ${ROOT}.plot.feature_firing_heatmap
python -m ${ROOT}.plot.steering_comparison_bars
python -m ${ROOT}.plot.per_offset_firing
python -m ${ROOT}.plot.cosine_matrix
python -m ${ROOT}.plot.sentence_act_distributions
python -m ${ROOT}.plot.text_examples
python -m ${ROOT}.plot.b2_difference_area
python -m ${ROOT}.plot.coherence
python -m ${ROOT}.plot.decoder_umap || true
python -m ${ROOT}.plot.decoder_umap_x_umap || true

echo
echo "[orchestrator] done. Plots in docs/aniket/experiments/ward_backtracking/images_b/"
