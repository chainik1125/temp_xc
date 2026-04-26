#!/usr/bin/env bash
# Verify the shifts_20 finding with 3-seed σ.
#
# Phase 5 single-seed result: phase57_partB_h8a_shifts_20 mp=0.8179
# (+0.004 over H8 3-seed 0.8126 ± 0.003). Edge-of-σ; need seeds 1, 2.

set -u
cd /workspace/temp_xc
export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc

LOG=/workspace/temp_xc/logs/overnight
PROBING=/workspace/temp_xc/experiments/phase5_downstream_utility/probing/run_probing.py
CKPT=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts

# Wait for GPU to free up — both main + extras using ~40 GB.
echo "[$(date +%H:%M:%S)] shifts_20 3-seed: waiting for GPU memory < 25 GB..."
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 25000 ]; do
    sleep 120
done
echo "[$(date +%H:%M:%S)] GPU sufficiently free"

train_and_probe_both() {
    local ARCH="$1"; local SEED="$2"
    local RID="${ARCH}__seed${SEED}"
    if [[ -f "${CKPT}/${RID}.pt" ]]; then
        echo "[$(date +%H:%M:%S)] SKIP ${RID} (ckpt exists)"
    else
        echo "[$(date +%H:%M:%S)] TRAIN ${RID}"
        .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[${SEED}], max_steps=25000, archs=['${ARCH}'])
" > "${LOG}/train_${RID}.log" 2>&1
        echo "[$(date +%H:%M:%S)] TRAIN ${RID} exit=$?"
    fi
    .venv/bin/python -u "${PROBING}" --aggregation last_position --skip-baselines \
        --run-ids "${RID}" > "${LOG}/probe_shifts20_${SEED}_lp.log" 2>&1
    .venv/bin/python -u "${PROBING}" --aggregation mean_pool --skip-baselines \
        --run-ids "${RID}" > "${LOG}/probe_shifts20_${SEED}_mp.log" 2>&1
    echo "[$(date +%H:%M:%S)] PROBE ${RID} done"
}

for SEED in 1 2; do
    train_and_probe_both phase57_partB_h8a_shifts_20 "${SEED}"
done

echo "[$(date +%H:%M:%S)] shifts_20 3-seed DONE"
