#!/usr/bin/env bash
# Belt-and-suspenders: run any T-sweep / shift variants that the main
# queue might have missed if bash buffered the script before my late edits.
#
# Idempotent: train_and_probe_both skips train if ckpt exists, but probes
# always run — if results already exist in jsonl they're effectively duplicate.

set -u
cd /workspace/temp_xc
export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc

LOG=/workspace/temp_xc/logs/overnight
PROBING=/workspace/temp_xc/experiments/phase5_downstream_utility/probing/run_probing.py
CKPT=/workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts

train() {
    local ARCH="$1"; local SEED="$2"
    local RID="${ARCH}__seed${SEED}"
    if [[ -f "${CKPT}/${RID}.pt" ]]; then
        echo "[$(date +%H:%M:%S)] SKIP train ${RID} (ckpt exists)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] TRAIN ${RID}"
    .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[${SEED}], max_steps=25000, archs=['${ARCH}'])
" > "${LOG}/train_${RID}.log" 2>&1
    local RC=$?
    echo "[$(date +%H:%M:%S)] TRAIN ${RID} exit=${RC}"
    [[ ${RC} -eq 0 && -f "${CKPT}/${RID}.pt" ]] || return 1
    return 0
}

probe() {
    local AGG="$1"; shift
    local RIDS=("$@")
    local TAG=$(echo "${RIDS[0]}" | tr '/' '_' | head -c 50)
    echo "[$(date +%H:%M:%S)] PROBE ${AGG} ${RIDS[*]}"
    .venv/bin/python -u "${PROBING}" \
        --aggregation "${AGG}" --skip-baselines \
        --run-ids "${RIDS[@]}" \
        > "${LOG}/probe_extras_${TAG}_${AGG}.log" 2>&1
    echo "[$(date +%H:%M:%S)] PROBE ${TAG} ${AGG} exit=$?"
}

train_and_probe_both() {
    local ARCH="$1"; local SEED="$2"
    train "${ARCH}" "${SEED}" || return 1
    probe last_position "${ARCH}__seed${SEED}"
    probe mean_pool     "${ARCH}__seed${SEED}"
}

wait_for_gpu_idle() {
    echo "[$(date +%H:%M:%S)] EXTRAS: waiting for in-flight python jobs..."
    while pgrep -af '\.venv/bin/python.*train_primary_archs' >/dev/null 2>&1; do
        sleep 60
    done
    while pgrep -af '\.venv/bin/python.*phase57_partB_h8_bare_multidistance' >/dev/null 2>&1; do
        sleep 60
    done
    while pgrep -af '\.venv/bin/python.*run_probing' >/dev/null 2>&1; do
        sleep 60
    done
}

wait_for_gpu_idle

# ====== H8 T-sweep granular fill-in (T=3, 4, 7, 9) ======
echo "================ EXTRAS: H8 T={3,4,7,9} fill-in ================"
for T in 3 4 7 9; do
    train_and_probe_both "phase57_partB_h8_bare_multidistance_t${T}" 42
done

# Re-probe T=8 if missing (in case main queue was still running it)
echo "================ EXTRAS: H8 T=8 (re-probe if needed) ================"
train_and_probe_both phase57_partB_h8_bare_multidistance_t8 42

# ====== Vanilla TXCDR fill-in T={4, 9} ======
echo "================ EXTRAS: vanilla TXCDR T={4, 9} ================"
for T in 4 9; do
    train_and_probe_both "txcdr_t${T}" 42
done

# ====== Long-range shift ablation (in case main queue missed) ======
echo "================ EXTRAS: long-range shifts ================"
for SPEC in 5 _10 _20 1_5 1_5_10 1_10 1_2_5_10; do
    train_and_probe_both "phase57_partB_h8a_shifts${SPEC}" 42
done

echo "================ EXTRAS DONE ================"
