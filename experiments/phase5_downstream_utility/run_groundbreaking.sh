#!/usr/bin/env bash
# Sequential launcher for the post-H8 groundbreaking-results queue.
#
# Runs jobs in dependency order. Each job logs to logs/overnight/.
# DO NOT launch this in parallel with other GPU jobs unless you've verified
# headroom — most archs use ~14GB at T=5 and proportionally more at larger T.

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
        > "${LOG}/probe_${TAG}_${AGG}.log" 2>&1
    echo "[$(date +%H:%M:%S)] PROBE ${TAG} ${AGG} exit=$?"
}

train_and_probe_both() {
    local ARCH="$1"; local SEED="$2"
    train "${ARCH}" "${SEED}" || { echo "TRAIN FAILED ${ARCH}__seed${SEED}"; return 1; }
    probe last_position "${ARCH}__seed${SEED}"
    probe mean_pool     "${ARCH}__seed${SEED}"
}

# Wait for ANY in-flight training process from the previous session/wrapper
# to finish before we start the sequential queue. This avoids GPU OOM.
wait_for_gpu_idle() {
    # Match only python training processes (not tail -F monitors that
    # happen to have these strings in their tailed-file paths).
    echo "[$(date +%H:%M:%S)] waiting for in-flight python training jobs to finish..."
    while pgrep -af '\.venv/bin/python.*train_primary_archs' >/dev/null 2>&1; do
        sleep 60
    done
    while pgrep -af '\.venv/bin/python.*phase57_partB_h8_bare_multidistance' >/dev/null 2>&1; do
        sleep 60
    done
    echo "[$(date +%H:%M:%S)] GPU should be idle now. Free MiB:"
    nvidia-smi --query-gpu=memory.free --format=csv,noheader
}

# === P0a: H8 T-sweep ===
# T=10 may already be training in background; this script waits for it.
echo "================ P0a H8 T-sweep ================"
wait_for_gpu_idle
train_and_probe_both phase57_partB_h8_bare_multidistance_t10 42
# Granular fill-in between T=5 (existing) and T=10 — cheap, paper-relevant.
train_and_probe_both phase57_partB_h8_bare_multidistance_t6 42
train_and_probe_both phase57_partB_h8_bare_multidistance_t7 42
train_and_probe_both phase57_partB_h8_bare_multidistance_t8 42
train_and_probe_both phase57_partB_h8_bare_multidistance_t15 42
train_and_probe_both phase57_partB_h8_bare_multidistance_t20 42
train_and_probe_both phase57_partB_h8_bare_multidistance_t30 42

# === P0b: shift-ablation at T=5 ===
# Now extended with LONG-RANGE shifts to chase the user's hypothesis:
#   "if shift=2 helps, why not 3, 4, or even more?". Adds variants
#   testing shifts of 5, 10, 20 — well beyond T=5's natural window.
echo "================ P0b shift ablation ================"
for SPEC in 1 123 1234 124 2 4 123_uniform; do
    train_and_probe_both "phase57_partB_h8a_shifts${SPEC}" 42
done
# Long-range shift ablation — user's "if shift=2 helps, why not 3, 4, or
# even more?" pursuit. Tests whether bigger shifts top H8's mp 0.8126.
# Naming convention: single-digit shifts use bare digits (shifts5);
# multi-digit / multi-shift specs use underscore-separated (shifts_10,
# shifts1_5_10). Dispatcher parses underscore-separated as integer list
# and falls back to per-char for single-digit-only specs.
# Token overlap at T=5: shift=5 → 0%, shift=10 → 0%, shift=20 → 0%.
for SPEC in 5 _10 _20 1_5 1_5_10 1_10 1_2_5_10; do
    train_and_probe_both "phase57_partB_h8a_shifts${SPEC}" 42
done

# === P0d: H13 multi-distance × multi-scale (orthogonal-axes stack) ===
# At T=5, shifts=(1,2), n_contr_scales=3, γ=0.5. Six InfoNCE terms.
echo "================ P0d H13 mdms stack ================"
train_and_probe_both phase57_partB_h13_md_x_ms 42

# === P0c: MLC + anti-dead fairness counterparts ===
# Apply the anti-dead stack to MLC so paper reviewers can't claim TXC got an
# unfair anti-dead advantage. Three variants matching TXC anti-dead family:
#   mlc_bare_antidead — recon-only (TXCBareAntidead counterpart)
#   mlc_bare_matryoshka_contrastive_antidead — single-shift adjacent-token
#     InfoNCE (Phase 6.2 C3 counterpart)
#   mlc_bare_multiscale_antidead — H7 counterpart (multi-scale InfoNCE on
#     adjacent-token pairs)
# Multi-distance has no MLC analog (no temporal shift axis).
echo "================ P0c MLC + anti-dead ================"
for ARCH in mlc_bare_antidead mlc_bare_matryoshka_contrastive_antidead mlc_bare_multiscale_antidead; do
    train_and_probe_both "${ARCH}" 42
done

# === P2: H9c contrastive seeds 1, 2 ===
echo "================ P2 H9c seeds 1, 2 ================"
for SEED in 1 2; do
    train_and_probe_both feature_nested_matryoshka_t5_contrastive ${SEED}
done

# === P3: H3 log-matryoshka T-sweep ===
echo "================ P3 H3 log-matryoshka T-sweep ================"
for T in 5 10 15 20 30; do
    train_and_probe_both "log_matryoshka_t${T}" 42
done

# === P4: alive-fraction retry (T=10/15/20 vanilla TXCDR + new H8 archs) ===
echo "================ P4 alive-fraction ================"
.venv/bin/python -u experiments/phase5_downstream_utility/analysis/alive_fraction.py \
    > "${LOG}/alive_fraction_retry.log" 2>&1
echo "[$(date +%H:%M:%S)] alive-fraction exit=$?"

# === P5: HF sync ===
echo "================ P5 HF sync ================"
if [[ -f scripts/hf_upload_ckpts.py ]]; then
    .venv/bin/python -u scripts/hf_upload_ckpts.py \
        > "${LOG}/hf_sync_final.log" 2>&1
    echo "[$(date +%H:%M:%S)] hf-sync exit=$?"
else
    echo "scripts/hf_upload_ckpts.py NOT FOUND — skipping"
fi

echo "================ ALL DONE ================"
