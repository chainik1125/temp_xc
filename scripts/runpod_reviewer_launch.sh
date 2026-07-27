#!/usr/bin/env bash
# Launch one frozen reviewer-multiseed lane on a persistent RunPod volume.
#
# Usage:
#   bash scripts/runpod_reviewer_launch.sh c7
#   bash scripts/runpod_reviewer_launch.sh em
#   bash scripts/runpod_reviewer_launch.sh rlhf
#
# The caller should run this under nohup/tmux. Each lane is idempotent and
# writes durable outputs below /workspace/reviewer_multiseed.

set -euo pipefail

LANE="${1:?usage: runpod_reviewer_launch.sh c7|em|rlhf}"
DRIVER_BRANCH="dmitry-btk-txc-sprint"
DRIVER_PIN="6d4f26a1"
REPO_URL="https://github.com/chainik1125/temp_xc.git"
DRIVER_ROOT="/workspace/reviewer-driver"
OUTPUT_ROOT="/workspace/reviewer_multiseed"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${TMPDIR}" "${OUTPUT_ROOT}" /workspace/logs

if [ ! -d "${DRIVER_ROOT}/.git" ]; then
    git clone --branch "${DRIVER_BRANCH}" "${REPO_URL}" "${DRIVER_ROOT}"
fi
git -C "${DRIVER_ROOT}" fetch origin "${DRIVER_BRANCH}"
git -C "${DRIVER_ROOT}" checkout "${DRIVER_BRANCH}"
git -C "${DRIVER_ROOT}" pull --ff-only origin "${DRIVER_BRANCH}"
git -C "${DRIVER_ROOT}" merge-base --is-ancestor "${DRIVER_PIN}" HEAD

if ! command -v uv >/dev/null 2>&1; then
    python -m pip install uv
fi

ensure_commit() {
    if ! git -C "${DRIVER_ROOT}" cat-file -e "${1}^{commit}" 2>/dev/null; then
        git -C "${DRIVER_ROOT}" fetch origin "${1}"
    fi
}

case "${LANE}" in
    c7)
        PAPER_ROOT="/workspace/c7-paper"
        PAPER_PIN="b8ab4b95dc8d5a7b6da28bdcb71acfaa9c42aff5"
        ensure_commit "${PAPER_PIN}"
        if [ ! -e "${PAPER_ROOT}/.git" ]; then
            git -C "${DRIVER_ROOT}" worktree add --detach "${PAPER_ROOT}" "${PAPER_PIN}"
        fi
        test "$(git -C "${PAPER_ROOT}" rev-parse HEAD)" = "${PAPER_PIN}"
        (
            cd "${PAPER_ROOT}/purified"
            uv sync --frozen
            export PYTHONPATH="${PAPER_ROOT}/purified/src:${PAPER_ROOT}/purified"
            .venv/bin/python \
                "${DRIVER_ROOT}/scripts/runpod_c7_paper_multiseed.py" \
                --paper-root "${PAPER_ROOT}" \
                --output-root "${OUTPUT_ROOT}/c7"
        )
        ;;
    em)
        PAPER_ROOT="/workspace/em-paper"
        PAPER_PIN="0d208fa6a11ddf775c09ccd4f89f52e6c8eea515"
        ensure_commit "${PAPER_PIN}"
        if [ ! -e "${PAPER_ROOT}/.git" ]; then
            git -C "${DRIVER_ROOT}" worktree add --detach "${PAPER_ROOT}" "${PAPER_PIN}"
        fi
        test "$(git -C "${PAPER_ROOT}" rev-parse HEAD)" = "${PAPER_PIN}"
        (
            cd "${PAPER_ROOT}"
            uv sync --frozen
            uv pip install peft
            export PYTHONPATH="${PAPER_ROOT}/src:${PAPER_ROOT}"
            export TEMP_BENCH_ALLOW_DIRTY=1
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            export TEMP_BENCH_EM_CELL_OUTPUT="${OUTPUT_ROOT}/em/txc_base_T5_seed2.json"

            # Caches are streamed from the completed Modal volume by the
            # controller. Wait for the two large files before validating them.
            for _ in $(seq 1 360); do
                if [ -s /workspace/em_data_cache/56a61e3776062439/acts.npy ] \
                    && [ -s /workspace/conv_depth_caches/em_medical/hs16.npy ]; then
                    break
                fi
                sleep 30
            done
            test -s /workspace/em_data_cache/56a61e3776062439/acts.npy
            test -s /workspace/conv_depth_caches/em_medical/hs16.npy

            .venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np

train = Path("/workspace/em_data_cache/56a61e3776062439")
cohort = Path("/workspace/conv_depth_caches/em_medical")
acts = np.load(train / "acts.npy", mmap_mode="r")
token_ids = np.load(train / "token_ids.npy", mmap_mode="r")
hs16 = np.load(cohort / "hs16.npy", mmap_mode="r")
labels = np.load(cohort / "labels.npy", mmap_mode="r")
assert acts.shape[:2] == token_ids.shape, (acts.shape, token_ids.shape)
assert hs16.shape[0] == labels.shape[0], (hs16.shape, labels.shape)
print(json.dumps({
    "train_acts_shape": list(acts.shape),
    "token_ids_shape": list(token_ids.shape),
    "cohort_hs16_shape": list(hs16.shape),
    "labels_shape": list(labels.shape),
}, indent=2), flush=True)
PY

            mkdir -p /workspace/em_checkpoints "${PAPER_ROOT}/results"
            cp -rn "${PAPER_ROOT}/checkpoints/." /workspace/em_checkpoints/ \
                2>/dev/null || true
            rm -rf "${PAPER_ROOT}/results/data_cache" "${PAPER_ROOT}/checkpoints"
            ln -s /workspace/em_data_cache "${PAPER_ROOT}/results/data_cache"
            ln -s /workspace/em_checkpoints "${PAPER_ROOT}/checkpoints"

            .venv/bin/python \
                "${DRIVER_ROOT}/experiments/explorations/btk_rerun/em_cell.py" \
                txc_base 5 2
        )
        ;;
    rlhf)
        PAPER_ROOT="/workspace/rlhf-paper"
        ensure_commit ed9a6c77
        PAPER_PIN="$(git -C "${DRIVER_ROOT}" rev-parse ed9a6c77)"
        if [ ! -e "${PAPER_ROOT}/.git" ]; then
            git -C "${DRIVER_ROOT}" worktree add --detach "${PAPER_ROOT}" "${PAPER_PIN}"
        fi
        test "$(git -C "${PAPER_ROOT}" rev-parse HEAD)" = "${PAPER_PIN}"
        (
            cd "${PAPER_ROOT}"
            uv sync --frozen
            export PYTHONPATH="${PAPER_ROOT}/src:${PAPER_ROOT}"

            # Do not compete with the pre-existing stacked-arm job on this pod.
            # Its wrapper writes this marker after releasing the GPU.
            for _ in $(seq 1 1440); do
                if grep -q "RLHF_ALL_DONE" /workspace/logs/fix2.log 2>/dev/null; then
                    break
                fi
                sleep 30
            done
            grep -q "RLHF_ALL_DONE" /workspace/logs/fix2.log

            .venv/bin/python \
                "${DRIVER_ROOT}/scripts/runpod_rlhf_papermatch_multiseed.py" \
                --paper-root "${PAPER_ROOT}" \
                --output-root "${OUTPUT_ROOT}/rlhf"
        )
        ;;
    *)
        echo "unknown lane: ${LANE}" >&2
        exit 2
        ;;
esac

echo "REVIEWER_${LANE^^}_DONE"
