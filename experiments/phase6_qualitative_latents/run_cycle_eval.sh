#!/usr/bin/env bash
# Phase 6.1 agentic-loop eval harness.
#
# Given a trained arch (ckpt at experiments/phase5_downstream_utility/
# results/ckpts/<arch>__seed42.pt), runs the full Phase 6.1 evaluation:
#   1. arch_health.py  — alive fraction + decoder cos
#   2. encode_archs.py — write z_cache/<concat>/<arch>__z.npy
#   3. run_autointerp.py — top-8 features, Claude Haiku labels
#   4. run_probing.py last_position + mean_pool — sparse-probe guard
#
# Usage:
#   bash experiments/phase6_qualitative_latents/run_cycle_eval.sh <arch>
# Example:
#   bash experiments/phase6_qualitative_latents/run_cycle_eval.sh agentic_txc_09_auxk

set -euo pipefail

ARCH="${1:?usage: run_cycle_eval.sh <arch>}"
SEED="${SEED:-42}"
CKPT="experiments/phase5_downstream_utility/results/ckpts/${ARCH}__seed${SEED}.pt"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: ckpt not found: $CKPT" >&2
  exit 1
fi

echo "=============================================================="
echo "Phase 6.1 eval for ${ARCH} (seed=${SEED})"
echo "ckpt: ${CKPT}"
echo "=============================================================="

LOGDIR="logs"
mkdir -p "$LOGDIR"

# Export Anthropic key for autointerp (if present in /workspace/)
if [ -f /workspace/.anthropic-key ]; then
  export ANTHROPIC_API_KEY="$(cat /workspace/.anthropic-key)"
fi

echo
echo "--- STEP 1: arch_health ---"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/arch_health.py \
  --archs "${ARCH}" --n-tokens 2048 \
  2>&1 | tee "${LOGDIR}/eval_${ARCH}_archhealth.log"

echo
echo "--- STEP 2: encode on concat A + B + C_v2 ---"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/encode_archs.py \
  --archs "${ARCH}" --sets A B C_v2 --seed "${SEED}" \
  2>&1 | tee "${LOGDIR}/eval_${ARCH}_encode.log"

echo
echo "--- STEP 3: autointerp (Claude Haiku + Sonnet judges) ---"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/run_autointerp.py \
  --archs "${ARCH}" --seed "${SEED}" \
  2>&1 | tee "${LOGDIR}/eval_${ARCH}_autointerp.log"

echo
echo "--- STEP 4a: sparse-probe @ last_position ---"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase5_downstream_utility/probing/run_probing.py \
  --aggregation last_position --run-ids "${ARCH}__seed${SEED}" \
  --skip-baselines \
  2>&1 | tee "${LOGDIR}/eval_${ARCH}_probe_last.log"

echo
echo "--- STEP 4b: sparse-probe @ mean_pool ---"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase5_downstream_utility/probing/run_probing.py \
  --aggregation mean_pool --run-ids "${ARCH}__seed${SEED}" \
  --skip-baselines \
  2>&1 | tee "${LOGDIR}/eval_${ARCH}_probe_mean.log"

echo
echo "=============================================================="
echo "DONE. Labels: experiments/phase6_qualitative_latents/results/autointerp/${ARCH}__labels.json"
echo "        Manual /8 semantic count required — inspect the 8 label strings."
echo "=============================================================="
