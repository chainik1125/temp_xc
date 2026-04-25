#!/usr/bin/env bash
# Phase 6.1 post-training pipeline. Runs after
# `run_phase61_triangle_train.sh` has produced all ckpts.
#
# Steps (all GPU, sequential; no API calls):
#   1. Encode concat_random for ALL 9 seed=42 archs (z_cache seed doesn't
#      exist for random yet — backfills).
#   2. Encode concat_A/B/random for the new seeds
#      (Cycle F s{1,2}, 2x2 cell s{42,1,2}, tsae_paper s{1,2}).
#   3. Sparse probing on Cycle F + 2x2 cell across all seeds.
#
# Autointerp over every cell is a separate script (non-GPU, API cost).
set -euo pipefail
cd /workspace/temp_xc
source /workspace/temp_xc/.envrc

mkdir -p logs

echo "===== PHASE 6.1 POST-TRAINING: $(date -u) ====="

echo
echo "--- (1) Encode concat_random for all seed=42 archs ---"
# All Phase 6 archs at seed=42 need random-FineWeb z cache backfilled.
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/encode_archs.py \
  --archs agentic_txc_02 agentic_txc_02_batchtopk agentic_txc_09_auxk \
          agentic_txc_10_bare agentic_txc_11_stack \
          agentic_txc_12_bare_batchtopk \
          agentic_mlc_08 tsae_paper tsae_ours tfa_big \
  --sets random --seed 42 2>&1

echo
echo "--- (2) Encode A/B/random for new seeds ---"
for SEED in 1 2; do
  echo "  seed=${SEED}"
  TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
    experiments/phase6_qualitative_latents/encode_archs.py \
    --archs agentic_txc_02_batchtopk agentic_txc_12_bare_batchtopk tsae_paper \
    --sets A B random --seed "${SEED}" 2>&1
done
# 2x2 cell at seed=42 also needs its own A/B encode (ckpt is new)
echo "  seed=42 (2x2 cell)"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/encode_archs.py \
  --archs agentic_txc_12_bare_batchtopk \
  --sets A B --seed 42 2>&1

echo
echo "--- (3) Sparse probing: Cycle F + 2x2 cell, all seeds, both aggregations ---"
for AGG in last_position mean_pool; do
  for ARCH in agentic_txc_02_batchtopk agentic_txc_12_bare_batchtopk; do
    for SEED in 42 1 2; do
      CKPT="experiments/phase5_downstream_utility/results/ckpts/${ARCH}__seed${SEED}.pt"
      if [ ! -f "$CKPT" ]; then
        echo "  SKIP ${ARCH} seed=${SEED}: ckpt not found"
        continue
      fi
      echo "  probe ${ARCH} seed=${SEED} agg=${AGG}"
      TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
        experiments/phase5_downstream_utility/probing/run_probing.py \
        --aggregation "${AGG}" \
        --run-ids "${ARCH}__seed${SEED}" \
        --skip-baselines 2>&1 || echo "    PROBING FAILED: ${ARCH} seed=${SEED} ${AGG}"
    done
  done
done

echo "===== PHASE 6.1 POST-TRAINING DONE: $(date -u) ====="
