#!/bin/bash
# Han 2026-05-01: 3-seed AUC benchmark of Y/W's promising T>5 hill-climbed archs.
# Selected T=8 archs (T>5 per Han's request). The hill-climb evidence at
# T=2/3/5 was strongest for soft-pool / max-pool / contrastive-merge / spatial
# matryoshka variants; this benchmark scales them to T=8 to see if the
# advantage holds at longer windows.
#
# Architectures + seed grid:
#   1. TXCSoftMaxPool      (Galaxy 8) — Y's #1 steering arch
#   2. TXCMaxPool          (Galaxy 6) — Y's #2 steering arch
#   3. TXCContrastiveMergeH8 (W's mystery contrastive) — H8 + contrastive merge
#   4. SpatialMatryoshkaH8 (T=10 by design) — Y/W spatial matryoshka
#
# 4 archs × 3 seeds = 12 ckpts, ~6 hr training on this A40 pod.
#
# Run from repo root:
#   bash experiments/phase7_unification/case_studies/run_yw_t8_benchmark.sh

set -e

cd /workspace/temp_xc

export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export HF_HOME=/workspace/hf_cache
export UV_LINK_MODE=copy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

T=8
SEEDS=(42 1 2)

mkdir -p /workspace/temp_xc/logs

for SEED in "${SEEDS[@]}"; do
  echo "===================="
  echo "seed=${SEED}, T=${T}"
  echo "===================="

  echo ""
  echo "--- TXCSoftMaxPool (Galaxy 8) T=${T} seed=${SEED} ---"
  .venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_galaxy8 \
    --T $T --seed $SEED \
    2>&1 | tee /workspace/temp_xc/logs/yw_galaxy8_t${T}_seed${SEED}.log

  echo ""
  echo "--- TXCMaxPool (Galaxy 6) T=${T} seed=${SEED} ---"
  .venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_galaxy6 \
    --T $T --seed $SEED \
    2>&1 | tee /workspace/temp_xc/logs/yw_galaxy6_t${T}_seed${SEED}.log

  echo ""
  echo "--- TXCContrastiveMergeH8 (W's mystery) T=${T} seed=${SEED} shifts=2 ---"
  .venv/bin/python -m experiments.phase7_unification.case_studies.train_contrastive_merge_h8 \
    --T $T --shifts 2 --seed $SEED \
    2>&1 | tee /workspace/temp_xc/logs/yw_contrastive_merge_h8_t${T}_seed${SEED}.log

done

echo ""
echo "--- SpatialMatryoshkaH8 T=10 seeds=(42, 1, 2) shifts=2 uniform ---"
for SEED in "${SEEDS[@]}"; do
  echo "  seed=${SEED}"
  .venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \
    --T 10 --shifts 2 --subset-mode uniform --seed $SEED \
    2>&1 | tee /workspace/temp_xc/logs/yw_spatial_matryoshka_t10_seed${SEED}.log
done

echo ""
echo "=== ALL TRAINING DONE ==="
