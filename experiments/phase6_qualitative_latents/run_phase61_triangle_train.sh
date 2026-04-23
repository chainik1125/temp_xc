#!/usr/bin/env bash
# Phase 6.1 triangle seed-variance + 2x2 cell training.
#
# Runs in sequence on a single A40:
#   1. Cycle F (agentic_txc_02_batchtopk) seeds {1, 2}       — 2 × ~30 min
#   2. 2x2 cell (agentic_txc_12_bare_batchtopk) seeds {42,1,2} — 3 × ~30 min
#   3. tsae_paper seeds {1, 2}                                — 2 × ~30 min
#
# Total ~3.5 hr.
set -euo pipefail
cd /workspace/temp_xc
source /workspace/temp_xc/.envrc

mkdir -p logs

echo "===== PHASE 6.1 TRAINING: $(date -u) ====="

echo
echo "--- (1) Cycle F seeds {1, 2} ---"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase5_downstream_utility/train_primary_archs.py \
  --archs agentic_txc_02_batchtopk \
  --seeds 1 2 \
  --max-steps 25000 2>&1

echo
echo "--- (2) 2x2 cell seeds {42, 1, 2} ---"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase5_downstream_utility/train_primary_archs.py \
  --archs agentic_txc_12_bare_batchtopk \
  --seeds 42 1 2 \
  --max-steps 25000 2>&1

echo
echo "--- (3) tsae_paper seed 1 ---"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/train_tsae_paper.py --seed 1 2>&1

echo
echo "--- (3) tsae_paper seed 2 ---"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/train_tsae_paper.py --seed 2 2>&1

echo "===== PHASE 6.1 TRAINING DONE: $(date -u) ====="
