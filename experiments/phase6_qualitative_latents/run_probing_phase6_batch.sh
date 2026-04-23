#!/usr/bin/env bash
# Phase 6 probing batch: Track 2 + Cycle A + Cycle H at seed=42,
# both aggregations. Complements run_phase61_post_train.sh which
# does Cycle F + 2x2 cell at all seeds.
#
# These three archs are all in the Phase 5 probing dispatcher but
# weren't part of the Phase 5.7 benchmark. The paper needs them
# on the same comparison as Cycle F and 2x2 cell.
#
# Usage (safely queue after current probing finishes):
#   bash experiments/phase6_qualitative_latents/run_probing_phase6_batch.sh

set -euo pipefail
cd /workspace/temp_xc
source /workspace/temp_xc/.envrc
mkdir -p logs

echo "===== Phase 6 probing batch: Track 2, Cycle A, Cycle H ====="
for ARCH in agentic_txc_10_bare agentic_txc_09_auxk agentic_txc_11_stack; do
  for AGG in last_position mean_pool; do
    CKPT="experiments/phase5_downstream_utility/results/ckpts/${ARCH}__seed42.pt"
    if [ ! -f "$CKPT" ]; then
      echo "[skip] $ARCH seed=42 (ckpt missing)"
      continue
    fi
    echo "--- $ARCH seed=42 / $AGG ---"
    TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
      experiments/phase5_downstream_utility/probing/run_probing.py \
      --aggregation "$AGG" \
      --run-ids "${ARCH}__seed42" \
      --skip-baselines 2>&1 | tail -5
  done
done
echo "===== Phase 6 probing batch DONE ====="
