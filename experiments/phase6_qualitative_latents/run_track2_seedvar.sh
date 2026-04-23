#!/usr/bin/env bash
# Train Track 2 (agentic_txc_10_bare) at seeds {1, 2} for paper seed
# variance. Track 2 emerged as the TXC-family qualitative winner on
# the rigorous metric (5/32 random vs Cycle F's 0/32) and probing-
# tied with baseline at seed=42. Seed variance cinches the claim.
#
# Queued to run after Phase 6.1 pipeline completes (so it doesn't
# compete with Cycle F + 2x2 cell + tsae_paper training for GPU).

set -euo pipefail
cd /workspace/temp_xc
source /workspace/temp_xc/.envrc
mkdir -p logs

echo "===== Track 2 seed-variance training: $(date -u) ====="
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase5_downstream_utility/train_primary_archs.py \
  --archs agentic_txc_10_bare \
  --seeds 1 2 \
  --max-steps 25000 2>&1

echo
echo "--- Encode Track 2 seeds 1, 2 on A/B/random ---"
for SEED in 1 2; do
  TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
    experiments/phase6_qualitative_latents/encode_archs.py \
    --archs agentic_txc_10_bare \
    --sets A B random --seed "$SEED" 2>&1
done

echo
echo "--- Autointerp Track 2 seeds 1, 2 on A/B/random ---"
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/run_autointerp.py \
  --archs agentic_txc_10_bare \
  --seeds 1 2 --concats A B random 2>&1

echo
echo "--- Probe Track 2 seeds 1, 2 at both aggregations ---"
for SEED in 1 2; do
  for AGG in last_position mean_pool; do
    TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
      experiments/phase5_downstream_utility/probing/run_probing.py \
      --aggregation "$AGG" \
      --run-ids agentic_txc_10_bare__seed${SEED} \
      --skip-baselines 2>&1 | tail -5
  done
done

echo "===== Track 2 seed-variance DONE: $(date -u) ====="
