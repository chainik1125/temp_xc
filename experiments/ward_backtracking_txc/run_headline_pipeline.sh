#!/usr/bin/env bash
# Headline pipeline for the NeurIPS final-push backtracking case study.
#
# Prereqs (must be done before this script):
#  1. config.yaml has the densified 25-magnitude grid       (already shipped)
#  2. config.yaml has tsae.kval_topk=20                     (already shipped)
#  3. TSAE retrain completed for resid_L10 and ln1_L10      (kick off via
#     `train_txc.py --cell tsae__<hp>__k32__s42`)
#
# Steps:
#  A. Re-mine TSAE features at the new k=20 dictionary
#  B. Multi-arch B3 cut25 sweep (2-way parallel across GPUs)
#  C. Build flip-matrix parquet + McNemar table
#  D. Compute 95th-percentile per-arch calibration
#  E. Render calibrated + raw headline plots
#
# Usage:  bash experiments/ward_backtracking_txc/run_headline_pipeline.sh

set -euo pipefail

OUT_ROOT="results/ward_backtracking_txc/b3_math500_cut25"
PHASE1="results/ward_backtracking_txc/b3_math500/phase1_unsteered.json"
INCLUDE_CORRECT="${INCLUDE_CORRECT:-30}"
GEN_BATCH="${GEN_BATCH:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

echo "===== A. Re-mine TSAE features (k=20) ====="
for cell in tsae__resid_L10__k32__s42 tsae__ln1_L10__k32__s42 tsae__attn_L10__k32__s42; do
  ckpt="results/ward_backtracking_txc/checkpoints/${cell}.pt"
  if [ -f "$ckpt" ]; then
    echo "[mine] $cell"
    uv run python -m experiments.ward_backtracking_txc.mine_features --cell "$cell"
  else
    echo "[skip mine] $cell (no checkpoint yet)"
  fi
done

echo "===== B. Multi-arch B3 cut25 sweep (2-way GPU parallel) ====="
# Resolve TSAE feature pick (after re-mining).
TSAE_FEATURE=$(uv run python -c "
import numpy as np
z = np.load('results/ward_backtracking_txc/features/tsae__resid_L10__k32__s42.npz', allow_pickle=True)
print(int(z['top_features'][0]))
")
echo "[tsae feature] $TSAE_FEATURE"

# Pair 1 (GPU 0): TXC + SAE
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.ward_backtracking_txc.run_b3_multi_arch \
    --variant cut25 \
    --include-correct "$INCLUDE_CORRECT" \
    --gen-batch-size "$GEN_BATCH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --tsae-feature "$TSAE_FEATURE" \
    --archs TXC SAE \
    --out-root "$OUT_ROOT" > logs/b3_sweep_pair1.log 2>&1 &
P1=$!
echo "[gpu0 pair1] pid=$P1 (TXC + SAE)"

# Pair 2 (GPU 1): TXC-H8 + TSAE-paper
CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.ward_backtracking_txc.run_b3_multi_arch \
    --variant cut25 \
    --include-correct "$INCLUDE_CORRECT" \
    --gen-batch-size "$GEN_BATCH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --tsae-feature "$TSAE_FEATURE" \
    --archs TXC-H8 TSAE-paper \
    --out-root "$OUT_ROOT" > logs/b3_sweep_pair2.log 2>&1 &
P2=$!
echo "[gpu1 pair2] pid=$P2 (TXC-H8 + TSAE-paper)"

wait $P1 || echo "[warn] pair1 exit=$?"
wait $P2 || echo "[warn] pair2 exit=$?"

echo "===== C. Build flip matrix + McNemar ====="
RUN_ARGS=""
for d in "$OUT_ROOT"/*__f*_*/; do
  if [ -f "$d/meta.json" ] && [ -f "$d/phase2_rescue.json" ]; then
    RUN_ARGS+=" $d"
  fi
done
echo "[runs found]$RUN_ARGS"

uv run python -m experiments.ward_backtracking_txc.build_flip_matrix \
    --phase1 "$PHASE1" \
    --runs $RUN_ARGS \
    --out "$OUT_ROOT"

echo "===== D. Compute calibration ====="
uv run python -m experiments.ward_backtracking_txc.calibrate_magnitudes \
    --runs $RUN_ARGS \
    --out "$OUT_ROOT/calibration.json"

echo "===== E. Render headline plots ====="
uv run python -m experiments.ward_backtracking_txc.plot.headline_steering \
    --runs $RUN_ARGS \
    --calibration "$OUT_ROOT/calibration.json" \
    --flip-matrix "$OUT_ROOT/flip_matrix.parquet" \
    --out "$OUT_ROOT"

echo "===== Done. ====="
echo "Headline plots:    $OUT_ROOT/headline_calibrated.png  +  $OUT_ROOT/headline_raw.png"
echo "Flip matrix:       $OUT_ROOT/flip_matrix.parquet"
echo "McNemar table:     $OUT_ROOT/mcnemar_table.csv"
echo "Calibration:       $OUT_ROOT/calibration.json"
