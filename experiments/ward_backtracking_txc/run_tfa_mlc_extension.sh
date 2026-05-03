#!/usr/bin/env bash
# Post-sweep extension: cache layers {8,9,11,12}, train TFA + MLC, mine, run
# B3 sweeps for both, then regenerate the headline plot with all 5 lines
# (TXC, SAE, TSAE-paper, TFA, MLC).
#
# Prereqs: the main sweep (run_headline_pipeline.sh) has completed for
# {TXC, SAE, TXC-H8, TSAE-paper}.
#
# Runs sequentially because the caching, training, and sweeps each want
# substantial GPU. With 2 H100s available some pairs run in parallel.
#
# Usage: bash experiments/ward_backtracking_txc/run_tfa_mlc_extension.sh

set -euo pipefail

OUT_ROOT="results/ward_backtracking_txc/b3_math500_cut25"
PHASE1="results/ward_backtracking_txc/b3_math500/phase1_unsteered.json"
INCLUDE_CORRECT="${INCLUDE_CORRECT:-30}"
GEN_BATCH="${GEN_BATCH:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

echo "===== F. Cache extra layers for MLC: L8, L9, L11, L12 ====="
# L10 already cached. L8/L9/L11/L12 share one model load.
uv run python -m experiments.ward_backtracking_txc.cache_activations \
    --override-hookpoint resid_L8:8:resid \
    --override-hookpoint resid_L9:9:resid \
    --override-hookpoint resid_L11:11:resid \
    --override-hookpoint resid_L12:12:resid

echo "===== G. Retrain TFA at resid_L10 ====="
# TFA reuses the same single-layer cache as TSAE; just a new arch entry.
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.ward_backtracking_txc.train_txc \
    --cell tfa__resid_L10__k32__s42 > logs/train_tfa_resid_L10.log 2>&1 &
P_TFA=$!

echo "===== H. Train MLC across L{8,9,10,11,12} ====="
CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.ward_backtracking_txc.train_txc \
    --cell mlc__resid_L10__k32__s42 > logs/train_mlc_resid_L10.log 2>&1 &
P_MLC=$!

wait $P_TFA && echo "[TFA train] done" || echo "[TFA train] failed exit=$?"
wait $P_MLC && echo "[MLC train] done" || echo "[MLC train] failed exit=$?"

echo "===== I. Mine TFA + MLC features ====="
uv run python -m experiments.ward_backtracking_txc.mine_features --cell tfa__resid_L10__k32__s42
uv run python -m experiments.ward_backtracking_txc.mine_features --cell mlc__resid_L10__k32__s42

echo "===== J. Resolve top features for new arches ====="
TFA_FEATURE=$(uv run python -c "
import numpy as np
z = np.load('results/ward_backtracking_txc/features/tfa__resid_L10__k32__s42.npz', allow_pickle=True)
print(int(z['top_features'][0]))
")
MLC_FEATURE=$(uv run python -c "
import numpy as np
z = np.load('results/ward_backtracking_txc/features/mlc__resid_L10__k32__s42.npz', allow_pickle=True)
print(int(z['top_features'][0]))
")
echo "[features] TFA=$TFA_FEATURE  MLC=$MLC_FEATURE"

echo "===== K. B3 cut25 sweeps for TFA + MLC (parallel across GPUs) ====="
# Build per-arch run dirs with meta.json files for build_flip_matrix.
TFA_RUN_DIR="$OUT_ROOT/tfa__resid_L10__k32__s42__f${TFA_FEATURE}_pos0"
MLC_RUN_DIR="$OUT_ROOT/mlc__resid_L10__k32__s42__f${MLC_FEATURE}_pos0"
mkdir -p "$TFA_RUN_DIR" "$MLC_RUN_DIR"

cat > "$TFA_RUN_DIR/meta.json" <<EOF
{"label": "TFA", "cell_id": "tfa__resid_L10__k32__s42", "feature_id": $TFA_FEATURE, "feature_mode": "pos0"}
EOF
cat > "$MLC_RUN_DIR/meta.json" <<EOF
{"label": "MLC", "cell_id": "mlc__resid_L10__k32__s42", "feature_id": $MLC_FEATURE, "feature_mode": "pos0"}
EOF

# Pull magnitudes from config so the grid stays aligned with the main sweep.
MAGS=$(uv run python -c "
import yaml; print(' '.join(str(m) for m in yaml.safe_load(open('experiments/ward_backtracking_txc/config.yaml'))['steering']['magnitudes']))
")

CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.ward_backtracking_txc.b3_variants \
    --variant cut25 \
    --steering-cell tfa__resid_L10__k32__s42 \
    --feature-id "$TFA_FEATURE" \
    --feature-mode pos0 \
    --magnitudes $MAGS \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --gen-batch-size "$GEN_BATCH" \
    --include-correct "$INCLUDE_CORRECT" \
    --correct-seed 42 \
    --out "$TFA_RUN_DIR" > logs/b3_sweep_tfa.log 2>&1 &
P_TFA_B3=$!

CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.ward_backtracking_txc.b3_variants \
    --variant cut25 \
    --steering-cell mlc__resid_L10__k32__s42 \
    --feature-id "$MLC_FEATURE" \
    --feature-mode pos0 \
    --magnitudes $MAGS \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --gen-batch-size "$GEN_BATCH" \
    --include-correct "$INCLUDE_CORRECT" \
    --correct-seed 42 \
    --out "$MLC_RUN_DIR" > logs/b3_sweep_mlc.log 2>&1 &
P_MLC_B3=$!

wait $P_TFA_B3 && echo "[TFA b3] done" || echo "[TFA b3] failed exit=$?"
wait $P_MLC_B3 && echo "[MLC b3] done" || echo "[MLC b3] failed exit=$?"

echo "===== L. Rebuild flip-matrix + calibration + headline plot (5 archs) ====="
RUN_ARGS=""
for d in "$OUT_ROOT"/*__f*_*/; do
  if [ -f "$d/meta.json" ] && [ -f "$d/phase2_rescue.json" ]; then
    RUN_ARGS+=" $d"
  fi
done

uv run python -m experiments.ward_backtracking_txc.build_flip_matrix \
    --phase1 "$PHASE1" --runs $RUN_ARGS --out "$OUT_ROOT"

uv run python -m experiments.ward_backtracking_txc.calibrate_magnitudes \
    --runs $RUN_ARGS --out "$OUT_ROOT/calibration.json"

uv run python -m experiments.ward_backtracking_txc.plot.headline_steering \
    --runs $RUN_ARGS \
    --calibration "$OUT_ROOT/calibration.json" \
    --flip-matrix "$OUT_ROOT/flip_matrix.parquet" \
    --out "$OUT_ROOT"

echo "===== Done. ====="
echo "Headline plot:    $OUT_ROOT/headline_calibrated.png"
echo "All run dirs:    $RUN_ARGS"
