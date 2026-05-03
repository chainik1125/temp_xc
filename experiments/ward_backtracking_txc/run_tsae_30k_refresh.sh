#!/usr/bin/env bash
# Post-train: re-mine TSAE features, re-run TSAE b3 sweep, rebuild flip matrix +
# calibration + plots, refresh docs/images_b/ copies. Mirrors the steps from
# run_headline_pipeline.sh and run_tfa_mlc_extension.sh but only for TSAE.
set -euo pipefail
cd /workspace/aniket/temp_xc

OUT_ROOT="results/ward_backtracking_txc/b3_math500_cut25"
PHASE1="results/ward_backtracking_txc/b3_math500/phase1_unsteered.json"
INCLUDE_CORRECT="${INCLUDE_CORRECT:-30}"
GEN_BATCH="${GEN_BATCH:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
DST_IMAGES="docs/aniket/experiments/ward_backtracking/images_b"

echo "[refresh] $(date) === re-mining TSAE features ==="
uv run python -m experiments.ward_backtracking_txc.mine_features \
    --cell tsae__resid_L10__k32__s42

# Resolve top-mean-diff feature for the new TSAE
TSAE_FEATURE=$(uv run python -c "
import numpy as np
z = np.load('results/ward_backtracking_txc/features/tsae__resid_L10__k32__s42.npz', allow_pickle=True)
print(int(z['top_features'][0]))
")
echo "[refresh] TSAE top feature = $TSAE_FEATURE"

# Setup new run dir for TSAE
TSAE_RUN_DIR="$OUT_ROOT/tsae__resid_L10__k32__s42__f${TSAE_FEATURE}_pos0"
mkdir -p "$TSAE_RUN_DIR"
cat > "$TSAE_RUN_DIR/meta.json" <<EOF
{"label": "TSAE-paper", "cell_id": "tsae__resid_L10__k32__s42", "feature_id": $TSAE_FEATURE, "feature_mode": "pos0"}
EOF

# Pull magnitudes from config
MAGS=$(uv run python -c "
import yaml; print(' '.join(str(m) for m in yaml.safe_load(open('experiments/ward_backtracking_txc/config.yaml'))['steering']['magnitudes']))
")

echo "[refresh] $(date) === b3 sweep for TSAE-paper (single GPU) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m experiments.ward_backtracking_txc.b3_variants \
    --variant cut25 \
    --steering-cell tsae__resid_L10__k32__s42 \
    --feature-id "$TSAE_FEATURE" \
    --feature-mode pos0 \
    --magnitudes $MAGS \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --gen-batch-size "$GEN_BATCH" \
    --include-correct "$INCLUDE_CORRECT" \
    --correct-seed 42 \
    --out "$TSAE_RUN_DIR" > logs/b3_sweep_tsae_30k_refresh.log 2>&1
echo "[refresh] $(date) === sweep done ==="

# Rebuild flip-matrix + calibration + plots across all 6 archs
RUN_ARGS=""
for d in "$OUT_ROOT"/*__f*_*/; do
  if [ -f "$d/meta.json" ] && [ -f "$d/phase2_rescue.json" ]; then
    RUN_ARGS+=" $d"
  fi
done
echo "[refresh] running:$RUN_ARGS"

uv run python -m experiments.ward_backtracking_txc.build_flip_matrix \
    --phase1 "$PHASE1" --runs $RUN_ARGS --out "$OUT_ROOT"
uv run python -m experiments.ward_backtracking_txc.calibrate_magnitudes \
    --runs $RUN_ARGS --out "$OUT_ROOT/calibration.json"
uv run python -m experiments.ward_backtracking_txc.plot.headline_steering \
    --runs $RUN_ARGS \
    --calibration "$OUT_ROOT/calibration.json" \
    --flip-matrix "$OUT_ROOT/flip_matrix.parquet" \
    --out "$OUT_ROOT"
uv run python -m experiments.ward_backtracking_txc.plot.repetition_rate \
    --runs $RUN_ARGS \
    --calibration "$OUT_ROOT/calibration.json" \
    --out "$OUT_ROOT" --label-filter
uv run python -m experiments.ward_backtracking_txc.plot.repetition_rate \
    --runs $RUN_ARGS \
    --calibration "$OUT_ROOT/calibration.json" \
    --out "$OUT_ROOT"

# Re-run hygiene + detection + flip-matrix grids
uv run python -m experiments.ward_backtracking_txc.build_hygiene_table
uv run python -m experiments.ward_backtracking_txc.detection.build_detection_probe
uv run python -m experiments.ward_backtracking_txc.plot.flip_matrix_grid \
    --flip-matrix "$OUT_ROOT/flip_matrix.parquet" --out "$OUT_ROOT"

# Refresh images_b copies
for f in headline_calibrated headline_raw appendix_calibrated appendix_raw \
         repetition_rate_headline repetition_rate \
         flip_matrix_grid_headline flip_matrix_grid_appendix \
         flip_matrix_grid_at_mag_0 flip_matrix_grid_at_mag_p8; do
  cp "$OUT_ROOT/$f.png" "$DST_IMAGES/np_$f.png"
done
cp results/ward_backtracking_txc/detection/detection_headline.png "$DST_IMAGES/np_detection_headline.png"
cp results/ward_backtracking_txc/detection/detection_appendix.png "$DST_IMAGES/np_detection_appendix.png"
cp results/ward_backtracking_txc/hygiene/training_curves/tsae_paper.png "$DST_IMAGES/np_training_curves/tsae_paper.png"

echo "[refresh] $(date) === Done. ==="
EOF
chmod +x experiments/ward_backtracking_txc/run_tsae_30k_refresh.sh
echo "wrote refresh script"