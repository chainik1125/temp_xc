#!/usr/bin/env bash
# All-layers MLC phase — train a MultiLayerCrosscoder across EVERY transformer
# block (L=28 for Qwen-2.5-7B) and produce its own frontier for comparison.
#
# Memory analysis on an 80GB H100 (Qwen-7B fp16 ≈ 14GB resident):
#   buffer @ 250 seqs × 256 × 28 × fp16 ≈ 12.8 GB
#   MLC @ d_sae=8192, L=28, fp32 Adam ≈ 26 GB (params + grad + Adam)
#   + model + misc ≈ 55 GB total. Safe.
#
# For the sweep, we pick k=10 features; each feature contributes 28 steering
# directions so ActivationSteerer hooks all 28 layers simultaneously. The
# per-layer α magnitude is the same as the L=5 runs; total perturbation is
# ~5× larger, which may collapse the model at extreme α. If so, rerun the
# sweep with a shrunken α grid (TODO: surface that as a CLI flag later).
set -euo pipefail

TEMP_XC=/root/temp_xc
EM=/root/em_features
RESULTS=$EM/results
FIG_DIR=$RESULTS/figures
DOCS_RESULTS=$TEMP_XC/docs/dmitry/results/em_features
CKPT_DIR=$TEMP_XC/experiments/em_features/checkpoints
DIFF=$RESULTS/qwen_l15_sae/01_differences/difference_vectors.pt
SAE_SWEEP=$RESULTS/qwen_l15_sae_bundled_frontier_extended.json
ALPHA_GRID=(-10 -8 -6 -5 -4 -3 -2 -1.5 -1.0 1.0 2.0 5.0)
BASELINE_ALIGN=64.19
BASELINE_COH=84.88

D_SAE_ALL=8192
K_TOTAL=128
LAYERS_ALL="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27"
BUFFER_SEQS=250

mkdir -p "$CKPT_DIR" "$FIG_DIR" "$DOCS_RESULTS"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "GPU before all-layers MLC:"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader || true

ALL_PREFIX=$CKPT_DIR/qwen_mlc_all_d${D_SAE_ALL}
SNAP_40K=${ALL_PREFIX}_step40000.pt
SNAP_100K=${ALL_PREFIX}_step100000.pt
SNAP_300K=${ALL_PREFIX}_step300000.pt

if [ ! -s "$SNAP_300K" ]; then
    log "TRAIN all-layers MLC (d_sae=$D_SAE_ALL, L=28) × 300k steps, snapshots at 40k/100k/300k"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_training_mlc_snapshots \
        --config $TEMP_XC/experiments/em_features/config.yaml \
        --layers $LAYERS_ALL \
        --d_sae $D_SAE_ALL --k_total $K_TOTAL \
        --buffer_seqs $BUFFER_SEQS \
        --total_steps 300000 --snapshot_at 40000 100000 300000 \
        --out_prefix "$ALL_PREFIX"
fi

SWEEP_ARGS=("--sweep" "SAE_k10=$SAE_SWEEP")

for STEP in 40000 100000 300000; do
    CKPT=${ALL_PREFIX}_step${STEP}.pt
    OUT_DIR=$TEMP_XC/experiments/em_features/results/qwen_mlc_all_d${D_SAE_ALL}_step${STEP}
    OUT_JSON=$RESULTS/qwen_mlc_all_d${D_SAE_ALL}_step${STEP}_frontier.json

    log "Stage A for all-layers MLC @ ${STEP}"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_find_misalignment_features_mlc \
        --ckpt "$CKPT" \
        --diff_vectors "$DIFF" \
        --out "$OUT_DIR"

    log "Sweep all-layers MLC @ ${STEP}"
    cd $EM && python -m feature_ablation.frontier_sweep \
        --steerer mlc --model qwen --layer 15 \
        --features_json $OUT_DIR/top_200_features_mlc.json \
        --mlc_ckpt "$CKPT" \
        --k 10 \
        --alpha_grid "${ALPHA_GRID[@]}" \
        --n_rollouts 8 \
        --out_path "$OUT_JSON"

    SWEEP_ARGS+=("--sweep" "MLC_all${D_SAE_ALL}_${STEP}=$OUT_JSON")
done

log "plot + summary all-layers MLC"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.plot_frontier \
    "${SWEEP_ARGS[@]}" \
    --out $FIG_DIR/frontier_mlc_all_layers.png \
    --title "All-layers MLC (L=28, d_sae=${D_SAE_ALL}) vs SAE baseline"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.write_frontier_summary \
    "${SWEEP_ARGS[@]}" \
    --baseline_align $BASELINE_ALIGN --baseline_coh $BASELINE_COH \
    --out $DOCS_RESULTS/summary_mlc_all_layers.md \
    --title "All-layers MLC (L=28) frontier"

log "all-layers MLC complete"
