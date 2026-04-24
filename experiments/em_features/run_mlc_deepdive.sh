#!/usr/bin/env bash
# MLC deep-dive (mirror of run_txc_deepdive.sh) — runs on h100_2.
#
#   Run A — small MLC, d_sae=32768, L=5 (layers 11/13/15/17/19), 300k steps.
#   Runs B/C/D — big MLC, d_sae=$BIG_DSAE, single 300k-step training with
#                intermediate snapshots at 40k/100k/300k; one sweep per ckpt.
#
# Sweeps reuse the SAE / TXC / small-MLC α grid for a direct apples-to-apples
# comparison with the frontier results from a100_1 (file paths below assume
# those JSONs were rsync'd onto h100_2 at /root/em_features/results/).
set -euo pipefail

TEMP_XC=/root/temp_xc
EM=/root/em_features
RESULTS=$EM/results
FIG_DIR=$RESULTS/figures
DOCS_RESULTS=$TEMP_XC/docs/dmitry/results/em_features
CKPT_DIR=$TEMP_XC/experiments/em_features/checkpoints
DIFF=$RESULTS/qwen_l15_sae/01_differences/difference_vectors.pt
SAE_SWEEP=$RESULTS/qwen_l15_sae_bundled_frontier_extended.json
SMALL_MLC_SWEEP=$RESULTS/qwen_mlc_bundled_frontier.json   # the original 40k L5 MLC
ALPHA_GRID=(-10 -8 -6 -5 -4 -3 -2 -1.5 -1.0 1.0 2.0 5.0)
BASELINE_ALIGN=64.19
BASELINE_COH=84.88
LAYER=15
LAYERS_L5="11 13 15 17 19"
BIG_DSAE=65536

mkdir -p "$CKPT_DIR" "$FIG_DIR" "$DOCS_RESULTS"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "GPU before Run A:"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader || true

# ---------------------------------------------------------------------------
# Run A — small MLC (d_sae=32768, L=5), 300k steps.
# ---------------------------------------------------------------------------
RUN_A_PREFIX=$CKPT_DIR/qwen_mlc_small
RUN_A_CKPT=${RUN_A_PREFIX}_step300000.pt
if [ ! -s "$RUN_A_CKPT" ]; then
    log "RUN A: small MLC (d_sae=32768, L=5) × 300k steps"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_training_mlc_snapshots \
        --config $TEMP_XC/experiments/em_features/config.yaml \
        --layers $LAYERS_L5 \
        --d_sae 32768 --k_total 128 \
        --total_steps 300000 --snapshot_at 300000 \
        --out_prefix "$RUN_A_PREFIX"
fi

log "Stage A for small MLC @ 300k"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_find_misalignment_features_mlc \
    --ckpt "$RUN_A_CKPT" \
    --diff_vectors "$DIFF" \
    --out $TEMP_XC/experiments/em_features/results/qwen_mlc_small_step300000

RUN_A_SWEEP=$RESULTS/qwen_mlc_small_step300000_frontier.json
log "Sweep small MLC @ 300k"
cd $EM && python -m feature_ablation.frontier_sweep \
    --steerer mlc --model qwen --layer $LAYER \
    --features_json $TEMP_XC/experiments/em_features/results/qwen_mlc_small_step300000/top_200_features_mlc.json \
    --mlc_ckpt "$RUN_A_CKPT" \
    --k 10 \
    --alpha_grid "${ALPHA_GRID[@]}" \
    --n_rollouts 8 \
    --out_path "$RUN_A_SWEEP"

log "intermediate plot after MLC Run A"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.plot_frontier \
    --sweep "SAE_k10=$SAE_SWEEP" \
    --sweep "MLC_small_40k=$SMALL_MLC_SWEEP" \
    --sweep "MLC_small_300k=$RUN_A_SWEEP" \
    --out $FIG_DIR/frontier_mlc_small_scaling.png \
    --title "MLC small-dict (d_sae=32768, L=5): 40k vs 300k training steps"

# ---------------------------------------------------------------------------
# Big-MLC L=5 phase intentionally SKIPPED — the L=5 at 40k and 300k data
# points already tell us dict-size and length scaling for regional MLC;
# next-highest-ROI work is all-layers MLC (see run_mlc_all_layers.sh).
# To re-enable, set EM_RUN_BIG_MLC=1 in the env.
# ---------------------------------------------------------------------------

SWEEP_ARGS=("--sweep" "SAE_k10=$SAE_SWEEP"
            "--sweep" "MLC_small_40k=$SMALL_MLC_SWEEP"
            "--sweep" "MLC_small_300k=$RUN_A_SWEEP")

log "MLC L=5 plot + summary (small-dict only)"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.plot_frontier \
    "${SWEEP_ARGS[@]}" \
    --out $FIG_DIR/frontier_mlc_l5_scaling.png \
    --title "MLC L=5, d_sae=32768: 40k → 300k steps vs SAE baseline"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.write_frontier_summary \
    "${SWEEP_ARGS[@]}" \
    --baseline_align $BASELINE_ALIGN --baseline_coh $BASELINE_COH \
    --out $DOCS_RESULTS/summary_mlc_l5_scaling.md \
    --title "MLC L=5 dict=32768: training-length scaling"

log "MLC L=5 phase complete"

# ---------------------------------------------------------------------------
# If the deepdive succeeded, auto-continue to the all-layers MLC phase.
# ---------------------------------------------------------------------------
if [ -x $TEMP_XC/experiments/em_features/run_mlc_all_layers.sh ] && [ ! -e "$EM/.stop_mlc_all" ]; then
    log "auto-continue: run_mlc_all_layers.sh"
    bash $TEMP_XC/experiments/em_features/run_mlc_all_layers.sh
fi

log "everything complete on h100_2"
