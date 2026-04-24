#!/usr/bin/env bash
# TXC deep-dive: two training regimes after MLC completes.
#
#   Run A — small TXC, d_sae=32768 (current default), 300k steps, sweep.
#   Runs B/C/D — big TXC, d_sae=$BIG_DSAE, single 300k-step training with
#                intermediate snapshots at 40k/100k/300k; one sweep per ckpt.
#
# Launch on a100_1 in the em_features venv:
#   cd /root/em_features && source .venv/bin/activate && set -a && source .env && set +a
#   export TEMP_XC_REPO=/root/temp_xc
#   nohup bash /root/temp_xc/experiments/em_features/run_txc_deepdive.sh \
#     > /root/em_features/logs/txc_deepdive.log 2>&1 &
set -euo pipefail

TEMP_XC=/root/temp_xc
EM=/root/em_features
RESULTS=$EM/results
FIG_DIR=$RESULTS/figures
DOCS_RESULTS=$TEMP_XC/docs/dmitry/results/em_features
CKPT_DIR=$TEMP_XC/experiments/em_features/checkpoints
DIFF=$RESULTS/qwen_l15_sae/01_differences/difference_vectors.pt
SAE_SWEEP=$RESULTS/qwen_l15_sae_bundled_frontier_extended.json
MLC_SWEEP=$RESULTS/qwen_mlc_bundled_frontier.json
ALPHA_GRID=(-10 -8 -6 -5 -4 -3 -2 -1.5 -1.0 1.0 2.0 5.0)
BASELINE_ALIGN=64.19
BASELINE_COH=84.88
LAYER=15

# d_sae=65536 is the largest that fits with fp32 Adam alongside Qwen-7B +
# streaming buffer (~58 GB total vs 80 GB H100). Doubles the small-TXC dict.
BIG_DSAE=65536

mkdir -p "$CKPT_DIR" "$FIG_DIR" "$DOCS_RESULTS"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

wait_for() {
    local path=$1
    log "waiting for $path"
    while ! [ -s "$path" ]; do sleep 60; done
    log "found $path"
}

# 0. Wait for MLC chain to produce its frontier (signals the earlier chain is done).
wait_for "$MLC_SWEEP"
# Make doubly sure the Arditi phase won't fire mid-deepdive.
touch "$EM/.stop_arditi"

# Helpful sanity ping.
log "GPU before Run A:"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader || true

# ---------------------------------------------------------------------------
# Run A — small TXC (d_sae=32768), 300k steps.
# ---------------------------------------------------------------------------
RUN_A_CKPT=$CKPT_DIR/qwen_l15_txc_small_step300000.pt
if [ ! -s "$RUN_A_CKPT" ]; then
    log "RUN A: small TXC (d_sae=32768) × 300k steps"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_training_txc_snapshots \
        --config $TEMP_XC/experiments/em_features/config.yaml \
        --d_sae 32768 --T 5 --k_total 128 \
        --total_steps 300000 --snapshot_at 300000 \
        --out_prefix $CKPT_DIR/qwen_l15_txc_small
fi

log "Stage A for small TXC @ 300k"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_find_misalignment_features \
    --ckpt "$RUN_A_CKPT" \
    --diff_vectors "$DIFF" \
    --layer $LAYER \
    --out $TEMP_XC/experiments/em_features/results/qwen_l15_txc_small_step300000

RUN_A_SWEEP=$RESULTS/qwen_l15_txc_small_step300000_frontier.json
log "Sweep small TXC @ 300k"
cd $EM && python -m feature_ablation.frontier_sweep \
    --steerer txc --model qwen --layer $LAYER \
    --features_json $TEMP_XC/experiments/em_features/results/qwen_l15_txc_small_step300000/top_200_features_layer_${LAYER}.json \
    --txc_ckpt "$RUN_A_CKPT" \
    --k 10 \
    --alpha_grid "${ALPHA_GRID[@]}" \
    --n_rollouts 8 \
    --out_path "$RUN_A_SWEEP"

log "intermediate plot after Run A"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.plot_frontier \
    --sweep "SAE_k10=$SAE_SWEEP" \
    --sweep "TXC_small_40k=$RESULTS/qwen_l15_txc_bundled_frontier.json" \
    --sweep "TXC_small_300k=$RUN_A_SWEEP" \
    --out $FIG_DIR/frontier_txc_small_scaling.png \
    --title "TXC small-dict (d_sae=32768): 40k vs 300k training steps"

# ---------------------------------------------------------------------------
# Runs B / C / D — big TXC (d_sae=$BIG_DSAE), single training to 300k with
# mid-run snapshots. GPU: training ≈ 58 GB, safe on 80 GB H100.
# ---------------------------------------------------------------------------
log "GPU before Run B:"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader || true

BIG_PREFIX=$CKPT_DIR/qwen_l15_txc_big_d${BIG_DSAE}
SNAP_40K=$CKPT_DIR/qwen_l15_txc_big_d${BIG_DSAE}_step40000.pt
SNAP_100K=$CKPT_DIR/qwen_l15_txc_big_d${BIG_DSAE}_step100000.pt
SNAP_300K=$CKPT_DIR/qwen_l15_txc_big_d${BIG_DSAE}_step300000.pt

if [ ! -s "$SNAP_300K" ]; then
    log "RUN B+C+D: big TXC (d_sae=$BIG_DSAE) × 300k steps with snapshots at 40k/100k/300k"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_training_txc_snapshots \
        --config $TEMP_XC/experiments/em_features/config.yaml \
        --d_sae $BIG_DSAE --T 5 --k_total 128 \
        --total_steps 300000 --snapshot_at 40000 100000 300000 \
        --out_prefix "$BIG_PREFIX"
fi

SWEEP_ARGS=("--sweep" "SAE_k10=$SAE_SWEEP"
            "--sweep" "TXC_small_300k=$RUN_A_SWEEP")

for STEP in 40000 100000 300000; do
    CKPT=$CKPT_DIR/qwen_l15_txc_big_d${BIG_DSAE}_step${STEP}.pt
    OUT_DIR=$TEMP_XC/experiments/em_features/results/qwen_l15_txc_big_d${BIG_DSAE}_step${STEP}
    OUT_JSON=$RESULTS/qwen_l15_txc_big_d${BIG_DSAE}_step${STEP}_frontier.json

    log "Stage A for big TXC @ ${STEP}"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_find_misalignment_features \
        --ckpt "$CKPT" \
        --diff_vectors "$DIFF" \
        --layer $LAYER \
        --out "$OUT_DIR"

    log "Sweep big TXC @ ${STEP}"
    cd $EM && python -m feature_ablation.frontier_sweep \
        --steerer txc --model qwen --layer $LAYER \
        --features_json $OUT_DIR/top_200_features_layer_${LAYER}.json \
        --txc_ckpt "$CKPT" \
        --k 10 \
        --alpha_grid "${ALPHA_GRID[@]}" \
        --n_rollouts 8 \
        --out_path "$OUT_JSON"

    SWEEP_ARGS+=("--sweep" "TXC_big${BIG_DSAE}_${STEP}=$OUT_JSON")
done

log "MEGA TXC plot + summary"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.plot_frontier \
    "${SWEEP_ARGS[@]}" \
    --out $FIG_DIR/frontier_txc_deepdive.png \
    --title "TXC deep-dive: small(32k)@300k, big(${BIG_DSAE})@40k/100k/300k vs SAE baseline"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.write_frontier_summary \
    "${SWEEP_ARGS[@]}" \
    --baseline_align $BASELINE_ALIGN --baseline_coh $BASELINE_COH \
    --out $DOCS_RESULTS/summary_txc_deepdive.md \
    --title "TXC deep-dive: dict-size × training-length scaling"

log "txc deepdive complete"
