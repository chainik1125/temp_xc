#!/usr/bin/env bash
# Big-TXC chain: train d_sae=65536 TXC × 200k steps with INTERMIXED
# training + sweep phases so we see frontiers at 40k and 100k midway rather
# than only at the end.
#
# Phases:
#   1. train 0 → 40k      (fresh)
#   2. sweep @ 40k        (Stage A + frontier sweep)
#   3. train 40k → 100k   (--resume_from the 40k ckpt)
#   4. sweep @ 100k
#   5. train 100k → 200k  (--resume_from the 100k ckpt)
#   6. sweep @ 200k
#   7. plot + summary
#   8. hand off to Arditi phase (resid_mid + resid_pre) unless stopped.
#
# Model-only resume loses Adam momentum at each phase boundary; we accept
# the small convergence hit in exchange for manageable checkpoint sizes
# (9 GB/ckpt at d_sae=65k, vs 28 GB if we also saved optimizer state).
set -euo pipefail

TEMP_XC=/root/temp_xc
EM=/root/em_features
RESULTS=$EM/results
FIG_DIR=$RESULTS/figures
DOCS_RESULTS=$TEMP_XC/docs/dmitry/results/em_features
CKPT_DIR=$TEMP_XC/experiments/em_features/checkpoints
DIFF=$RESULTS/qwen_l15_sae/01_differences/difference_vectors.pt
SAE_SWEEP=$RESULTS/qwen_l15_sae_bundled_frontier_extended.json
SMALL_TXC_200K=$RESULTS/qwen_l15_txc_small_step200000_frontier.json
ALPHA_GRID=(-10 -8 -6 -5 -4 -3 -2 -1.5 -1.0 1.0 2.0 5.0)
BASELINE_ALIGN=64.19
BASELINE_COH=84.88
LAYER=15
BIG_DSAE=65536
PREFIX=$CKPT_DIR/qwen_l15_txc_big_d${BIG_DSAE}

mkdir -p "$CKPT_DIR" "$FIG_DIR" "$DOCS_RESULTS"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

train_phase() {
    local total_steps=$1
    local resume=${2:-}
    local args=(
        --config $TEMP_XC/experiments/em_features/config.yaml
        --d_sae $BIG_DSAE --T 5 --k_total 128
        --total_steps $total_steps --snapshot_at $total_steps
        --out_prefix "$PREFIX"
    )
    if [ -n "$resume" ]; then
        args+=(--resume_from "$resume")
    fi
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_training_txc_snapshots "${args[@]}"
}

sweep_phase() {
    local step=$1
    local ckpt=${PREFIX}_step${step}.pt
    local out_dir=$TEMP_XC/experiments/em_features/results/qwen_l15_txc_big_d${BIG_DSAE}_step${step}
    local out_json=$RESULTS/qwen_l15_txc_big_d${BIG_DSAE}_step${step}_frontier.json

    log "Stage A big TXC @ ${step}"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_find_misalignment_features \
        --ckpt "$ckpt" --diff_vectors "$DIFF" --layer $LAYER --out "$out_dir"

    log "Sweep big TXC @ ${step}"
    cd $EM && python -m feature_ablation.frontier_sweep \
        --steerer txc --model qwen --layer $LAYER \
        --features_json $out_dir/top_200_features_layer_${LAYER}.json \
        --txc_ckpt "$ckpt" \
        --k 10 --alpha_grid "${ALPHA_GRID[@]}" --n_rollouts 8 \
        --out_path "$out_json"
    echo "$out_json"
}

log "RUN B: big TXC (d_sae=$BIG_DSAE) — phase 1: 0 → 40k"
[ -s "${PREFIX}_step40000.pt" ] || train_phase 40000
JSON_40K=$(sweep_phase 40000 | tail -1)

log "phase 2: 40k → 100k (resume)"
[ -s "${PREFIX}_step100000.pt" ] || train_phase 100000 "${PREFIX}_step40000.pt"
JSON_100K=$(sweep_phase 100000 | tail -1)

log "phase 3: 100k → 200k (resume)"
[ -s "${PREFIX}_step200000.pt" ] || train_phase 200000 "${PREFIX}_step100000.pt"
JSON_200K=$(sweep_phase 200000 | tail -1)

log "plot + summary big-TXC scaling"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.plot_frontier \
    --sweep "SAE_k10=$SAE_SWEEP" \
    --sweep "TXC_small_200k=$SMALL_TXC_200K" \
    --sweep "TXC_big${BIG_DSAE}_40k=$JSON_40K" \
    --sweep "TXC_big${BIG_DSAE}_100k=$JSON_100K" \
    --sweep "TXC_big${BIG_DSAE}_200k=$JSON_200K" \
    --out $FIG_DIR/frontier_txc_big_scaling.png \
    --title "TXC big-dict (d_sae=${BIG_DSAE}): 40k → 100k → 200k vs SAE + small TXC 200k"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.write_frontier_summary \
    --sweep "SAE_k10=$SAE_SWEEP" \
    --sweep "TXC_small_200k=$SMALL_TXC_200K" \
    --sweep "TXC_big${BIG_DSAE}_40k=$JSON_40K" \
    --sweep "TXC_big${BIG_DSAE}_100k=$JSON_100K" \
    --sweep "TXC_big${BIG_DSAE}_200k=$JSON_200K" \
    --baseline_align $BASELINE_ALIGN --baseline_coh $BASELINE_COH \
    --out $DOCS_RESULTS/summary_txc_big_scaling.md \
    --title "TXC big-dict (d_sae=${BIG_DSAE}) dict×steps scaling"

log "big-TXC chain complete"

if [ -e "$EM/.stop_arditi" ]; then
    log "Arditi phase skipped — $EM/.stop_arditi present (rm to re-enable)"
else
    log "auto-chain: Arditi phase (resid_mid + resid_pre only, per scope)"
    bash $TEMP_XC/experiments/em_features/run_arditi_sae_phase.sh
fi

log "everything complete on a100_1"
