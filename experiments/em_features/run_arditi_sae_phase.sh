#!/usr/bin/env bash
# Train custom TopK SAEs on Qwen-2.5-7B-Instruct residuals at
# resid_pre / resid_mid / ln_1_hook_normalized (layer 15), then run the
# em-features frontier sweep at k ∈ {50, 100} for each, plus a final
# mega-comparison plot and .md.
#
# Dispatched by run_post_txc_chain.sh after the MLC comparison completes,
# unless /root/em_features/.stop_arditi exists.
set -euo pipefail

TEMP_XC=/root/temp_xc
EM=/root/em_features
RESULTS=$EM/results
FIG_DIR=$RESULTS/figures
DOCS_RESULTS=$TEMP_XC/docs/dmitry/results/em_features
DIFF=$RESULTS/qwen_l15_sae/01_differences/difference_vectors.pt
ALPHA_GRID=(-10 -8 -6 -5 -4 -3 -2 -1.5 -1.0 1.0 2.0 5.0)
BASELINE_ALIGN=64.19
BASELINE_COH=84.88

LAYER=15
# Scope: just resid_pre and resid_mid for this pass (ln1_normalized dropped
# per user request — we'll add it back later if resid_mid/pre show interesting
# diffs vs resid_post).
HOOKPOINTS=(resid_pre resid_mid)
CUSTOM_DIFF_DIR=$RESULTS/qwen_l${LAYER}_custom_diffs

mkdir -p "$FIG_DIR" "$DOCS_RESULTS" "$CUSTOM_DIFF_DIR"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# Compute diffs for hookpoints not already in em-features' difference_vectors.pt.
# (resid_pre can be read off layer-1's resid_post in that file; resid_mid and
# ln1_normalized need module-level hooks.)
if [ ! -s "$CUSTOM_DIFF_DIR/custom_diffs_resid_mid_L${LAYER}.pt" ]; then
    log "compute_custom_diffs for resid_mid"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.compute_custom_diffs \
        --dataset /root/em_features/data/medical_advice_prompt_only.jsonl \
        --base Qwen/Qwen2.5-7B-Instruct \
        --bad andyrdt/Qwen2.5-7B-Instruct_bad-medical \
        --hookpoints resid_mid \
        --layer $LAYER \
        --out_dir "$CUSTOM_DIFF_DIR"
fi

diff_path_for() {
    # Echo the correct --diff_vectors path for a given hookpoint.
    local hp=$1
    case "$hp" in
        resid_pre|resid_post) echo "$DIFF" ;;
        resid_mid)            echo "$CUSTOM_DIFF_DIR/custom_diffs_resid_mid_L${LAYER}.pt" ;;
        ln1_normalized)       echo "$CUSTOM_DIFF_DIR/custom_diffs_ln1_normalized_L${LAYER}.pt" ;;
        *) echo "UNKNOWN"; return 1 ;;
    esac
}

SWEEP_ARGS=()
# Seed the mega-plot with the earlier results (SAE/TXC/MLC at resid_post, k=10).
SWEEP_ARGS+=("--sweep" "SAE_post_k10=$RESULTS/qwen_l15_sae_bundled_frontier_extended.json")
SWEEP_ARGS+=("--sweep" "TXC_post_k10=$RESULTS/qwen_l15_txc_bundled_frontier.json")
SWEEP_ARGS+=("--sweep" "MLC_multi_k10=$RESULTS/qwen_mlc_bundled_frontier.json")

for HP in "${HOOKPOINTS[@]}"; do
    CKPT=$TEMP_XC/experiments/em_features/checkpoints/qwen_l${LAYER}_sae_${HP}_k128.pt
    FEATS_JSON=$TEMP_XC/experiments/em_features/results/qwen_l${LAYER}_sae_${HP}/top_200_features.json

    log "TRAIN custom TopKSAE @ $HP (layer $LAYER)"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_training_sae_custom \
        --config $TEMP_XC/experiments/em_features/config.yaml \
        --hookpoint "$HP" \
        --layer $LAYER \
        --out "$CKPT"

    log "STAGE A @ $HP"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_find_custom_sae_features \
        --ckpt "$CKPT" \
        --diff_vectors "$(diff_path_for "$HP")" \
        --hookpoint "$HP" \
        --layer $LAYER \
        --out "$TEMP_XC/experiments/em_features/results/qwen_l${LAYER}_sae_${HP}"

    for K in 50 100; do
        OUT_JSON=$RESULTS/qwen_l${LAYER}_sae_${HP}_bundled_frontier_k${K}.json
        log "SWEEP custom SAE @ $HP, k=$K"
        cd $EM && python -m feature_ablation.frontier_sweep \
            --steerer custom_sae --model qwen --layer $LAYER \
            --features_json "$FEATS_JSON" \
            --custom_sae_ckpt "$CKPT" \
            --hookpoint "$HP" \
            --k $K \
            --alpha_grid "${ALPHA_GRID[@]}" \
            --n_rollouts 8 \
            --out_path "$OUT_JSON"
        SWEEP_ARGS+=("--sweep" "SAE_${HP}_k${K}=$OUT_JSON")
    done
done

log "MEGA plot + summary"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.plot_frontier \
    "${SWEEP_ARGS[@]}" \
    --out $FIG_DIR/frontier_mega_qwen_l${LAYER}.png \
    --title "Coherence / suppression frontier — mega comparison (Qwen-2.5-7B bad-medical, L$LAYER)"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.write_frontier_summary \
    "${SWEEP_ARGS[@]}" \
    --baseline_align $BASELINE_ALIGN --baseline_coh $BASELINE_COH \
    --out $DOCS_RESULTS/summary_mega_qwen_l${LAYER}.md \
    --title "Mega frontier comparison (Qwen-2.5-7B bad-medical, L$LAYER, all hookpoints)"

log "arditi phase complete"
