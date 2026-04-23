#!/usr/bin/env bash
# Orchestrate everything that happens AFTER TXC training finishes.
# Polls for the TXC checkpoint; then runs TXC Stage A, TXC frontier sweep,
# 2-way plot+summary, MLC training, MLC Stage A, MLC frontier sweep, 3-way
# plot+summary. Then triggers the Arditi-SAE phase (3 hookpoints: resid_mid,
# resid_pre, ln_1_hook_normalized) as a separate stage.
#
# Usage (on a100_1):
#   cd /root/em_features && source .venv/bin/activate && set -a && source .env && set +a
#   nohup bash /root/temp_xc/experiments/em_features/run_post_txc_chain.sh \
#     > /root/em_features/logs/post_txc_chain.log 2>&1 &
set -euo pipefail

TEMP_XC=/root/temp_xc
EM=/root/em_features
RESULTS=$EM/results
FIG_DIR=$RESULTS/figures
DOCS_RESULTS=$TEMP_XC/docs/dmitry/results/em_features
TXC_CKPT=$TEMP_XC/experiments/em_features/checkpoints/qwen_l15_txc_t5_k128.pt
MLC_CKPT=$TEMP_XC/experiments/em_features/checkpoints/qwen_mlc_l11-13-15-17-19_k128.pt
DIFF=$RESULTS/qwen_l15_sae/01_differences/difference_vectors.pt
SAE_SWEEP=$RESULTS/qwen_l15_sae_bundled_frontier_extended.json
TXC_SWEEP=$RESULTS/qwen_l15_txc_bundled_frontier.json
MLC_SWEEP=$RESULTS/qwen_mlc_bundled_frontier.json
ALPHA_GRID=(-10 -8 -6 -5 -4 -3 -2 -1.5 -1.0 1.0 2.0 5.0)
# Baseline numbers from the extended SAE sweep at α=0 (k=10, n_rollouts=8).
BASELINE_ALIGN=64.19
BASELINE_COH=84.88

mkdir -p "$FIG_DIR" "$DOCS_RESULTS"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

wait_for() {
    local path=$1
    log "waiting for $path"
    while ! [ -s "$path" ]; do sleep 60; done
    log "found $path"
}

# 1. Wait for TXC training to complete.
wait_for "$TXC_CKPT"

# 2. TXC Stage A — decompose diff onto TXC decoder.
log "TXC Stage A start"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_find_misalignment_features \
    --ckpt "$TXC_CKPT" \
    --diff_vectors "$DIFF" \
    --layer 15 \
    --out $TEMP_XC/experiments/em_features/results/qwen_l15_txc
log "TXC Stage A done"

# 3. TXC frontier sweep — same α grid as the extended SAE sweep.
log "TXC frontier sweep start"
cd $EM && python -m feature_ablation.frontier_sweep \
    --steerer txc --model qwen --layer 15 \
    --features_json $TEMP_XC/experiments/em_features/results/qwen_l15_txc/top_200_features_layer_15.json \
    --txc_ckpt "$TXC_CKPT" \
    --k 10 \
    --alpha_grid "${ALPHA_GRID[@]}" \
    --n_rollouts 8 \
    --out_path $TXC_SWEEP
log "TXC frontier sweep done"

# 3a. 2-way plot + markdown summary (SAE vs TXC).
log "2-way plot + summary"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.plot_frontier \
    --sweep "SAE=$SAE_SWEEP" \
    --sweep "TXC=$TXC_SWEEP" \
    --out $FIG_DIR/frontier_sae_vs_txc.png \
    --title "Coherence / suppression frontier: SAE vs TXC (Qwen-2.5-7B bad-medical, L15, k=10)"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.write_frontier_summary \
    --sweep "SAE=$SAE_SWEEP" \
    --sweep "TXC=$TXC_SWEEP" \
    --baseline_align $BASELINE_ALIGN --baseline_coh $BASELINE_COH \
    --out $DOCS_RESULTS/summary_sae_vs_txc.md \
    --title "SAE vs TXC coherence/suppression frontier (Qwen-2.5-7B bad-medical, L15)"

# 4. MLC training.
log "MLC training start"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_training_mlc \
    --config $TEMP_XC/experiments/em_features/config.yaml \
    --out "$MLC_CKPT"
log "MLC training done"

# 5. MLC Stage A.
log "MLC Stage A start"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_find_misalignment_features_mlc \
    --ckpt "$MLC_CKPT" \
    --diff_vectors "$DIFF" \
    --out $TEMP_XC/experiments/em_features/results/qwen_mlc
log "MLC Stage A done"

# 6. MLC frontier sweep.
log "MLC frontier sweep start"
cd $EM && python -m feature_ablation.frontier_sweep \
    --steerer mlc --model qwen --layer 15 \
    --features_json $TEMP_XC/experiments/em_features/results/qwen_mlc/top_200_features_mlc.json \
    --mlc_ckpt "$MLC_CKPT" \
    --k 10 \
    --alpha_grid "${ALPHA_GRID[@]}" \
    --n_rollouts 8 \
    --out_path $MLC_SWEEP
log "MLC frontier sweep done"

# 7. 3-way comparison plot + markdown summary.
log "3-way plot + summary"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.plot_frontier \
    --sweep "SAE=$SAE_SWEEP" \
    --sweep "TXC=$TXC_SWEEP" \
    --sweep "MLC=$MLC_SWEEP" \
    --out $FIG_DIR/frontier_sae_vs_txc_vs_mlc.png \
    --title "Coherence / suppression frontier: SAE vs TXC vs MLC (Qwen-2.5-7B bad-medical, L15, k=10)"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.write_frontier_summary \
    --sweep "SAE=$SAE_SWEEP" \
    --sweep "TXC=$TXC_SWEEP" \
    --sweep "MLC=$MLC_SWEEP" \
    --baseline_align $BASELINE_ALIGN --baseline_coh $BASELINE_COH \
    --out $DOCS_RESULTS/summary_sae_vs_txc_vs_mlc.md \
    --title "SAE vs TXC vs MLC coherence/suppression frontier (Qwen-2.5-7B bad-medical, L15)"
log "3-way done"

# 8. Auto-continue to the Arditi SAE phase (unless a stop sentinel exists).
if [ -e /root/em_features/.stop_arditi ]; then
    log "stop sentinel present — skipping Arditi phase"
else
    log "handing off to run_arditi_sae_phase.sh"
    bash $TEMP_XC/experiments/em_features/run_arditi_sae_phase.sh
fi

log "everything complete"
