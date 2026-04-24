#!/usr/bin/env bash
# TXC ranking-and-steering-protocol comparison at the 200k small-TXC ckpt.
#
# Waits for the deepdive's 200k sweep JSON (last_cos / standard steering) to
# land AND for the GPU to be free. Then runs three alternative configs:
#   (1) sum_cos ranking + last-slot steering
#   (2) encoder ranking (per-token diffs → TXC encoder) + last-slot steering
#   (3) last_cos ranking + window steering (hook writes to last T residual positions)
# Plots all four frontiers against SAE + baseline.
set -euo pipefail

TEMP_XC=/root/temp_xc
EM=/root/em_features
RESULTS=$EM/results
FIG_DIR=$RESULTS/figures
DOCS_RESULTS=$TEMP_XC/docs/dmitry/results/em_features
CKPT=$TEMP_XC/experiments/em_features/checkpoints/qwen_l15_txc_small_step200000.pt
DIFF=$RESULTS/qwen_l15_sae/01_differences/difference_vectors.pt
PER_TOKEN_DIFFS=$RESULTS/qwen_l15_per_token_diffs_T5.pt
SAE_SWEEP=$RESULTS/qwen_l15_sae_bundled_frontier_extended.json
BASELINE_TXC_SWEEP=$RESULTS/qwen_l15_txc_small_step200000_frontier.json
ALPHA_GRID=(-10 -8 -6 -5 -4 -3 -2 -1.5 -1.0 1.0 2.0 5.0)
BASELINE_ALIGN=64.19
BASELINE_COH=84.88
LAYER=15

mkdir -p "$FIG_DIR" "$DOCS_RESULTS"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

wait_for() {
    local path=$1
    log "waiting for $path"
    while ! [ -s "$path" ]; do sleep 60; done
    log "found $path"
}

wait_gpu_free() {
    # Block until nothing is eating GPU memory. deepdive sweeps release GPU
    # between sweeps but big-TXC training would not — the sentinel handles
    # that case by having big-TXC skipped entirely.
    log "waiting for GPU to be free"
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [ "${used:-0}" -lt 2000 ]; then
            log "GPU free (${used} MiB)"; return
        fi
        sleep 30
    done
}

# 0. Wait for deepdive small-TXC sweep phase to finish.
wait_for "$BASELINE_TXC_SWEEP"
wait_gpu_free

# 1. Compute per-token diffs for the encoder-ranking path (one-time, ~5 min).
if [ ! -s "$PER_TOKEN_DIFFS" ]; then
    log "computing per-token diffs"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.compute_per_token_diffs \
        --dataset $EM/data/medical_advice_prompt_only.jsonl \
        --base Qwen/Qwen2.5-7B-Instruct \
        --bad andyrdt/Qwen2.5-7B-Instruct_bad-medical \
        --layer $LAYER --T 5 \
        --n_prompts 256 --max_ctx_len 512 --batch_size 4 \
        --out "$PER_TOKEN_DIFFS"
fi

# 2. sum_cos ranking + last-slot steering.
SUMCOS_DIR=$TEMP_XC/experiments/em_features/results/qwen_l15_txc_small_step200000_sumcos
SUMCOS_JSON=$RESULTS/qwen_l15_txc_small_step200000_sumcos_frontier.json
if [ ! -s "$SUMCOS_JSON" ]; then
    log "Stage A sum_cos"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_find_misalignment_features \
        --ckpt "$CKPT" --diff_vectors "$DIFF" --layer $LAYER --out "$SUMCOS_DIR" \
        --ranking sum_cos
    log "Sweep sum_cos"
    cd $EM && python -m feature_ablation.frontier_sweep \
        --steerer txc --model qwen --layer $LAYER \
        --features_json $SUMCOS_DIR/top_200_features_layer_${LAYER}.json \
        --txc_ckpt "$CKPT" \
        --k 10 --alpha_grid "${ALPHA_GRID[@]}" --n_rollouts 8 \
        --out_path "$SUMCOS_JSON"
fi

# 3. Encoder ranking + last-slot steering.
ENC_DIR=$TEMP_XC/experiments/em_features/results/qwen_l15_txc_small_step200000_encoder
ENC_JSON=$RESULTS/qwen_l15_txc_small_step200000_encoder_frontier.json
if [ ! -s "$ENC_JSON" ]; then
    log "Stage A encoder"
    cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.run_find_misalignment_features \
        --ckpt "$CKPT" --diff_vectors "$DIFF" --layer $LAYER --out "$ENC_DIR" \
        --ranking encoder --per_token_windows "$PER_TOKEN_DIFFS"
    log "Sweep encoder"
    cd $EM && python -m feature_ablation.frontier_sweep \
        --steerer txc --model qwen --layer $LAYER \
        --features_json $ENC_DIR/top_200_features_layer_${LAYER}.json \
        --txc_ckpt "$CKPT" \
        --k 10 --alpha_grid "${ALPHA_GRID[@]}" --n_rollouts 8 \
        --out_path "$ENC_JSON"
fi

# 4. last_cos ranking + window steering (uses the baseline top-200 JSON).
WINDOW_JSON=$RESULTS/qwen_l15_txc_small_step200000_window_frontier.json
BASE_FEATURES_JSON=$TEMP_XC/experiments/em_features/results/qwen_l15_txc_small_step200000/top_200_features_layer_${LAYER}.json
if [ ! -s "$WINDOW_JSON" ]; then
    log "Sweep window-steering (last_cos features)"
    cd $EM && python -m feature_ablation.frontier_sweep \
        --steerer txc_window --model qwen --layer $LAYER \
        --features_json "$BASE_FEATURES_JSON" \
        --txc_ckpt "$CKPT" \
        --k 10 --alpha_grid "${ALPHA_GRID[@]}" --n_rollouts 8 \
        --out_path "$WINDOW_JSON"
fi

# 5. Comparison plot + summary.
log "plot + summary across ranking/steering variants"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.plot_frontier \
    --sweep "SAE=$SAE_SWEEP" \
    --sweep "TXC_last_cos=$BASELINE_TXC_SWEEP" \
    --sweep "TXC_sum_cos=$SUMCOS_JSON" \
    --sweep "TXC_encoder=$ENC_JSON" \
    --sweep "TXC_window=$WINDOW_JSON" \
    --out $FIG_DIR/frontier_txc_ranking_comparison.png \
    --title "TXC 200k: ranking (last/sum/encoder) × steering (last/window) vs SAE"
cd $TEMP_XC && PYTHONPATH=$TEMP_XC python -m experiments.em_features.write_frontier_summary \
    --sweep "SAE=$SAE_SWEEP" \
    --sweep "TXC_last_cos=$BASELINE_TXC_SWEEP" \
    --sweep "TXC_sum_cos=$SUMCOS_JSON" \
    --sweep "TXC_encoder=$ENC_JSON" \
    --sweep "TXC_window=$WINDOW_JSON" \
    --baseline_align $BASELINE_ALIGN --baseline_coh $BASELINE_COH \
    --out $DOCS_RESULTS/summary_txc_ranking_comparison.md \
    --title "TXC 200k ranking and steering-protocol comparison"

log "ranking comparison complete"
