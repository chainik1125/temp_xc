#!/usr/bin/env bash
# Full Phase 8 pipeline: matches fra_proj defaults across all 15 archs.
# Expected runtime on a single A40: ~3-4 hours end-to-end.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results outputs/data outputs/plots
LOG=results/full.log
: > "$LOG"

run() {
  local label="$1"; shift
  echo "=== $label ===" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "  $label OK" | tee -a "$LOG"
  else
    echo "  $label FAILED (exit $?)" | tee -a "$LOG"
    exit 1
  fi
}

export TQDM_DISABLE=1
export PATH="$HOME/.local/bin:$PATH"

# All 15 archs. fra_proj's 8 (mlc + 3 TXC + 4 SAE) plus the Phase 8 additions
# (4 T-SAE + 3 H8). Layer indexing matches MLC_HOOK_NAMES (0-indexed in
# layer_name_to_idx).
ARCHS=(
  mlc
  txc_early txc_mid txc_late
  sae_layer0 sae_layer1 sae_layer2 sae_layer3
  tsae_layer0 tsae_layer1 tsae_layer2 tsae_layer3
  h8_early h8_mid h8_late
)

run harvest \
  uv run python harvest_activations.py \
    --n_train 10000 --n_val 200 --n_test 200 \
    --seq_len 128 --device cuda --chunk_size 16

run train \
  uv run python train_crosscoders.py \
    --d_sae 1536 --k_total 32 --T 30 \
    --n_steps 8000 --batch_size 4096 \
    --lr 5e-4 --device cuda \
    --archs "${ARCHS[@]}"

run sweep \
  uv run python run_ablation_sweep.py \
    --top_k 100 --stage2_keep 10 \
    --alphas 0.25 0.5 1.0 1.5 2.0 \
    --delta_util 0.05 \
    --device cuda \
    --archs "${ARCHS[@]}"

run plot \
  uv run python plot_pareto.py

echo "=== FULL PIPELINE DONE ===" | tee -a "$LOG"
