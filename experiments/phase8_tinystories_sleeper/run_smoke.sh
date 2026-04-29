#!/usr/bin/env bash
# Smoke pipeline for phase 8 — tiny d_sae, few steps, just enough to verify
# the four-script pipeline runs end-to-end with the new T-SAE / H8-lite
# architectures. Expected runtime ≤ 5 minutes on an A40.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p results outputs/data outputs/plots
LOG=results/smoke.log
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

ARCHS=(sae_layer0 tsae_layer0 h8_early txc_early mlc)

run harvest \
  uv run python harvest_activations.py \
    --n_train 512 --n_val 64 --n_test 64 --device cuda

run train \
  uv run python train_crosscoders.py \
    --d_sae 128 --k_total 16 --T 10 --n_steps 200 \
    --batch_size 512 --device cuda \
    --archs "${ARCHS[@]}"

run sweep \
  uv run python run_ablation_sweep.py \
    --top_k 20 --stage2_keep 5 --alphas 0.5 1.0 2.0 \
    --device cuda \
    --archs "${ARCHS[@]}"

run plot \
  uv run python plot_pareto.py

echo "=== SMOKE DONE ===" | tee -a "$LOG"
