#!/usr/bin/env bash
# Phase 8 v2 (focused) — SAE / T-SAE / TXC at 5 specific hookpoints:
#
#   1. blocks.0.ln1.hook_normalized   tag suffix _l0_ln1
#   2. blocks.0.hook_resid_pre        tag suffix _l0_pre
#   3. blocks.0.hook_resid_mid        tag suffix _l0_mid
#   4. blocks.0.hook_resid_post       tag suffix _l0_post
#   5. blocks.1.ln1.hook_normalized   tag suffix _l1_ln1
#
# fra_proj's `recreate_layer0/` and `recreate_ln1/` configs use the same
# convention: arbitrary `sae_layer{i}` tags overridden onto non-default
# hookpoints. We extend that to T-SAE and TXC.
#
# Drops MLC and H8 (per user direction); n_steps=4000 (per fra_proj's
# real config — the 8000 in the README is outdated).
#
# Expected runtime on a single A40: ~30-60 min (3 arch types × 5 hooks ×
# 4000 steps × B=4096), much faster than the v1 H8-heavy run.

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
export PYTHONUNBUFFERED=1  # flush print() to full.log in realtime

# 5-hookpoint cache — matches `--hook_names` so train can index into the L dim.
HOOKS=(
  "blocks.0.ln1.hook_normalized"
  "blocks.0.hook_resid_pre"
  "blocks.0.hook_resid_mid"
  "blocks.0.hook_resid_post"
  "blocks.1.ln1.hook_normalized"
)

# Override tags → hooks for each arch family. Same 5 hookpoints, 3 arch
# families × 5 = 15 total instances. fra_proj's recreate_layer0 uses the
# same `sae_layer{i}=hookname` override pattern.
SAE_OVR=(
  "sae_l0_ln1=blocks.0.ln1.hook_normalized"
  "sae_l0_pre=blocks.0.hook_resid_pre"
  "sae_l0_mid=blocks.0.hook_resid_mid"
  "sae_l0_post=blocks.0.hook_resid_post"
  "sae_l1_ln1=blocks.1.ln1.hook_normalized"
)
TSAE_OVR=(
  "tsae_l0_ln1=blocks.0.ln1.hook_normalized"
  "tsae_l0_pre=blocks.0.hook_resid_pre"
  "tsae_l0_mid=blocks.0.hook_resid_mid"
  "tsae_l0_post=blocks.0.hook_resid_post"
  "tsae_l1_ln1=blocks.1.ln1.hook_normalized"
)
TXC_OVR=(
  "txc_l0_ln1=blocks.0.ln1.hook_normalized"
  "txc_l0_pre=blocks.0.hook_resid_pre"
  "txc_l0_mid=blocks.0.hook_resid_mid"
  "txc_l0_post=blocks.0.hook_resid_post"
  "txc_l1_ln1=blocks.1.ln1.hook_normalized"
)

# Architectures to actually train. We list every override tag explicitly.
# Default tags (sae_layer{0..3}, tsae_layer{0..3}, txc_{early,mid,late})
# would map to hooks not in our cache and get filtered out automatically.
ARCHS=(
  sae_l0_ln1 sae_l0_pre sae_l0_mid sae_l0_post sae_l1_ln1
  tsae_l0_ln1 tsae_l0_pre tsae_l0_mid tsae_l0_post tsae_l1_ln1
  txc_l0_ln1 txc_l0_pre txc_l0_mid txc_l0_post txc_l1_ln1
)

run harvest \
  uv run python harvest_activations.py \
    --n_train 10000 --n_val 200 --n_test 200 \
    --seq_len 128 --device cuda --chunk_size 32 \
    --hook_names "${HOOKS[@]}"

run train \
  uv run python train_crosscoders.py \
    --d_sae 1536 --k_total 32 --T 30 \
    --n_steps 4000 --batch_size 4096 \
    --lr 5e-4 --device cuda \
    --print_every 100 \
    --sae_layer_hooks_override "${SAE_OVR[@]}" \
    --tsae_layer_hooks_override "${TSAE_OVR[@]}" \
    --txc_layer_hooks_override "${TXC_OVR[@]}" \
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
