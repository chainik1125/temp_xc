#!/usr/bin/env bash
# Replicate fra_proj's recreate_layer0/ deep dive in isolation:
# 3 TopK SAEs at blocks.0.{resid_pre, resid_mid, resid_post}, no T-SAE,
# no TXC, no H8, no MLC. Same hyperparams as fra_proj's recreate_layer0/
# config.yaml (d_sae=1536, k=32, n_steps=4000). Outputs land under
# outputs/recreate_layer0/{data,plots}/ so the v2 5-hookpoint frontier
# files in outputs/data/ aren't clobbered.
#
# Purpose: settle whether v2's sae_l0_mid=0.54 (vs fra_proj's 0.00) is a
# joint-training artefact (15 archs sharing batches) or a real
# discrepancy. If this isolated run reproduces fra_proj's 0.00, the v2
# gap is feature-ranking variance from the joint setup.
#
# Expected runtime on a single A40: ~10-15 min (3 SAEs × 4000 steps,
# much faster than v2's 15-arch run).

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs/recreate_layer0/data outputs/recreate_layer0/plots results
LOG=results/recreate_layer0.log
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
export PYTHONUNBUFFERED=1

HOOKS=(
  "blocks.0.hook_resid_pre"
  "blocks.0.hook_resid_mid"
  "blocks.0.hook_resid_post"
)
SAE_OVR=(
  "sae_layer0=blocks.0.hook_resid_pre"
  "sae_layer1=blocks.0.hook_resid_mid"
  "sae_layer2=blocks.0.hook_resid_post"
)
ARCHS=(sae_layer0 sae_layer1 sae_layer2)
OUTDIR="outputs/recreate_layer0/data"

run harvest \
  uv run python harvest_activations.py \
    --n_train 10000 --n_val 200 --n_test 200 \
    --seq_len 128 --device cuda --chunk_size 32 \
    --hook_names "${HOOKS[@]}" \
    --output_dir "$OUTDIR"

run train \
  uv run python train_crosscoders.py \
    --d_sae 1536 --k_total 32 --T 30 \
    --n_steps 4000 --batch_size 4096 \
    --lr 5e-4 --device cuda \
    --print_every 100 \
    --input_dir "$OUTDIR" --output_dir "$OUTDIR" \
    --sae_layer_hooks_override "${SAE_OVR[@]}" \
    --archs "${ARCHS[@]}"

run sweep \
  uv run python run_ablation_sweep.py \
    --top_k 100 --stage2_keep 10 \
    --alphas 0.25 0.5 1.0 1.5 2.0 \
    --delta_util 0.05 \
    --device cuda \
    --input_dir "$OUTDIR" --output_dir "$OUTDIR" \
    --archs "${ARCHS[@]}"

run plot \
  uv run python plot_pareto.py \
    --input_dir "$OUTDIR" \
    --output_dir outputs/recreate_layer0/plots \
    --results_md_dir outputs/recreate_layer0

echo "=== RECREATE_LAYER0 DONE ===" | tee -a "$LOG"
