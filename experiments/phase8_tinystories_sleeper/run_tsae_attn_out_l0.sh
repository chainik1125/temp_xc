#!/usr/bin/env bash
# Single isolated T-SAE at blocks.0.hook_attn_out — the residual delta
# contributed by block-0 attention (resid_mid = resid_pre + attn_out).
# Motivated by:
# - v2 finding that T-SAE beats SAE at blocks.0.ln1.hook_normalized
#   (the post-LN input to attention) — the contrastive loss seems to
#   help when activations are already normalised.
# - fra_proj's recreate_layer0 finding that the trigger feature is
#   "constructed by block-0 attention" — attn_out is exactly that delta.
#
# This run trains T-SAE in isolation (no SAE/TXC/H8/MLC sharing the
# batch loop), so the feature ranking isn't perturbed by joint training
# variance. Outputs land under outputs/tsae_attn_out_l0/{data,plots}/.
#
# Expected runtime on a single A40: ~10-15 min.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs/tsae_attn_out_l0/data outputs/tsae_attn_out_l0/plots results
LOG=results/tsae_attn_out_l0.log
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

HOOKS=("blocks.0.hook_attn_out")
TSAE_OVR=("tsae_attn_out_l0=blocks.0.hook_attn_out")
ARCHS=(tsae_attn_out_l0)
OUTDIR="outputs/tsae_attn_out_l0/data"

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
    --tsae_layer_hooks_override "${TSAE_OVR[@]}" \
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
    --output_dir outputs/tsae_attn_out_l0/plots \
    --results_md_dir outputs/tsae_attn_out_l0

echo "=== TSAE_ATTN_OUT_L0 DONE ===" | tee -a "$LOG"
