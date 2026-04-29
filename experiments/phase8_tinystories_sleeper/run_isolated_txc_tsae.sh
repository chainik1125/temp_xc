#!/usr/bin/env bash
# Isolated TXC and T-SAE at the same 5 hookpoints as the v2 frontier
# table. Each (arch, hookpoint) pair gets its own training process so
# the (seq, pos) batch sampling and feature ranking aren't perturbed by
# co-training other archs. Settles whether v2's headline architecture
# flips are real or joint-training artifacts.
#
# Outputs to outputs/isolated/<tag>/data — separate subtree per pair.
#
# Expected runtime on a single A40: ~60-90 min total
# (5 hookpoints × 2 archs × ~3 min train + ~7 min sweep each).

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
LOG=results/isolated_txc_tsae.log
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

# (arch_family, hookpoint, tag) triples to train + sweep, one at a time.
# Hookpoints match the v2 frontier table.
PAIRS=(
  "tsae blocks.0.ln1.hook_normalized iso_tsae_l0_ln1"
  "tsae blocks.0.hook_resid_pre      iso_tsae_l0_pre"
  "tsae blocks.0.hook_resid_mid      iso_tsae_l0_mid"
  "tsae blocks.0.hook_resid_post     iso_tsae_l0_post"
  "tsae blocks.1.ln1.hook_normalized iso_tsae_l1_ln1"
  "txc  blocks.0.ln1.hook_normalized iso_txc_l0_ln1"
  "txc  blocks.0.hook_resid_pre      iso_txc_l0_pre"
  "txc  blocks.0.hook_resid_mid      iso_txc_l0_mid"
  "txc  blocks.0.hook_resid_post     iso_txc_l0_post"
  "txc  blocks.1.ln1.hook_normalized iso_txc_l1_ln1"
  # Fill the SAE gaps at the LN-normalised hookpoints so the isolated
  # frontier table has all 15 cells.
  "sae  blocks.0.ln1.hook_normalized iso_sae_l0_ln1"
  "sae  blocks.1.ln1.hook_normalized iso_sae_l1_ln1"
)

for entry in "${PAIRS[@]}"; do
  read -r FAMILY HOOK TAG <<< "$entry"
  echo "[isolated] === $TAG (family=$FAMILY hook=$HOOK) ==="
  OUTDIR="outputs/isolated/$TAG/data"
  PLOTDIR="outputs/isolated/$TAG/plots"
  mkdir -p "$OUTDIR" "$PLOTDIR"

  if [[ -f "$OUTDIR/activations_cache.pt" ]]; then
    echo "  [skip] $TAG.harvest (cache present)" | tee -a "$LOG"
  else
    run "$TAG.harvest" \
      uv run python harvest_activations.py \
        --n_train 10000 --n_val 200 --n_test 200 \
        --seq_len 128 --device cuda --chunk_size 32 \
        --hook_names "$HOOK" \
        --output_dir "$OUTDIR"
  fi

  case "$FAMILY" in
    tsae) OVR_FLAG="--tsae_layer_hooks_override" ;;
    txc)  OVR_FLAG="--txc_layer_hooks_override" ;;
    sae)  OVR_FLAG="--sae_layer_hooks_override" ;;
    *) echo "unknown family: $FAMILY" | tee -a "$LOG"; exit 1 ;;
  esac

  if [[ -f "$OUTDIR/crosscoder_$TAG.pt" ]]; then
    echo "  [skip] $TAG.train (checkpoint present)" | tee -a "$LOG"
  else
    run "$TAG.train" \
      uv run python train_crosscoders.py \
        --d_sae 1536 --k_total 32 --T 30 \
        --n_steps 4000 --batch_size 4096 \
        --lr 5e-4 --device cuda \
        --print_every 100 \
        --input_dir "$OUTDIR" --output_dir "$OUTDIR" \
        "$OVR_FLAG" "$TAG=$HOOK" \
        --archs "$TAG"
  fi

  if [[ -f "$OUTDIR/val_sweep_$TAG.json" ]]; then
    echo "  [skip] $TAG.sweep (already swept)" | tee -a "$LOG"
  else
    run "$TAG.sweep" \
      uv run python run_ablation_sweep.py \
        --top_k 100 --stage2_keep 10 \
        --alphas 0.25 0.5 1.0 1.5 2.0 \
        --delta_util 0.05 \
        --device cuda \
        --input_dir "$OUTDIR" --output_dir "$OUTDIR" \
        --archs "$TAG"
  fi
done

echo "=== ISOLATED_TXC_TSAE DONE ===" | tee -a "$LOG"
