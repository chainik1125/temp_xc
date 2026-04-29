#!/usr/bin/env bash
# Per-hookpoint version of run_seed_average.sh — for hosts where the
# full 5-hookpoint activations cache (~10 GB) doesn't fit. Harvests
# one hookpoint at a time into a per-hookpoint dir, runs all archs ×
# seeds for that hookpoint, then deletes the cache before moving to
# the next hookpoint.
#
# Tag scheme is identical (`<basetag>_s<seed>`) so test_results files
# from this runner are compatible with the main runner's outputs.
#
# SEEDS env var splits work across hosts (e.g. SEEDS="2" runs only seed 2).
#
# Estimated runtime per pair: ~10 min (2min harvest + 2min train + 7min
# sweep). Single seed × 15 cells = ~2.5 hours.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results outputs/seeded
LOG=results/seed_average_lowdisk.log
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

SEEDS="${SEEDS:-0 1 2}"
echo "[seeded-lowdisk] running seeds: $SEEDS" | tee -a "$LOG"

# Group archs by hookpoint so we cache once per hookpoint, then run all
# (arch × seed) combinations against that cache.
HOOKPOINTS=(
  "blocks.0.ln1.hook_normalized   l0_ln1"
  "blocks.0.hook_resid_pre        l0_pre"
  "blocks.0.hook_resid_mid        l0_mid"
  "blocks.0.hook_resid_post       l0_post"
  "blocks.1.ln1.hook_normalized   l1_ln1"
)

# Three arch families per hookpoint.
FAMILIES=(sae tsae txc)

for entry in "${HOOKPOINTS[@]}"; do
  read -r HOOK HOOKKEY <<< "$entry"
  CACHEDIR="outputs/seeded/per_hook_${HOOKKEY}/data"
  mkdir -p "$CACHEDIR"

  echo "[seeded-lowdisk] === hookpoint $HOOKKEY ($HOOK) ===" | tee -a "$LOG"

  # Harvest just this one hookpoint.
  if [[ -f "$CACHEDIR/activations_cache.pt" ]]; then
    echo "  [skip] $HOOKKEY.harvest (cache present)" | tee -a "$LOG"
  else
    run "$HOOKKEY.harvest" \
      uv run python harvest_activations.py \
        --n_train 10000 --n_val 200 --n_test 200 \
        --seq_len 128 --device cuda --chunk_size 32 \
        --hook_names "$HOOK" \
        --output_dir "$CACHEDIR"
  fi

  # All (family, seed) combinations at this hookpoint.
  for FAMILY in "${FAMILIES[@]}"; do
    BASETAG="${FAMILY}_${HOOKKEY}"
    for SEED in $SEEDS; do
      TAG="${BASETAG}_s${SEED}"
      echo "[seeded-lowdisk] === $TAG (seed=$SEED) ===" | tee -a "$LOG"

      if [[ -f "$CACHEDIR/val_sweep_${TAG}.json" ]]; then
        echo "  [skip] $TAG (val_sweep already present)" | tee -a "$LOG"
        continue
      fi

      case "$FAMILY" in
        tsae) OVR_FLAG="--tsae_layer_hooks_override" ;;
        txc)  OVR_FLAG="--txc_layer_hooks_override" ;;
        sae)  OVR_FLAG="--sae_layer_hooks_override" ;;
      esac

      if [[ -f "$CACHEDIR/crosscoder_${TAG}.pt" ]]; then
        echo "  [skip] $TAG.train (checkpoint present)" | tee -a "$LOG"
      else
        run "$TAG.train" \
          uv run python train_crosscoders.py \
            --d_sae 1536 --k_total 32 --T 30 \
            --n_steps 4000 --batch_size 4096 \
            --lr 5e-4 --device cuda \
            --print_every 500 \
            --seed "$SEED" \
            --input_dir "$CACHEDIR" --output_dir "$CACHEDIR" \
            "$OVR_FLAG" "$TAG=$HOOK" \
            --archs "$TAG"
      fi

      run "$TAG.sweep" \
        uv run python run_ablation_sweep.py \
          --top_k 100 --stage2_keep 10 \
          --alphas 0.25 0.5 1.0 1.5 2.0 \
          --delta_util 0.05 \
          --device cuda \
          --input_dir "$CACHEDIR" --output_dir "$CACHEDIR" \
          --archs "$TAG"
    done
  done

  # Free disk for the next hookpoint. Keep the small JSONs and
  # checkpoints; just delete the big activations + tokens caches.
  echo "[seeded-lowdisk] freeing $HOOKKEY caches (~2 GB)" | tee -a "$LOG"
  rm -f "$CACHEDIR/activations_cache.pt" "$CACHEDIR/tokens_cache.pt"
done

echo "=== SEED_AVERAGE_LOWDISK DONE ===" | tee -a "$LOG"
