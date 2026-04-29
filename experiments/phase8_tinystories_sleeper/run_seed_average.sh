#!/usr/bin/env bash
# Three-seed (0, 1, 2) isolated re-run of every (arch, hookpoint) cell
# from the Phase 8 frontier. Per-seed checkpoints and test_results are
# saved separately so we can compute mean ± std on test ASR per cell.
#
# Tag scheme: `<basetag>_s<seed>`  (e.g. `sae_l0_ln1_s0`). Suffix keeps
# the prefix-based per-token-vs-window dispatch intact.
#
# All 45 train+sweep runs share a single 5-hookpoint harvest at
# outputs/seeded/data/ to avoid 45× re-harvesting the same TinyStories
# activations.
#
# Skip-existing logic: if val_sweep_<tag>.json already exists for a
# given (basetag, seed), that pair is skipped on a re-run. So a partial
# resumption is safe.
#
# Estimated runtime on a single A40: ~6-7 hours
# (one ~5min shared harvest + 45 × ~8min train+sweep).

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results outputs/seeded/data
LOG=results/seed_average.log
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

CACHEDIR="outputs/seeded/data"

# Single shared 5-hookpoint harvest at the head of the run. All 45
# train+sweep pairs read activations from CACHEDIR.
HOOKS=(
  "blocks.0.ln1.hook_normalized"
  "blocks.0.hook_resid_pre"
  "blocks.0.hook_resid_mid"
  "blocks.0.hook_resid_post"
  "blocks.1.ln1.hook_normalized"
)

if [[ -f "$CACHEDIR/activations_cache.pt" ]]; then
  echo "[skip] shared harvest (cache present at $CACHEDIR)" | tee -a "$LOG"
else
  run shared.harvest \
    uv run python harvest_activations.py \
      --n_train 10000 --n_val 200 --n_test 200 \
      --seq_len 128 --device cuda --chunk_size 32 \
      --hook_names "${HOOKS[@]}" \
      --output_dir "$CACHEDIR"
fi

# 15 (family, hookpoint, basetag) triples — same set as the isolated
# frontier table. Each is run for seeds 0, 1, 2.
PAIRS=(
  "sae  blocks.0.ln1.hook_normalized sae_l0_ln1"
  "sae  blocks.0.hook_resid_pre      sae_l0_pre"
  "sae  blocks.0.hook_resid_mid      sae_l0_mid"
  "sae  blocks.0.hook_resid_post     sae_l0_post"
  "sae  blocks.1.ln1.hook_normalized sae_l1_ln1"
  "tsae blocks.0.ln1.hook_normalized tsae_l0_ln1"
  "tsae blocks.0.hook_resid_pre      tsae_l0_pre"
  "tsae blocks.0.hook_resid_mid      tsae_l0_mid"
  "tsae blocks.0.hook_resid_post     tsae_l0_post"
  "tsae blocks.1.ln1.hook_normalized tsae_l1_ln1"
  "txc  blocks.0.ln1.hook_normalized txc_l0_ln1"
  "txc  blocks.0.hook_resid_pre      txc_l0_pre"
  "txc  blocks.0.hook_resid_mid      txc_l0_mid"
  "txc  blocks.0.hook_resid_post     txc_l0_post"
  "txc  blocks.1.ln1.hook_normalized txc_l1_ln1"
)

# SEEDS can be overridden via env var to split work across hosts.
# Default: all three (0 1 2).
SEEDS="${SEEDS:-0 1 2}"
echo "[seeded] running seeds: $SEEDS" | tee -a "$LOG"

for SEED in $SEEDS; do
  for entry in "${PAIRS[@]}"; do
    read -r FAMILY HOOK BASETAG <<< "$entry"
    TAG="${BASETAG}_s${SEED}"
    echo "[seeded] === $TAG (family=$FAMILY hook=$HOOK seed=$SEED) ==="

    if [[ -f "$CACHEDIR/val_sweep_${TAG}.json" ]]; then
      echo "  [skip] $TAG (val_sweep already present)" | tee -a "$LOG"
      continue
    fi

    case "$FAMILY" in
      tsae) OVR_FLAG="--tsae_layer_hooks_override" ;;
      txc)  OVR_FLAG="--txc_layer_hooks_override" ;;
      sae)  OVR_FLAG="--sae_layer_hooks_override" ;;
      *) echo "unknown family: $FAMILY" | tee -a "$LOG"; exit 1 ;;
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

echo "=== SEED_AVERAGE DONE ===" | tee -a "$LOG"
