#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly PYTHON="${TXC_RUNPOD_PYTHON:-$ROOT/.venv-e0-extract/bin/python}"
readonly MODE="${BACKTRACKING_T16_MODE:-full}"
readonly PHASE="${BACKTRACKING_T16_PHASE:-all}"
readonly DEVICE_ID="${BACKTRACKING_T16_CUDA_DEVICE:-0}"
readonly C7="$ROOT/purified/artifacts/c7"
readonly ARTIFACT="${BACKTRACKING_T16_ARTIFACT:-$C7/sentence_acts_L10_T16.npz}"
readonly MANIFEST="${BACKTRACKING_T16_MANIFEST:-$C7/sentence_acts_L10_T16.manifest.json}"
readonly REFERENCE="${BACKTRACKING_T16_REFERENCE:-$C7/sentence_acts_L10.npz}"
readonly CACHE="${BACKTRACKING_ACTIVATION_CACHE:-$ROOT/purified/artifacts/hf_temp_bench_data/act_cache/fb2a74be884e512a/resid_post_L10.npy}"
readonly RESULT_ROOT="${BACKTRACKING_T16_RESULT_ROOT:-$ROOT/purified/results/neurips_rebuttal/backtracking_window_sweep_t16/$MODE}"
readonly CHECKPOINT_ROOT="${BACKTRACKING_T16_CHECKPOINT_ROOT:-$ROOT/checkpoints/backtracking_window_sweep_t16/$MODE}"
readonly LOG_ROOT="${BACKTRACKING_T16_LOG_ROOT:-$ROOT/purified/logs/backtracking_window_sweep_t16}"
readonly LOG="${BACKTRACKING_T16_LOG_FILE:-$LOG_ROOT/${MODE}_shard${BACKTRACKING_T16_SHARD_INDEX:-0}.log}"
readonly EXIT_FILE="${BACKTRACKING_T16_EXIT_FILE:-$LOG_ROOT/${MODE}_shard${BACKTRACKING_T16_SHARD_INDEX:-0}.exit}"

if [[ "$MODE" != "smoke" && "$MODE" != "memory-smoke" && "$MODE" != "full" ]]; then
  echo "BACKTRACKING_T16_MODE must be smoke, memory-smoke, or full" >&2
  exit 2
fi
if [[ "$PHASE" != "train" && "$PHASE" != "eval" && "$PHASE" != "all" ]]; then
  echo "BACKTRACKING_T16_PHASE must be train, eval, or all" >&2
  exit 2
fi
if [[ "$(git -C "$ROOT" branch --show-current)" != "neurips-aniket" ]]; then
  echo "refusing to run: current branch must be neurips-aniket" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG") 2>&1
write_exit() {
  local status=$?
  printf "%s\n" "$status" >"$EXIT_FILE"
}
trap write_exit EXIT

export CUDA_VISIBLE_DEVICES="$DEVICE_ID"
export PYTHONPATH="$ROOT/purified/src:$ROOT/purified"
args=(
  --mode "$MODE"
  --phase "$PHASE"
  --artifact "$ARTIFACT"
  --artifact-manifest "$MANIFEST"
  --reference-artifact "$REFERENCE"
  --activation-cache "$CACHE"
  --output-root "$RESULT_ROOT"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --device cuda:0
  --num-shards "${BACKTRACKING_T16_NUM_SHARDS:-1}"
  --shard-index "${BACKTRACKING_T16_SHARD_INDEX:-0}"
)
if [[ -n "${BACKTRACKING_T16_WINDOWS:-}" ]]; then
  args+=(--windows "$BACKTRACKING_T16_WINDOWS")
fi
if [[ -n "${BACKTRACKING_T16_SEEDS:-}" ]]; then
  args+=(--seeds "$BACKTRACKING_T16_SEEDS")
fi
if [[ -n "${BACKTRACKING_T16_STEPS:-}" ]]; then
  args+=(--steps "$BACKTRACKING_T16_STEPS")
fi
if [[ -n "${BACKTRACKING_T16_BATCH_SIZE:-}" ]]; then
  args+=(--batch-size "$BACKTRACKING_T16_BATCH_SIZE")
fi
if [[ -n "${BACKTRACKING_T16_D_SAE:-}" ]]; then
  args+=(--d-sae "$BACKTRACKING_T16_D_SAE")
fi
if [[ -n "${BACKTRACKING_T16_K_POS:-}" ]]; then
  args+=(--k-pos "$BACKTRACKING_T16_K_POS")
fi
if [[ -n "${BACKTRACKING_T16_MAX_ROWS:-}" ]]; then
  args+=(--max-rows "$BACKTRACKING_T16_MAX_ROWS")
fi
if [[ "${BACKTRACKING_T16_CONTINUE_ON_ERROR:-0}" == "1" ]]; then
  args+=(--continue-on-error)
fi
if [[ "${BACKTRACKING_T16_DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

"$PYTHON" -m experiments.backtracking_window_sweep.run_t16 "${args[@]}"
