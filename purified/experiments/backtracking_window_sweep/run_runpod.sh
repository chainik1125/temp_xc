#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly PYTHON="${TXC_RUNPOD_PYTHON:-$ROOT/.venv-e0-extract/bin/python}"
readonly MODE="${BACKTRACKING_RUN_MODE:-full}"
readonly PHASE="${BACKTRACKING_RUN_PHASE:-all}"
readonly DEVICE_ID="${BACKTRACKING_CUDA_DEVICE:-0}"
readonly ARTIFACT="${BACKTRACKING_ARTIFACT:-$ROOT/purified/artifacts/c7/sentence_acts_L10.npz}"
readonly CACHE="${BACKTRACKING_ACTIVATION_CACHE:-$ROOT/purified/artifacts/hf_temp_bench_data/act_cache/fb2a74be884e512a/resid_post_L10.npy}"
readonly RESULT_ROOT="${BACKTRACKING_RESULT_ROOT:-$ROOT/purified/results/neurips_rebuttal/backtracking_window_sweep/$MODE}"
readonly CHECKPOINT_ROOT="${BACKTRACKING_CHECKPOINT_ROOT:-$ROOT/checkpoints/backtracking_window_sweep/$MODE}"
readonly LOG_ROOT="${BACKTRACKING_LOG_ROOT:-$ROOT/purified/logs/backtracking_window_sweep}"
readonly LOG="${BACKTRACKING_LOG_FILE:-$LOG_ROOT/${MODE}_shard${BACKTRACKING_SHARD_INDEX:-0}.log}"
readonly EXIT_FILE="${BACKTRACKING_T_SWEEP_EXIT_FILE:-$ROOT/purified/logs/em_granularity/backtracking_t_sweep.exit}"

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "BACKTRACKING_RUN_MODE must be smoke or full" >&2
  exit 2
fi
if [[ "$PHASE" != "train" && "$PHASE" != "eval" && "$PHASE" != "all" ]]; then
  echo "BACKTRACKING_RUN_PHASE must be train, eval, or all" >&2
  exit 2
fi
if [[ ! "${BACKTRACKING_NUM_SHARDS:-1}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BACKTRACKING_NUM_SHARDS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${BACKTRACKING_SHARD_INDEX:-0}" =~ ^[0-9]+$ ]]; then
  echo "BACKTRACKING_SHARD_INDEX must be a non-negative integer" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$(dirname "$EXIT_FILE")"
rm -f "$EXIT_FILE"
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
  --activation-cache "$CACHE"
  --output-root "$RESULT_ROOT"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --device cuda:0
  --num-shards "${BACKTRACKING_NUM_SHARDS:-1}"
  --shard-index "${BACKTRACKING_SHARD_INDEX:-0}"
)
if [[ -n "${BACKTRACKING_WINDOWS:-}" ]]; then
  args+=(--windows "$BACKTRACKING_WINDOWS")
fi
if [[ -n "${BACKTRACKING_SEEDS:-}" ]]; then
  args+=(--seeds "$BACKTRACKING_SEEDS")
fi
if [[ -n "${BACKTRACKING_STEPS:-}" ]]; then
  args+=(--steps "$BACKTRACKING_STEPS")
fi
if [[ -n "${BACKTRACKING_BATCH_SIZE:-}" ]]; then
  args+=(--batch-size "$BACKTRACKING_BATCH_SIZE")
fi
if [[ -n "${BACKTRACKING_D_SAE:-}" ]]; then
  args+=(--d-sae "$BACKTRACKING_D_SAE")
fi
if [[ -n "${BACKTRACKING_K_POS:-}" ]]; then
  args+=(--k-pos "$BACKTRACKING_K_POS")
fi
if [[ -n "${BACKTRACKING_MAX_ROWS:-}" ]]; then
  args+=(--max-rows "$BACKTRACKING_MAX_ROWS")
fi
if [[ "${BACKTRACKING_CONTINUE_ON_ERROR:-0}" == "1" ]]; then
  args+=(--continue-on-error)
fi
if [[ "${BACKTRACKING_DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

"$PYTHON" -m experiments.backtracking_window_sweep.run "${args[@]}"
