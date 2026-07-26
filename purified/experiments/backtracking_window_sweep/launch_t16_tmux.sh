#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly WORKER="$ROOT/purified/experiments/backtracking_window_sweep/run_t16_runpod.sh"
readonly MODE="${BACKTRACKING_T16_MODE:-full}"
readonly GPU_LIST="${BACKTRACKING_T16_GPU_LIST:-0}"
readonly SESSION_PREFIX="${TXC_T16_TMUX_PREFIX:-backtracking-t16-$MODE}"
readonly DRY_RUN="${BACKTRACKING_T16_LAUNCH_DRY_RUN:-0}"
readonly PYTHON="${TXC_RUNPOD_PYTHON:-$ROOT/.venv-e0-extract/bin/python}"
readonly PHASE="${BACKTRACKING_T16_PHASE:-all}"
readonly C7="$ROOT/purified/artifacts/c7"
readonly ARTIFACT="${BACKTRACKING_T16_ARTIFACT:-$C7/sentence_acts_L10_T16.npz}"
readonly MANIFEST="${BACKTRACKING_T16_MANIFEST:-$C7/sentence_acts_L10_T16.manifest.json}"
readonly REFERENCE="${BACKTRACKING_T16_REFERENCE:-$C7/sentence_acts_L10.npz}"
readonly CACHE="${BACKTRACKING_ACTIVATION_CACHE:-$ROOT/purified/artifacts/hf_temp_bench_data/act_cache/fb2a74be884e512a/resid_post_L10.npy}"
readonly RESULT_ROOT="${BACKTRACKING_T16_RESULT_ROOT:-$ROOT/purified/results/neurips_rebuttal/backtracking_window_sweep_t16/$MODE}"
readonly CHECKPOINT_ROOT="${BACKTRACKING_T16_CHECKPOINT_ROOT:-$ROOT/checkpoints/backtracking_window_sweep_t16/$MODE}"
readonly LOG_ROOT="${BACKTRACKING_T16_LOG_ROOT:-$ROOT/purified/logs/backtracking_window_sweep_t16}"

if [[ "$(git -C "$ROOT" branch --show-current)" != "neurips-aniket" ]]; then
  echo "refusing to launch: current branch must be neurips-aniket" >&2
  exit 2
fi
IFS=',' read -r -a gpus <<<"$GPU_LIST"
readonly N_SHARDS="${#gpus[@]}"
if ((N_SHARDS < 1)); then
  echo "BACKTRACKING_T16_GPU_LIST must contain at least one GPU id" >&2
  exit 2
fi
if [[ ! -f "$WORKER" ]]; then
  echo "missing worker: $WORKER" >&2
  exit 1
fi

for shard in "${!gpus[@]}"; do
  session="${SESSION_PREFIX}-${shard}"
  exit_file="$LOG_ROOT/${session}.exit"
  if ((DRY_RUN == 1)); then
    printf "would launch tmux session %s on GPU %s (shard %s/%s)\n" \
      "$session" "${gpus[$shard]}" "$shard" "$N_SHARDS"
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session already exists: $session" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" env \
    "TXC_RUNPOD_ROOT=$ROOT" \
    "TXC_RUNPOD_PYTHON=$PYTHON" \
    "BACKTRACKING_T16_MODE=$MODE" \
    "BACKTRACKING_T16_PHASE=$PHASE" \
    "BACKTRACKING_T16_ARTIFACT=$ARTIFACT" \
    "BACKTRACKING_T16_MANIFEST=$MANIFEST" \
    "BACKTRACKING_T16_REFERENCE=$REFERENCE" \
    "BACKTRACKING_ACTIVATION_CACHE=$CACHE" \
    "BACKTRACKING_T16_RESULT_ROOT=$RESULT_ROOT" \
    "BACKTRACKING_T16_CHECKPOINT_ROOT=$CHECKPOINT_ROOT" \
    "BACKTRACKING_T16_LOG_ROOT=$LOG_ROOT" \
    "BACKTRACKING_T16_WINDOWS=${BACKTRACKING_T16_WINDOWS:-}" \
    "BACKTRACKING_T16_SEEDS=${BACKTRACKING_T16_SEEDS:-}" \
    "BACKTRACKING_T16_STEPS=${BACKTRACKING_T16_STEPS:-}" \
    "BACKTRACKING_T16_BATCH_SIZE=${BACKTRACKING_T16_BATCH_SIZE:-}" \
    "BACKTRACKING_T16_D_SAE=${BACKTRACKING_T16_D_SAE:-}" \
    "BACKTRACKING_T16_K_POS=${BACKTRACKING_T16_K_POS:-}" \
    "BACKTRACKING_T16_MAX_ROWS=${BACKTRACKING_T16_MAX_ROWS:-}" \
    "BACKTRACKING_T16_CONTINUE_ON_ERROR=${BACKTRACKING_T16_CONTINUE_ON_ERROR:-0}" \
    "BACKTRACKING_T16_DRY_RUN=${BACKTRACKING_T16_DRY_RUN:-0}" \
    "BACKTRACKING_T16_CUDA_DEVICE=${gpus[$shard]}" \
    "BACKTRACKING_T16_NUM_SHARDS=$N_SHARDS" \
    "BACKTRACKING_T16_SHARD_INDEX=$shard" \
    "BACKTRACKING_T16_EXIT_FILE=$exit_file" \
    bash "$WORKER"
  printf "launched %s on GPU %s; exit file %s\n" \
    "$session" "${gpus[$shard]}" "$exit_file"
done
