#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly WORKER="$ROOT/purified/experiments/backtracking_window_sweep/run_runpod.sh"
readonly MODE="${BACKTRACKING_RUN_MODE:-full}"
readonly PHASE="${BACKTRACKING_RUN_PHASE:-all}"
readonly SESSION_PREFIX="${TXC_TMUX_PREFIX:-backtracking-t-sweep-$MODE}"
readonly GPU_LIST="${BACKTRACKING_GPU_LIST:-0}"
readonly DRY_RUN="${BACKTRACKING_LAUNCH_DRY_RUN:-0}"
readonly RESULT_ROOT="${BACKTRACKING_RESULT_ROOT:-$ROOT/purified/results/neurips_rebuttal/backtracking_window_sweep/$MODE}"
readonly CHECKPOINT_ROOT="${BACKTRACKING_CHECKPOINT_ROOT:-$ROOT/checkpoints/backtracking_window_sweep/$MODE}"

IFS=',' read -r -a gpus <<<"$GPU_LIST"
readonly N_SHARDS="${#gpus[@]}"
if ((N_SHARDS < 1)); then
  echo "BACKTRACKING_GPU_LIST must contain at least one GPU id" >&2
  exit 2
fi
if [[ ! -f "$WORKER" ]]; then
  echo "missing worker: $WORKER" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "BACKTRACKING_LAUNCH_DRY_RUN must be 0 or 1" >&2
  exit 2
fi

for shard in "${!gpus[@]}"; do
  session="$SESSION_PREFIX"
  if ((N_SHARDS > 1)); then
    session="${SESSION_PREFIX}-${shard}"
  fi
  exit_file="$ROOT/purified/logs/backtracking_window_sweep/${session}.exit"
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
    "BACKTRACKING_RUN_MODE=$MODE" \
    "BACKTRACKING_RUN_PHASE=$PHASE" \
    "BACKTRACKING_RESULT_ROOT=$RESULT_ROOT" \
    "BACKTRACKING_CHECKPOINT_ROOT=$CHECKPOINT_ROOT" \
    "BACKTRACKING_CUDA_DEVICE=${gpus[$shard]}" \
    "BACKTRACKING_NUM_SHARDS=$N_SHARDS" \
    "BACKTRACKING_SHARD_INDEX=$shard" \
    "BACKTRACKING_T_SWEEP_EXIT_FILE=$exit_file" \
    bash "$WORKER"
  printf "launched %s on GPU %s; exit file %s\n" \
    "$session" "${gpus[$shard]}" "$exit_file"
done
