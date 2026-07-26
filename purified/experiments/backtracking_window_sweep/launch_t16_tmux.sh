#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly WORKER="$ROOT/purified/experiments/backtracking_window_sweep/run_t16_runpod.sh"
readonly MODE="${BACKTRACKING_T16_MODE:-full}"
readonly GPU_LIST="${BACKTRACKING_T16_GPU_LIST:-0}"
readonly SESSION_PREFIX="${TXC_T16_TMUX_PREFIX:-backtracking-t16-$MODE}"
readonly DRY_RUN="${BACKTRACKING_T16_LAUNCH_DRY_RUN:-0}"

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
  exit_file="$ROOT/purified/logs/backtracking_window_sweep_t16/${session}.exit"
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
    "BACKTRACKING_T16_MODE=$MODE" \
    "BACKTRACKING_T16_CUDA_DEVICE=${gpus[$shard]}" \
    "BACKTRACKING_T16_NUM_SHARDS=$N_SHARDS" \
    "BACKTRACKING_T16_SHARD_INDEX=$shard" \
    "BACKTRACKING_T16_EXIT_FILE=$exit_file" \
    bash "$WORKER"
  printf "launched %s on GPU %s; exit file %s\n" \
    "$session" "${gpus[$shard]}" "$exit_file"
done
