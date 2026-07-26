#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly WORKER="$ROOT/purified/experiments/backtracking_window_sweep/run_teacher_force_runpod.sh"
readonly GPU_LIST="${BACKTRACKING_TEACHER_GPU_LIST:-0}"
readonly SESSION_PREFIX="${BACKTRACKING_TEACHER_TMUX_PREFIX:-ward-t16-extract}"
readonly DRY_RUN="${BACKTRACKING_TEACHER_LAUNCH_DRY_RUN:-0}"
readonly TRACES="${BACKTRACKING_TEACHER_TRACES:?set BACKTRACKING_TEACHER_TRACES}"
readonly TRACES_SHA256="${BACKTRACKING_TEACHER_TRACES_SHA256:?set BACKTRACKING_TEACHER_TRACES_SHA256}"
readonly SOURCE_PATH="${BACKTRACKING_TEACHER_SOURCE_PATH:?set BACKTRACKING_TEACHER_SOURCE_PATH}"
readonly SOURCE_COMMIT="${BACKTRACKING_TEACHER_SOURCE_COMMIT:?set BACKTRACKING_TEACHER_SOURCE_COMMIT}"

if [[ "$(git -C "$ROOT" branch --show-current)" != "neurips-aniket" ]]; then
  echo "refusing to launch: current branch must be neurips-aniket" >&2
  exit 2
fi
if [[ ! -f "$WORKER" ]]; then
  echo "missing worker: $WORKER" >&2
  exit 2
fi
IFS=',' read -r -a gpus <<<"$GPU_LIST"
readonly N_SHARDS="${#gpus[@]}"
if ((N_SHARDS < 1)); then
  echo "BACKTRACKING_TEACHER_GPU_LIST must contain at least one GPU id" >&2
  exit 2
fi

for shard in "${!gpus[@]}"; do
  session="${SESSION_PREFIX}-${shard}"
  exit_file="$ROOT/purified/logs/backtracking_teacher_force/${session}.exit"
  if ((DRY_RUN == 1)); then
    printf "would launch tmux session %s on GPU %s (trace shard %s/%s)\n" \
      "$session" "${gpus[$shard]}" "$shard" "$N_SHARDS"
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session already exists: $session" >&2
    exit 2
  fi
  tmux new-session -d -s "$session" env \
    "TXC_RUNPOD_ROOT=$ROOT" \
    "BACKTRACKING_TEACHER_PHASE=extract" \
    "BACKTRACKING_TEACHER_TRACES=$TRACES" \
    "BACKTRACKING_TEACHER_TRACES_SHA256=$TRACES_SHA256" \
    "BACKTRACKING_TEACHER_SOURCE_PATH=$SOURCE_PATH" \
    "BACKTRACKING_TEACHER_SOURCE_COMMIT=$SOURCE_COMMIT" \
    "BACKTRACKING_TEACHER_CUDA_DEVICE=${gpus[$shard]}" \
    "BACKTRACKING_TEACHER_NUM_SHARDS=$N_SHARDS" \
    "BACKTRACKING_TEACHER_SHARD_INDEX=$shard" \
    "BACKTRACKING_TEACHER_EXIT_FILE=$exit_file" \
    bash "$WORKER"
  printf "launched %s on GPU %s; exit file %s\n" \
    "$session" "${gpus[$shard]}" "$exit_file"
done
