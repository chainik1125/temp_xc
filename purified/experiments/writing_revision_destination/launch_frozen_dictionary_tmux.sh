#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly WORKER="$ROOT/purified/experiments/writing_revision_destination/run_frozen_dictionary_runpod.sh"
readonly SESSION="${KLICKE_FROZEN_SESSION:-klicke-frozen-dictionary-t5}"
readonly GPU="${KLICKE_FROZEN_GPU:-3}"
readonly DRY_RUN="${KLICKE_FROZEN_LAUNCH_DRY_RUN:-0}"

if [[ "$(git -C "$ROOT" branch --show-current)" != "neurips-aniket" ]]; then
  echo "refusing to launch: current branch must be neurips-aniket" >&2
  exit 2
fi
if [[ ! -f "$WORKER" ]]; then
  echo "missing worker: $WORKER" >&2
  exit 1
fi
if [[ "$DRY_RUN" == "1" ]]; then
  printf "would launch tmux session %s on physical GPU %s\n" "$SESSION" "$GPU"
  exit 0
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" env \
  "TXC_RUNPOD_ROOT=$ROOT" \
  "KLICKE_PYTHON_BIN=${KLICKE_PYTHON_BIN:-$ROOT/purified/.venv/bin/python}" \
  "KLICKE_FROZEN_GPU=$GPU" \
  "KLICKE_FROZEN_TOKEN_COHORT=${KLICKE_FROZEN_TOKEN_COHORT:-}" \
  "KLICKE_FROZEN_TOKEN_MANIFEST=${KLICKE_FROZEN_TOKEN_MANIFEST:-}" \
  "KLICKE_FROZEN_ACTIVATION_CACHE=${KLICKE_FROZEN_ACTIVATION_CACHE:-}" \
  "KLICKE_FROZEN_CHECKPOINT_ROOT=${KLICKE_FROZEN_CHECKPOINT_ROOT:-}" \
  "KLICKE_FROZEN_OUTPUT_ROOT=${KLICKE_FROZEN_OUTPUT_ROOT:-}" \
  "KLICKE_FROZEN_CODE_DIR=${KLICKE_FROZEN_CODE_DIR:-}" \
  "KLICKE_FROZEN_LOG_ROOT=${KLICKE_FROZEN_LOG_ROOT:-}" \
  "KLICKE_FROZEN_LOG=${KLICKE_FROZEN_LOG:-}" \
  "KLICKE_FROZEN_EXIT_FILE=${KLICKE_FROZEN_EXIT_FILE:-}" \
  "KLICKE_FROZEN_DOWNLOAD_CHECKPOINTS=${KLICKE_FROZEN_DOWNLOAD_CHECKPOINTS:-1}" \
  "KLICKE_FROZEN_MIN_FREE_KB=${KLICKE_FROZEN_MIN_FREE_KB:-6291456}" \
  "KLICKE_FROZEN_BATCH_SIZE=${KLICKE_FROZEN_BATCH_SIZE:-32}" \
  "KLICKE_FROZEN_FOLDS=${KLICKE_FROZEN_FOLDS:-5}" \
  "KLICKE_FROZEN_S_GRID=${KLICKE_FROZEN_S_GRID:-8,16,32,64,128}" \
  "KLICKE_FROZEN_PRIMARY_BUDGET=${KLICKE_FROZEN_PRIMARY_BUDGET:-32}" \
  "KLICKE_FROZEN_C_VALUE=${KLICKE_FROZEN_C_VALUE:-1.0}" \
  "KLICKE_FROZEN_BOOTSTRAP_DRAWS=${KLICKE_FROZEN_BOOTSTRAP_DRAWS:-2000}" \
  "KLICKE_FROZEN_SEED=${KLICKE_FROZEN_SEED:-20260726}" \
  "KLICKE_FROZEN_GATE_MARGIN=${KLICKE_FROZEN_GATE_MARGIN:-0.02}" \
  bash "$WORKER"

printf "launched %s on physical GPU %s\n" "$SESSION" "$GPU"
