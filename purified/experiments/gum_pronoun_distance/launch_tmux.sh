#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly WORKER="$ROOT/purified/experiments/gum_pronoun_distance/run_gpu1.sh"
readonly SESSION="${GUM_PRONOUN_SESSION:-gum-pronoun-distance-t5}"
readonly GPU="${GUM_PRONOUN_GPU:-1}"
readonly DRY_RUN="${GUM_PRONOUN_LAUNCH_DRY_RUN:-0}"

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
  "GUM_PRONOUN_PYTHON_BIN=${GUM_PRONOUN_PYTHON_BIN:-$ROOT/purified/.venv/bin/python}" \
  "GUM_PRONOUN_GPU=$GPU" \
  "GUM_PRONOUN_RESULT_ROOT=${GUM_PRONOUN_RESULT_ROOT:-}" \
  "GUM_PRONOUN_SOURCE_ROOT=${GUM_PRONOUN_SOURCE_ROOT:-}" \
  "GUM_PRONOUN_TOKENIZER_ROOT=${GUM_PRONOUN_TOKENIZER_ROOT:-}" \
  "GUM_PRONOUN_COHORT=${GUM_PRONOUN_COHORT:-}" \
  "GUM_PRONOUN_MANIFEST=${GUM_PRONOUN_MANIFEST:-}" \
  "GUM_PRONOUN_ACTIVATION_CACHE=${GUM_PRONOUN_ACTIVATION_CACHE:-}" \
  "GUM_PRONOUN_CHECKPOINT_ROOT=${GUM_PRONOUN_CHECKPOINT_ROOT:-}" \
  "GUM_PRONOUN_CODE_DIR=${GUM_PRONOUN_CODE_DIR:-}" \
  "GUM_PRONOUN_OUTPUT_DIR=${GUM_PRONOUN_OUTPUT_DIR:-}" \
  "GUM_PRONOUN_LOG_ROOT=${GUM_PRONOUN_LOG_ROOT:-}" \
  "GUM_PRONOUN_HF_HOME=${GUM_PRONOUN_HF_HOME:-/workspace/.cache/huggingface}" \
  "GUM_PRONOUN_MIN_FREE_KB=${GUM_PRONOUN_MIN_FREE_KB:-27262976}" \
  "GUM_PRONOUN_SHARD_SIZE=${GUM_PRONOUN_SHARD_SIZE:-256}" \
  "GUM_PRONOUN_ENCODER_BATCH_SIZE=${GUM_PRONOUN_ENCODER_BATCH_SIZE:-32}" \
  "GUM_PRONOUN_BUDGETS=${GUM_PRONOUN_BUDGETS:-8,16,32,64,128}" \
  "GUM_PRONOUN_PRIMARY_BUDGET=${GUM_PRONOUN_PRIMARY_BUDGET:-32}" \
  "GUM_PRONOUN_FOLDS=${GUM_PRONOUN_FOLDS:-5}" \
  "GUM_PRONOUN_BOOTSTRAP_DRAWS=${GUM_PRONOUN_BOOTSTRAP_DRAWS:-2000}" \
  "GUM_PRONOUN_SEED=${GUM_PRONOUN_SEED:-20260726}" \
  "GUM_PRONOUN_GATE_MARGIN=${GUM_PRONOUN_GATE_MARGIN:-0.02}" \
  bash "$WORKER"

printf "launched %s on physical GPU %s\n" "$SESSION" "$GPU"
