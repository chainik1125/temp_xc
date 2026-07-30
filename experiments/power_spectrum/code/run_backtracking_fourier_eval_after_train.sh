#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly TRAIN_EXIT="$ROOT/purified/logs/backtracking_fourier_matched/train.exit"
readonly ARTIFACT="${BACKTRACKING_FOURIER_ARTIFACT:?set artifact path}"
readonly MANIFEST="${BACKTRACKING_FOURIER_MANIFEST:?set manifest path}"
readonly SEED="${BACKTRACKING_FOURIER_SEEDS:?set one evaluation seed}"

if [[ "$SEED" == *,* ]]; then
  echo "evaluation watcher accepts exactly one seed" >&2
  exit 2
fi

while pgrep -f "run_backtracking_fourier --phase train" >/dev/null; do
  sleep 30
done

if [[ ! -f "$TRAIN_EXIT" ]] || [[ "$(cat "$TRAIN_EXIT")" != "0" ]]; then
  echo "training did not complete successfully" >&2
  exit 1
fi

while [[ ! -s "$ARTIFACT" ]] || [[ ! -s "$MANIFEST" ]]; do
  sleep 30
done

export BACKTRACKING_FOURIER_PHASE=eval
export BACKTRACKING_FOURIER_ALLOW_RECOVERED_ARTIFACT=1
exec bash \
  "$ROOT/purified/experiments/power_spectrum/code/run_backtracking_fourier_runpod.sh"
