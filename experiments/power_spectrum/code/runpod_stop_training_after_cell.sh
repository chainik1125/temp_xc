#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${TXC_RUNPOD_ROOT:-/workspace/txc-neurips-aniket}"
readonly CELL="${BACKTRACKING_FOURIER_STOP_AFTER_CELL:?set checkpoint cell name}"
readonly SUMMARY="$ROOT/checkpoints/backtracking_fourier_matched/reviewer-five-point-v1/$CELL/training_summary.json"

while [[ ! -s "$SUMMARY" ]]; do
  sleep 2
done

if ! grep -q '"status": "complete"' "$SUMMARY"; then
  echo "$CELL did not record a complete training summary" >&2
  exit 1
fi

train_pid="$(pgrep -f 'run_backtracking_fourier --phase train' | head -n 1 || true)"
if [[ -n "$train_pid" ]]; then
  kill -TERM "$train_pid"
fi

printf "stopped training after %s\n" "$CELL"
