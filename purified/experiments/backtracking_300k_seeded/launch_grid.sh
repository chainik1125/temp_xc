#!/usr/bin/env bash
set -euo pipefail

role="${1:?usage: launch_grid.sh h100|a40 HISTORICAL_ROOT OUTPUT_ROOT CACHE_FILE PYTHON}"
historical_root="${2:?missing HISTORICAL_ROOT}"
output_root="${3:?missing OUTPUT_ROOT}"
cache_file="${4:?missing CACHE_FILE}"
python_bin="${5:?missing PYTHON}"
repo_root="$(git rev-parse --show-toplevel)"
experiment_dir="$repo_root/purified/experiments/backtracking_300k_seeded"

if [[ "$(git -C "$repo_root" branch --show-current)" != "neurips-aniket" ]]; then
  echo "refusing to continue: current branch must be neurips-aniket" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python runtime is not executable: $python_bin" >&2
  exit 2
fi
mkdir -p "$output_root/logs"

launch_cell() {
  local session="$1"
  local gpu="$2"
  local arch="$3"
  local d_sae="$4"
  local seed="$5"
  local log_file="$output_root/logs/$session.log"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session already exists: $session" >&2
    exit 2
  fi
  tmux new-session -d -s "$session" \
    "cd '$repo_root' && CUDA_VISIBLE_DEVICES='$gpu' CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONUNBUFFERED=1 '$python_bin' '$experiment_dir/train.py' \
    --historical-root '$historical_root' --output-root '$output_root' \
    --cache-file '$cache_file' --arch '$arch' --d-sae '$d_sae' --seed '$seed' \
    2>&1 | tee '$log_file'"
}

case "$role" in
  h100)
    launch_cell c7-300k-txc-seed1 0 txc_base 32768 1
    launch_cell c7-300k-txc-seed2 1 txc_base 32768 2
    launch_cell c7-300k-txc-seed42 2 txc_base 32768 42
    ;;
  a40)
    launch_cell c7-300k-tsae16k-seed42 0 tsae_paper 16384 42
    ;;
  *)
    echo "unknown role: $role" >&2
    exit 2
    ;;
esac

tmux list-sessions
