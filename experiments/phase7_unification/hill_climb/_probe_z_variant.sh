#!/bin/bash
# Wrapper to probe a Z-variant ckpt under PAPER methodology, then dump
# a one-line summary vs the leaderboard. Run AFTER training completes
# and the run_id is registered in training_index.jsonl.
#
# Usage:
#     bash experiments/phase7_unification/hill_climb/_probe_z_variant.sh \
#          hill_z_ranked_T20_s5__seed42

set -euo pipefail
cd "$(dirname "$0")/../../.."

run_id="${1:-}"
if [[ -z "$run_id" ]]; then
  echo "usage: $0 <run_id>" >&2
  exit 2
fi

logfile="/tmp/z_logs/probe_${run_id}.log"
mkdir -p "$(dirname "$logfile")"

echo "=== probing $run_id under PAPER methodology, S=32, k_feat ∈ {5, 20} ==="
HF_HOME=/home/elysium/.cache/huggingface \
TQDM_DISABLE=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python -m experiments.phase7_unification.run_probing_phase7 \
    --run_ids "$run_id" \
    --S 32 \
    --k_feat 5 20 \
  2>&1 | tee "$logfile"

echo ""
echo "=== summary vs leaderboard (PAPER set) ==="
HF_HOME=/home/elysium/.cache/huggingface \
TQDM_DISABLE=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python -m experiments.phase7_unification.hill_climb._summarize_results \
    2>&1 | tail -40
