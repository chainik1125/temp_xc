#!/usr/bin/env bash
# C7 sweep launcher — runs the 3 available archs × seed=42 in background.
#
# Owned by agent_back. Invoked after smoke passes. Logs to
# logs/c7_sweep_seed42.log; cell metrics + judge outputs land in
# results/leaderboard.jsonl + results/runs/<eval_key>/judge_outputs.jsonl.
#
# Once additional arches land (agent_paper), re-run with:
#   ARCHS="stacked_sae tfa mlc txc_pro" bash scripts/c7_run_sweep.sh
#
# Cells are cached by (act_cache_key, train_key, eval_key) so re-running
# is idempotent; only missing cells compute.

set -eu

cd "$(dirname "$0")/.."   # purified/

mkdir -p logs

ARCHS="${ARCHS:-topk_sae tsae_paper txc_base}"
SEEDS="${SEEDS:-42}"
LOG="logs/c7_sweep_seed${SEEDS// /_}.log"

echo "[c7-sweep] starting — archs='$ARCHS' seeds='$SEEDS' log=$LOG" >&2

TQDM_DISABLE=1 .venv/bin/python -m experiments.c7_backtracking.run \
    --archs $ARCHS --seeds $SEEDS \
    > "$LOG" 2>&1 &
PID=$!
echo "[c7-sweep] background PID=$PID" >&2
echo "$PID"
