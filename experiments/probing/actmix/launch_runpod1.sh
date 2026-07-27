#!/usr/bin/env bash
# ACTMIX P1 launcher — runpod-1, GPUs 0,1. Freeze-commit-run discipline:
#   PIN=<sha> bash experiments/probing/actmix/launch_runpod1.sh
# Asserts HEAD == PIN, PIN is an ancestor of origin/arxiv, clean tree;
# then launches the queue passes detached (nohup) one shard per GPU.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
: "${PIN:?set PIN=<40-hex freeze commit sha>}"

# ── _assert_pinned ──
HEAD_SHA=$(git rev-parse HEAD)
[ "$HEAD_SHA" = "$PIN" ] || { echo "FATAL: HEAD $HEAD_SHA != PIN $PIN"; exit 1; }
git fetch -q origin
git merge-base --is-ancestor "$PIN" origin/arxiv \
  || { echo "FATAL: PIN not an ancestor of origin/arxiv (push first)"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "FATAL: dirty tree"; exit 1; }

source scripts/set_agent_env.sh runpod-1
mkdir -p /workspace/logs

# Pool-row dirty-stamp convention (task_hunt precedent — see
# diafaces/merge_panel_payload.py): run_experiment appends to the
# TRACKED leaderboard.jsonl, so every cell after the first sees a
# growing `git diff`. Integrity is carried by the PIN assert above
# (launch-clean at the pinned sha); rows after cell 1 carry
# dirty=true stamps whose diff is the leaderboard growth itself.
# NO code edits are permitted in this clone while the queue runs.
export TEMP_BENCH_ALLOW_DIRTY=1
export PYTHONUNBUFFERED=1   # nohup logs must stream, not block-buffer

PY=.venv/bin/python
# AMENDMENT 2: tsae_btkonly TRAINED cells dropped from tonight's queue —
# its consumes='sequence' serving moves full 128-token batches per step
# to sample one pair (measured 7.8 s/step = 43 h/train vs the v1
# pipeline's train_window_size=2 pair serving; a v2 serving mismatch,
# NOT an arch defect — flagged to mac-a, fix belongs to the arch/serving
# owner). TSAE is the spec's parenthetical-optional column; it remains
# covered by the paper-match arm (complete) + untrained twins.
COMMON_TOKEN="--token-archs batchtopk_sae_btkonly"
TXC_PRE="--txc-archs txc_batchtopk_pre_btkonly"
TXC_POST="--txc-archs txc_batchtopk_post_btkonly"
TS="--Ts 1 2 4 8 16"

launch_shard() {  # $1=gpu  $2=logname  $3...=driver args
  local gpu="$1" log="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$gpu" nohup $PY -m experiments.probing.actmix.sweep \
      "$@" --shard-index "$gpu" --shard-count 2 \
      > "/workspace/logs/${log}_gpu${gpu}.log" 2>&1 &
  echo "launched $log on GPU $gpu (pid $!)"
}

# Each GPU runs the SAME pass sequence; shards split cells round-robin.
# Passes within a GPU run sequentially via a wrapper shell.
run_gpu() {  # $1=gpu
  local gpu="$1"
  CUDA_VISIBLE_DEVICES="$gpu" nohup bash -c "
    set -e
    $PY -m experiments.probing.actmix.sweep --arm btk-only $COMMON_TOKEN \
        --txc-archs txc_batchtopk_pre_btkonly txc_batchtopk_post_btkonly $TS \
        --seeds 42 --untrained-only --shard-index $gpu --shard-count 2
    $PY -m experiments.probing.actmix.sweep --arm btk-only $COMMON_TOKEN $TS \
        --seeds 1 2 42 --shard-index $gpu --shard-count 2
    $PY -m experiments.probing.actmix.sweep --arm btk-only $TXC_PRE $TS \
        --seeds 1 2 42 --shard-index $gpu --shard-count 2
    $PY -m experiments.probing.actmix.sweep --arm btk-only $TXC_POST $TS \
        --seeds 42 --shard-index $gpu --shard-count 2
    $PY -m experiments.probing.actmix.sweep --arm btk-only $TXC_POST $TS \
        --seeds 1 2 --shard-index $gpu --shard-count 2
    echo QUEUE_DONE_GPU_$gpu
  " > "/workspace/logs/actmix_p1_gpu${gpu}.log" 2>&1 &
  echo "GPU $gpu queue launched (pid $!) -> /workspace/logs/actmix_p1_gpu${gpu}.log"
}

run_gpu 0
run_gpu 1
echo "PIN=$PIN launched. Tail: tail -f /workspace/logs/actmix_p1_gpu{0,1}.log"
