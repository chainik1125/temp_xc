#!/usr/bin/env bash
# ACTMIX P1-RM launcher — relu-mix arm T-sweep (CARD_RELUMIX.md).
#   PIN=<sha> [SHARDS=2] bash experiments/probing/actmix/launch_relumix.sh
# Same discipline as launch_runpod1.sh: PIN assert (HEAD == PIN,
# ancestor of origin/arxiv, clean tree), then TEMP_BENCH_ALLOW_DIRTY=1
# (pool-row dirty-stamp convention) and detached per-GPU pass chains.
# SHARDS=3 only under a LOG agreement with runpod-a lending a GPU
# (their shard runs from THEIR clone with the same PIN).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
: "${PIN:?set PIN=<40-hex freeze commit sha>}"
SHARDS="${SHARDS:-2}"

HEAD_SHA=$(git rev-parse HEAD)
[ "$HEAD_SHA" = "$PIN" ] || { echo "FATAL: HEAD $HEAD_SHA != PIN $PIN"; exit 1; }
git fetch -q origin
git merge-base --is-ancestor "$PIN" origin/arxiv \
  || { echo "FATAL: PIN not an ancestor of origin/arxiv (push first)"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "FATAL: dirty tree"; exit 1; }

source scripts/set_agent_env.sh runpod-1
mkdir -p /workspace/logs
export TEMP_BENCH_ALLOW_DIRTY=1   # pool-row convention (CARD.md launcher note)
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
TS="--Ts 1 2 4 8 16"

run_gpu() {  # $1=gpu(=shard index)
  local gpu="$1"
  CUDA_VISIBLE_DEVICES="$gpu" nohup bash -c "
    set -e
    $PY -m experiments.probing.actmix.sweep --arm relu-mix \
        --token-archs batchtopk_sae \
        --txc-archs txc_batchtopk_pre txc_batchtopk_post $TS \
        --seeds 42 --untrained-only --shard-index $gpu --shard-count $SHARDS
    $PY -m experiments.probing.actmix.sweep --arm relu-mix \
        --token-archs batchtopk_sae $TS \
        --seeds 1 2 42 --shard-index $gpu --shard-count $SHARDS
    $PY -m experiments.probing.actmix.sweep --arm relu-mix \
        --txc-archs txc_batchtopk_pre $TS \
        --seeds 1 2 42 --shard-index $gpu --shard-count $SHARDS
    $PY -m experiments.probing.actmix.sweep --arm relu-mix \
        --txc-archs txc_batchtopk_post $TS \
        --seeds 42 --shard-index $gpu --shard-count $SHARDS
    echo QUEUE_DONE_RM_GPU_$gpu
  " > "/workspace/logs/actmix_rm_gpu${gpu}.log" 2>&1 &
  echo "RM GPU $gpu queue launched (pid $!) -> /workspace/logs/actmix_rm_gpu${gpu}.log"
}

for g in $(seq 0 $((SHARDS - 1))); do
  [ "$g" -le 1 ] && run_gpu "$g"   # this pod launches shards 0,1 only
done
echo "PIN=$PIN SHARDS=$SHARDS launched (post seeds 1/2 TRAIL per card)."
