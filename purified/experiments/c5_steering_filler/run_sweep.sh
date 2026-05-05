#!/usr/bin/env bash
# Launch the 6-cell C5 MW sweep in parallel on the 8× A40 pod.
# Each subprocess is pinned to its own GPU via run_on_gpu.sh and
# fully detached from the parent shell via `setsid -f` so cells survive
# CC restarts / shell exits.
#
# Layout:
#   GPUs 0..2 — txc_pro_mw (slow, ~10-15 hr/cell, InfoNCE all-pairs)
#   GPUs 3..5 — txc_base_mw (fast, ~3-6 hr/cell)
#   GPUs 6, 7 — idle (held in reserve for retries / stretch goals)
#
# Wall-time = max(per-cell), not sum. Expected ≈ 10-15 hr.
#
# Why setsid -f instead of `&` + `wait`: the original version used
# bash backgrounding which makes the cells children of the parent
# shell. When CC restarts (or the launching shell exits), SIGHUP /
# SIGTERM propagates through the process tree and kills the cells
# mid-train. `setsid -f` puts each cell in its own session, so it's
# orphaned and adopted by init (PID 1) — survives shell death.

set -e
cd "$(dirname "$0")/../.."   # purified/

mkdir -p logs

declare -A ASSIGN=(
  [0]="txc_pro_mw 42"
  [1]="txc_pro_mw 1"
  [2]="txc_pro_mw 2"
  [3]="txc_base_mw 42"
  [4]="txc_base_mw 1"
  [5]="txc_base_mw 2"
)

for gpu in "${!ASSIGN[@]}"; do
  read -r arch seed <<<"${ASSIGN[$gpu]}"
  log="logs/c5_filler_gpu${gpu}_${arch}_seed${seed}.log"
  echo "[run_sweep] GPU ${gpu} → ${arch} seed=${seed} → ${log}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    .venv/bin/python -m experiments.c5_steering_filler.run \
    --arch "${arch}" --seed "${seed}" \
    < /dev/null > "${log}" 2>&1
done

echo "[run_sweep] launched 6 detached cells (setsid -f); PIDs:"
pgrep -f "experiments.c5_steering_filler.run" | tee /tmp/p_filler_pids.txt
echo "[run_sweep] check via: ps -ef | grep experiments.c5_steering_filler"
echo "[run_sweep] tail -f logs/c5_filler_gpu*.log to monitor"
