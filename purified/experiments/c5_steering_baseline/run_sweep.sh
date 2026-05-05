#!/usr/bin/env bash
# Launch the 3-cell C5 T-SAE baseline T=2 re-train in parallel.
# tsae_paper × seeds {42, 1, 2} pinned to GPUs 0..2.
# GPUs 3..7 idle (only T-SAE is in scope; TXC stays at agent_steer's
# v1.1.0 cells, no re-run needed).
#
# Wall-time = max(per-cell), not sum. Expected ≈ 45-60 min.
#
# Cells launched with `setsid -f` (orphaned to PID 1, own session)
# so they survive CC restarts / shell exits — same lesson learned
# from the rescinded MW sweep at experiments/c5_steering_filler/.

set -e
cd "$(dirname "$0")/../.."   # purified/

mkdir -p logs

declare -A ASSIGN=(
  [0]="42"
  [1]="1"
  [2]="2"
)

for gpu in "${!ASSIGN[@]}"; do
  seed="${ASSIGN[$gpu]}"
  log="logs/c5_baseline_gpu${gpu}_tsae_seed${seed}.log"
  echo "[run_sweep] GPU ${gpu} → tsae_paper seed=${seed} → ${log}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    .venv/bin/python -m experiments.c5_steering_baseline.run \
    --seed "${seed}" \
    < /dev/null > "${log}" 2>&1
done

echo "[run_sweep] launched 3 detached cells (setsid -f); PIDs:"
pgrep -f "experiments.c5_steering_baseline.run" | tee /tmp/p_baseline_pids.txt
echo "[run_sweep] check via: ps -ef | grep experiments.c5_steering_baseline"
echo "[run_sweep] tail -f logs/c5_baseline_gpu*.log to monitor"
