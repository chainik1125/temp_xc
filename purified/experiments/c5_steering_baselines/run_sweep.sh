#!/usr/bin/env bash
# Launch the 6-cell C5 TopK + TFA baselines sweep in parallel.
# 3 seeds × 2 archs pinned to GPUs 0..5; GPUs 6, 7 idle.
#
# Per-cell wall:
#   topk_sae  ~10-15 min train + ~30 min judge ≈ ~45 min per cell
#   tfa       ~30-50 min train + ~30 min judge ≈ ~1-1.5 hr per cell
# Wall = max(per-cell) ≈ 1.5 hr.
#
# Cells launched with `setsid -f` (orphaned to PID 1, own session)
# so they survive CC restarts / shell exits — same lesson learned
# from the rescinded MW sweep at experiments/c5_steering_filler/.

set -e
cd "$(dirname "$0")/../.."   # purified/

mkdir -p logs

declare -A ASSIGN=(
  [0]="topk_sae 42"
  [1]="topk_sae 1"
  [2]="topk_sae 2"
  [3]="tfa 42"
  [4]="tfa 1"
  [5]="tfa 2"
)

for gpu in "${!ASSIGN[@]}"; do
  read -r arch seed <<<"${ASSIGN[$gpu]}"
  log="logs/c5_baselines_gpu${gpu}_${arch}_seed${seed}.log"
  echo "[run_sweep] GPU ${gpu} → ${arch} seed=${seed} → ${log}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    .venv/bin/python -m experiments.c5_steering_baselines.run \
    --arch "${arch}" --seed "${seed}" \
    < /dev/null > "${log}" 2>&1
done

echo "[run_sweep] launched 6 detached cells (setsid -f); PIDs:"
pgrep -f "experiments.c5_steering_baselines.run" | tee /tmp/p_baselines_pids.txt
echo "[run_sweep] check via: ps -ef | grep experiments.c5_steering_baselines"
echo "[run_sweep] tail -f logs/c5_baselines_gpu*.log to monitor"
