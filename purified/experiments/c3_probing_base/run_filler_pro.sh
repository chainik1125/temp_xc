#!/usr/bin/env bash
# Launch BASE C3 txc_pro × 3 seeds on GPUs 3-5 (currently free; C5
# done, C1+C2 are on GPUs 0-2 + 6-7).
# Per-cell: ~3.5 hr (T_max=10 InfoNCE all-pairs).
# Wall = max(per-cell) ≈ 3.5 hr (parallel).
#
# Cells launched with `setsid -f` (orphaned to PID 1) so they survive
# CC restarts / shell exits — same lesson as the C5 sweep.

set -e
cd "$(dirname "$0")/../.."   # purified/

mkdir -p logs

# Cap thread oversubscription with the c1+c2 sweep still running.
# 76 cores / (5 c1+c2 + 3 new c3 procs) = ~9 threads each clean budget;
# OMP=8 keeps headroom.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

declare -A ASSIGN=(
  [3]="42"
  [4]="1"
  [5]="2"
)

for gpu in "${!ASSIGN[@]}"; do
  seed="${ASSIGN[$gpu]}"
  log="logs/c3_base_filler_gpu${gpu}_txc_pro_seed${seed}.log"
  echo "[run_filler_pro] GPU ${gpu} → txc_pro seed=${seed} → ${log}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    .venv/bin/python -m experiments.c3_probing_base.run \
    --archs txc_pro \
    --seeds "${seed}" \
    --k-feats 5 20 \
    < /dev/null > "${log}" 2>&1
done

echo "[run_filler_pro] launched 3 detached cells (setsid -f); PIDs:"
pgrep -af "experiments.c3_probing_base.run --archs txc_pro" | head | tee /tmp/p_filler_c3pro_pids.txt
echo "[run_filler_pro] tail -f logs/c3_base_filler_gpu*.log to monitor"
