#!/usr/bin/env bash
# Launch BASE C3 txc_base × 3 seeds on GPUs 0-2 (DO NOT RUN until
# C1+C2 sweeps wrap — those are on GPUs 0-2 + 6-7).
#
# Each seed iterates T={5, 10, 20} sequentially via the driver's
# ARCH_TRAINING_CFGS for txc_base (3 cfgs × 2 k_feats = 6 evals
# per seed, 3 trainings per seed). Per-seed wall ≈ 7-8 hr; 3 seeds
# parallel on 3 GPUs → wall = max(per-seed) ≈ 7-8 hr.
#
# Cells launched with `setsid -f` (orphaned to PID 1) so they survive
# CC restarts / shell exits.

set -e
cd "$(dirname "$0")/../.."   # purified/

mkdir -p logs

# Pre-flight check: refuse if C1+C2 still on GPUs 0-2 / 6-7.
if pgrep -f "experiments.c[12]_" >/dev/null 2>&1; then
  echo "[run_filler_base] ABORT: C1+C2 procs still running:" >&2
  pgrep -af "experiments.c[12]_" | head >&2
  echo "[run_filler_base] Wait for them to wrap (or kill them) before launching txc_base." >&2
  exit 1
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

declare -A ASSIGN=(
  [0]="42"
  [1]="1"
  [2]="2"
)

for gpu in "${!ASSIGN[@]}"; do
  seed="${ASSIGN[$gpu]}"
  log="logs/c3_base_filler_gpu${gpu}_txc_base_seed${seed}.log"
  echo "[run_filler_base] GPU ${gpu} → txc_base seed=${seed} (T={5,10,20}) → ${log}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    .venv/bin/python -m experiments.c3_probing_base.run \
    --archs txc_base \
    --seeds "${seed}" \
    --k-feats 5 20 \
    < /dev/null > "${log}" 2>&1
done

echo "[run_filler_base] launched 3 detached cells (setsid -f); PIDs:"
pgrep -af "experiments.c3_probing_base.run --archs txc_base" | head | tee /tmp/p_filler_c3base_pids.txt
echo "[run_filler_base] tail -f logs/c3_base_filler_gpu*.log to monitor"
