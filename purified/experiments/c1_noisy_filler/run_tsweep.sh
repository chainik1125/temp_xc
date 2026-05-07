#!/usr/bin/env bash
# c1_noisy TXC-base T-sweep training: T ∈ {4, 6, 8, 10, 12} for the
# wasteland-style scatter plot. Each T-value sharded across 3 seeds.
# Wasteland uses T = 2..12 stride 1; (decision 2026-05-06) specified stride 2
# = {2, 4, 6, 8, 10, 12}. T=2 already trained — only need {4,6,8,10,12}.
#
# Auto-skip: k_pos * T > d_sae=40 cells skipped by run.py:_is_valid_cell.
# Valid k counts per T:
#   T=4:  k ≤ 10 → 8 cells per seed
#   T=6:  k ≤ 6  → 6 cells per seed
#   T=8:  k ≤ 5  → 5 cells per seed
#   T=10: k ≤ 4  → 4 cells per seed
#   T=12: k ≤ 3  → 3 cells per seed
# Total: 26 cells × 3 seeds = 78 trainings.
#
# Layout: shard by (T, seed) across 7 idle GPUs (0-5, 7). 15 (T, seed)
# pairs, parked round-robin on 7 GPUs.

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# (T, seed) → GPU mapping. 5 T-values × 3 seeds = 15 jobs across 7 GPUs.
# Most GPUs get 2 jobs serial.
declare -a JOBS=(
  # T=4
  "0:4:1"   "1:4:2"   "2:4:42"
  # T=6
  "3:6:1"   "4:6:2"   "5:6:42"
  # T=8
  "7:8:1"   "0:8:2"   "1:8:42"
  # T=10
  "2:10:1"  "3:10:2"  "4:10:42"
  # T=12
  "5:12:1"  "7:12:2"  "0:12:42"
)

# Group by GPU and concat T-values per GPU into one driver call.
# Each driver call runs all k_pos for that (arch, T, seed).
declare -A GPU_JOBS
for j in "${JOBS[@]}"; do
  IFS=":" read -r gpu T seed <<< "$j"
  GPU_JOBS[$gpu]+=" $T:$seed"
done

for gpu in "${!GPU_JOBS[@]}"; do
  pairs="${GPU_JOBS[$gpu]}"
  log="logs/c1_noisy_tsweep_gpu${gpu}.log"
  echo "[run_tsweep] GPU ${gpu} → ${pairs} → ${log}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env AGENT_NAME=agent_filler OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    bash -c "
      cd /workspace/temp_xc/purified
      for pair in $pairs; do
        T=\$(echo \$pair | cut -d: -f1)
        seed=\$(echo \$pair | cut -d: -f2)
        echo \"[gpu${gpu}] start T=\$T seed=\$seed at \$(date -u +%H:%M:%S)\"
        TQDM_DISABLE=1 .venv/bin/python -m experiments.c1_noisy_filler.run_t \
          --T \$T --seed \$seed
      done
    " < /dev/null > "${log}" 2>&1
done

echo "[run_tsweep] launched 7 detached procs; tail logs/c1_noisy_tsweep_gpu*.log"
sleep 1
pgrep -af "experiments.c1_noisy_filler" | head -10
