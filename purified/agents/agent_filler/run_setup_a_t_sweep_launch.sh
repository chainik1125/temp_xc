#!/usr/bin/env bash
# Setup A T-sweep launcher — txc_base T={2,4,6,8,10,12} × 3 seeds = 18 shards.
#
# 8× A40 pod. ρ-sweep currently on GPUs 1-5; GPUs 0, 6, 7 idle. As ρ-sweep
# completes, more GPUs free up and the runner cache-hits any duplicate work.
#
# Strategy: round-robin shards across GPUs 0, 6, 7 first (idle now). The
# runner's idempotency means a shard launched after the ρ-sweep proc on
# GPU 1 finishes will cache-hit on existing trains — but the txc_base
# T-sweep cells are NEW (no overlap with ρ-sweep, which was topk_sae /
# txc_base T=5 / txc_pro at ρ ∈ {0.0, 0.3, 0.6, 0.9}).
#
# Cells: T-sweep auto-skips invalid k_pos. Per T:
#   T=2:  k ≤ 20 → 12 valid → 36 cells (×3 seeds)
#   T=4:  k ≤ 10 →  8 valid → 24 cells
#   T=6:  k ≤  6 →  6 valid → 18 cells
#   T=8:  k ≤  5 →  5 valid → 15 cells
#   T=10: k ≤  4 →  4 valid → 12 cells
#   T=12: k ≤  3 →  3 valid →  9 cells
# Total: 114 NEW cells. ~3 min/cell on A40 toy → ~6 hr serial / ~50 min on 7 GPUs.

set -e
cd "$(dirname "$0")/../.."   # purified/

mkdir -p logs
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 18 shards (T × seed) → distribute across 7 GPUs (0, 1-7).
# GPUs 1-5 currently busy with ρ-sweep; their shards will queue here
# but only START when GPU is free (run_on_gpu.sh sets CUDA_VISIBLE_DEVICES
# but doesn't gate on memory). To avoid clobbering ρ-sweep, we use only
# IDLE GPUs (0, 6, 7) initially. As ρ-sweep ends, manually launch
# remaining shards.

declare -a SHARDS=(
  "0:2:1"   "6:2:2"   "7:2:42"     # T=2 × 3 seeds
  "0:4:1"   "6:4:2"   "7:4:42"     # T=4 × 3 seeds  (chained on 0/6/7)
  "0:6:1"   "6:6:2"   "7:6:42"     # T=6 × 3 seeds
  "0:8:1"   "6:8:2"   "7:8:42"     # T=8 × 3 seeds
  "0:10:1"  "6:10:2"  "7:10:42"    # T=10 × 3 seeds
  "0:12:1"  "6:12:2"  "7:12:42"    # T=12 × 3 seeds
)

# Group by GPU; chain shards per GPU.
declare -A GPU_CHAIN
for s in "${SHARDS[@]}"; do
  IFS=":" read -r gpu T seed <<< "$s"
  GPU_CHAIN[$gpu]+=" $T:$seed"
done

for gpu in "${!GPU_CHAIN[@]}"; do
  pairs="${GPU_CHAIN[$gpu]}"
  log="logs/setup_a_tsweep_gpu${gpu}.log"
  echo "[setup_a_tsweep] GPU ${gpu} → ${pairs} → ${log}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env AGENT_NAME=agent_filler OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TQDM_DISABLE=1 \
    bash -c "
      cd /workspace/temp_xc/purified
      for pair in ${pairs}; do
        T=\$(echo \$pair | cut -d: -f1)
        seed=\$(echo \$pair | cut -d: -f2)
        echo \"[gpu${gpu}] start T=\$T seed=\$seed at \$(date -u +%H:%M:%S)\"
        TQDM_DISABLE=1 .venv/bin/python -m experiments.c2_synthetic_coupled.run_setup_a_t_sweep \
          --T \$T --seed \$seed
      done
    " < /dev/null > "${log}" 2>&1
done

echo "[setup_a_tsweep] launched 3 detached chains on GPUs 0, 6, 7"
sleep 2
pgrep -af "experiments.c2_synthetic_coupled.run_setup_a_t_sweep" | head
