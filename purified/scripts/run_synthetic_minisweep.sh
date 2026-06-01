#!/usr/bin/env bash
# Mini-sweep for § 4 paper reproduction.
# 6 archs × 2 datasets × 5 k_pos × 1 seed = 60 cells.
# Per-cell wall: ~30s on RTX 5090 at n_steps=10K, batch=1024.
# Total wall: ~30 min.

set -e
cd "$(dirname "$0")/.."          # purified/

export TEMP_BENCH_ALLOW_DIRTY=1
PY=.venv/bin/python

ARCHS=(txc_base topk_sae stacked_sae tsae)
DATASOURCES=(toy_coupled_K10_M20_d256 toy_markov_n20_d40_noisy)
K_POSES=(1 2 5 10 20)
SEED=${SEED:-1}
N_STEPS=${N_STEPS:-10000}
BATCH=${BATCH:-1024}

mkdir -p logs
LOG="logs/synth_minisweep_seed${SEED}.log"
: > "$LOG"

echo "[minisweep] archs=${ARCHS[*]}"
echo "[minisweep] datasources=${DATASOURCES[*]}"
echo "[minisweep] k_poses=${K_POSES[*]}  seed=$SEED  n_steps=$N_STEPS  batch=$BATCH"
echo "[minisweep] log: $LOG"
echo ""

total=$(( ${#ARCHS[@]} * ${#DATASOURCES[@]} * ${#K_POSES[@]} ))
i=0
for ds in "${DATASOURCES[@]}"; do
  for arch in "${ARCHS[@]}"; do
    for kp in "${K_POSES[@]}"; do
      i=$((i + 1))
      printf "[%3d/%d] %-12s %-32s k=%-2d ... " \
        $i $total $arch $ds $kp | tee -a "$LOG"
      t0=$SECONDS
      out=$($PY run.py synthetic \
              --arch "$arch" --seed "$SEED" \
              --datasource "$ds" \
              --k-pos $kp \
              --n-steps $N_STEPS --batch-size $BATCH 2>&1 || echo "FAILED")
      dt=$((SECONDS - t0))
      if echo "$out" | grep -q "FAILED\|Traceback"; then
        echo "FAIL ($dt s)" | tee -a "$LOG"
        echo "$out" | tail -5 >> "$LOG"
      else
        e=$(echo "$out" | grep "eauc " | awk '{print $3}')
        g=$(echo "$out" | grep "gauc " | awk '{print $3}')
        n=$(echo "$out" | grep "nmse" | awk '{print $3}')
        cached=$(echo "$out" | grep "eval_key" | grep -o "cached=True\|cached=False" | head -1)
        echo "eAUC=$e gAUC=$g NMSE=$n ($dt s, $cached)" | tee -a "$LOG"
      fi
    done
  done
done

echo ""
echo "[minisweep] done. results: $LOG"
