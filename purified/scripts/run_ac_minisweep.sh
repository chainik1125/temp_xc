#!/usr/bin/env bash
# AC-only signed-motion sweep (FrequencyBench § 5).
#
# Grid: 4 archs × 3 d_sae × 4 k_pos for ONE seed (pass via SEED=).
#       = 48 cells/seed. Launch seeds {1,2,42} in parallel (RTX 5090 has
#       32 GB; each toy cell uses ~1.8 GB) for a 3-seed run.
#
# Why a d_sae sweep: the bench's headline (window encoders recover the
# hidden sign; per-token SAEs cannot, by DPI) only manifests once the
# window encoder has enough atoms to represent the 2M=38 distinct windows.
# d_sae=20 is the scarce regime (even TXC is reconstruction-bottlenecked);
# d_sae=64 is the ample regime where the architectural gap is clean.
#
# Per-cell wall: ~30-60s on RTX 5090 at n_steps=10K, batch=1024.

set -e
cd "$(dirname "$0")/.."          # purified/

export TEMP_BENCH_ALLOW_DIRTY=1
export TQDM_DISABLE=1
PY=.venv/bin/python

ARCHS=(txc_base topk_sae stacked_sae tsae)
DSAES=(20 40 64)
K_POSES=(1 2 3 4)
DS=toy_signed_motion_M19_d40
SEED=${SEED:-1}
N_STEPS=${N_STEPS:-10000}
BATCH=${BATCH:-1024}

mkdir -p logs
LOG="logs/ac_minisweep_seed${SEED}.log"
: > "$LOG"

total=$(( ${#ARCHS[@]} * ${#DSAES[@]} * ${#K_POSES[@]} ))
echo "[ac-sweep] seed=$SEED archs=${ARCHS[*]} d_sae=${DSAES[*]} k_pos=${K_POSES[*]}" | tee -a "$LOG"
echo "[ac-sweep] $total cells  n_steps=$N_STEPS batch=$BATCH  log=$LOG" | tee -a "$LOG"

i=0
for dsae in "${DSAES[@]}"; do
  for arch in "${ARCHS[@]}"; do
    for kp in "${K_POSES[@]}"; do
      i=$((i + 1))
      printf "[%2d/%d] %-12s d_sae=%-3d k_pos=%d ... " "$i" "$total" "$arch" "$dsae" "$kp" | tee -a "$LOG"
      t0=$SECONDS
      out=$($PY run.py synthetic --arch "$arch" --seed "$SEED" \
              --datasource "$DS" --d-sae "$dsae" --k-pos "$kp" \
              --n-steps "$N_STEPS" --batch-size "$BATCH" 2>&1 || echo "FAILED")
      dt=$((SECONDS - t0))
      if echo "$out" | grep -q "FAILED\|Traceback"; then
        echo "FAIL (${dt}s)" | tee -a "$LOG"
        echo "$out" | tail -6 >> "$LOG"
      else
        s=$(echo "$out"  | grep "s_temp"           | awk '{print $3}')
        acc=$(echo "$out"| grep "sign_probe_acc"   | awk '{print $3}')
        nmse=$(echo "$out"| grep "nmse"            | awk '{print $3}')
        echo "s_temp=$s acc=$acc nmse=$nmse (${dt}s)" | tee -a "$LOG"
      fi
    done
  done
done
echo "[ac-sweep] seed=$SEED DONE" | tee -a "$LOG"
