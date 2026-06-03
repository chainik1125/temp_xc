#!/usr/bin/env bash
# AC-only signed-motion sweep, per docs/synthetic_benchmark_guidance.md.
#
# Ground truth: F = 19 feature directions (the alphabet). The hidden sign is
# a dynamical latent, not a feature. We sweep d_sae anchored on F, focused on
# the scarce regime d_sae <= F (the realistic, memorization-free regime: the
# per-tile sign probe has < 2M = 38 features there), plus ONE over-complete
# reference (d_sae = 38 = 2F) where the crosscoder can tabulate the windows
# and the probe is consequently confounded.
#
# Window archs (txc_base, stacked_sae) are swept over power-of-two T; per-token
# archs (topk_sae, tsae) are T=1. Every metric is scored on a common L=32
# eval window, tiled non-overlapping into L/T sub-windows.
#
# Grid per seed: 8 (arch,T) x 5 d_sae = 40 cells. Run seeds {1,2,42}.
# k_pos=1 (sparsest; in the scarce regime k_pos>1 mostly clips k_win to d_sae).

set -e
cd "$(dirname "$0")/.."          # purified/

export TEMP_BENCH_ALLOW_DIRTY=1
export TQDM_DISABLE=1
PY=.venv/bin/python

# (arch, T) pairs.
ARCH_TS=(
  "txc_base 2" "txc_base 4" "txc_base 8"
  "stacked_sae 2" "stacked_sae 4" "stacked_sae 8"
  "topk_sae 1" "tsae 1"
)
DSAES=(4 8 16 19 38)
DS=toy_signed_motion_M19_d40
L=32
SEED=${SEED:-1}
KPOS=${KPOS:-1}
N_STEPS=${N_STEPS:-10000}
BATCH=${BATCH:-1024}

mkdir -p logs
LOG="logs/ac_minisweep_seed${SEED}.log"
: > "$LOG"

total=$(( ${#ARCH_TS[@]} * ${#DSAES[@]} ))
echo "[ac-sweep] seed=$SEED k_pos=$KPOS L=$L n_steps=$N_STEPS  ($total cells)" | tee -a "$LOG"

i=0
for dsae in "${DSAES[@]}"; do
  for at in "${ARCH_TS[@]}"; do
    set -- $at; arch=$1; T=$2
    i=$((i + 1))
    printf "[%2d/%d] %-12s T=%s d_sae=%-3d ... " "$i" "$total" "$arch" "$T" "$dsae" | tee -a "$LOG"
    t0=$SECONDS
    out=$($PY run.py synthetic --arch "$arch" --seed "$SEED" \
            --datasource "$DS" --d-sae "$dsae" --k-pos "$KPOS" --T "$T" \
            --eval-window-l "$L" --n-steps "$N_STEPS" --batch-size "$BATCH" 2>&1 || echo "FAILED")
    dt=$((SECONDS - t0))
    if echo "$out" | grep -q "FAILED\|Traceback"; then
      echo "FAIL (${dt}s)" | tee -a "$LOG"
      echo "$out" | tail -6 >> "$LOG"
    else
      s=$(echo "$out"  | grep "s_temp "         | awk '{print $3}')
      acc=$(echo "$out"| grep "sign_probe_acc"  | awk '{print $3}')
      e=$(echo "$out"  | grep "eauc "           | awk '{print $3}')
      n=$(echo "$out"  | grep "nmse "           | awk '{print $3}')
      echo "s_temp=$s acc=$acc eauc=$e nmse=$n (${dt}s)" | tee -a "$LOG"
    fi
  done
done
echo "[ac-sweep] seed=$SEED DONE" | tee -a "$LOG"
