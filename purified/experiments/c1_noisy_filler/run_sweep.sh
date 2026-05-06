#!/usr/bin/env bash
# Launch c1_noisy + C2 across all 8 A40 GPUs.
#
# Layout:
#   GPU 0-2: c1_noisy --seeds {1, 2, 42} (one seed per GPU, 6 archs each)
#   GPU 3-5: c2 --seeds {1, 2, 42} --k-poses 1..8 (safe low-k slice)
#   GPU 6:   c2 --archs txc_pro --k-poses 10..20 (high-k; txc_pro
#            t_sample=2 keeps k_pos × t_sample ≤ 20 valid for all k)
#   GPU 7:   c2 --archs topk_sae --k-poses 10..20 (per-token, no
#            constraint; mostly cache hits since topk_sae already done)
#
# Cells launched with `setsid -f` (orphaned to PID 1).

set -e
cd "$(dirname "$0")/../.."   # purified/

mkdir -p logs

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# ── c1_noisy across GPUs 0-2, one seed each ──────────────────────
for gpu in 0 1 2; do
  case "$gpu" in
    0) seed=1 ;;
    1) seed=2 ;;
    2) seed=42 ;;
  esac
  log="logs/c1_noisy_gpu${gpu}_seed${seed}.log"
  echo "[run_sweep] GPU ${gpu} → c1_noisy seed=${seed} → ${log}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    .venv/bin/python -m experiments.c1_noisy_filler.run \
    --seeds "${seed}" \
    < /dev/null > "${log}" 2>&1
done

# ── c2 low-k slice (k=1..8), per-seed across GPUs 3-5 ─────────────
for gpu in 3 4 5; do
  case "$gpu" in
    3) seed=1 ;;
    4) seed=2 ;;
    5) seed=42 ;;
  esac
  log="logs/c2_lowk_gpu${gpu}_seed${seed}.log"
  echo "[run_sweep] GPU ${gpu} → c2 lowk seed=${seed} → ${log}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    .venv/bin/python -m experiments.c2_synthetic_coupled.run \
    --seeds "${seed}" \
    --k-poses 1 2 3 4 5 6 8 \
    < /dev/null > "${log}" 2>&1
done

# ── c2 high-k slice on txc_pro (GPU 6) + topk_sae (GPU 7) ─────────
log="logs/c2_highk_txcpro_gpu6.log"
echo "[run_sweep] GPU 6 → c2 highk txc_pro all seeds → ${log}"
setsid -f bash scripts/run_on_gpu.sh 6 -- \
  env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  .venv/bin/python -m experiments.c2_synthetic_coupled.run \
  --archs txc_pro \
  --k-poses 10 12 15 17 20 \
  < /dev/null > "${log}" 2>&1

log="logs/c2_highk_topk_gpu7.log"
echo "[run_sweep] GPU 7 → c2 highk topk_sae all seeds → ${log}"
setsid -f bash scripts/run_on_gpu.sh 7 -- \
  env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  .venv/bin/python -m experiments.c2_synthetic_coupled.run \
  --archs topk_sae \
  --k-poses 10 12 15 17 20 \
  < /dev/null > "${log}" 2>&1

echo "[run_sweep] launched 8 detached procs (setsid -f); PIDs:"
pgrep -af "experiments.c[12]" | head -20 | tee /tmp/p_c1c2_combined_pids.txt
echo "[run_sweep] tail -f logs/c1_noisy_gpu*.log logs/c2_lowk_gpu*.log logs/c2_highk*.log"
