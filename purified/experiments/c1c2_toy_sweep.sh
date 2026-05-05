#!/usr/bin/env bash
# Launch C1 + C2 toy synthetic sweeps across spare A40s.
# 5 GPUs free now (0, 1, 2, 6, 7) — GPUs 3-5 still on the C5 TFA sweep.
# Toy cells are ~1.8 GB VRAM each; tens of cells fit per GPU.
#
# Partition:
#   GPU 0: C1 archs {topk_sae, tsae_paper}                        (72 cells)
#   GPU 1: C1 archs {tfa, tfa_pos, stacked_sae}                    (108 cells)
#   GPU 2: C1 archs {txc_base, txc_pro}                            (72 cells)
#   GPU 6: C2 archs {topk_sae, stacked_sae}                        (~72-108 cells with T variants)
#   GPU 7: C2 archs {txc_base, txc_pro}                            (~144 cells incl. txc_pro T_max sweep)
# Wall ≈ max(per-GPU) ≈ 72 min (GPU 7 the long pole).
#
# Cells launched with `setsid -f` (orphaned to PID 1) so they survive
# CC restarts / shell exits — same lesson as the C5 sweeps.

set -e
cd "$(dirname "$0")/.."   # purified/

mkdir -p logs

# Cap thread oversubscription. Drivers don't set this themselves and
# default OMP_NUM_THREADS = all cores (76) — with 5 parallel procs,
# that's 380 threads on 76 cores = ~5× oversubscription, which made
# the first launch attempt take >10 min/cell on toy data. Cap at 8 to
# match the c5 driver's convention. Toy archs are GPU-resident; CPU
# threads only matter for data preprocessing.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# ─── C1 — Markov-chain TopK sweep ─────────────────────────────────
declare -A C1=(
  [0]="topk_sae tsae_paper"
  [1]="tfa tfa_pos stacked_sae"
  [2]="txc_base txc_pro"
)
for gpu in "${!C1[@]}"; do
  archs="${C1[$gpu]}"
  log="logs/c1_gpu${gpu}.log"
  echo "[c1c2_sweep] GPU ${gpu} → C1 archs={${archs}} → ${log}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    .venv/bin/python -m experiments.c1_synthetic_topk.run \
    --archs ${archs} \
    < /dev/null > "${log}" 2>&1
done

# ─── C2 — Coupled HMM gAUC sweep ──────────────────────────────────
declare -A C2=(
  [6]="topk_sae stacked_sae"
  [7]="txc_base txc_pro"
)
for gpu in "${!C2[@]}"; do
  archs="${C2[$gpu]}"
  log="logs/c2_gpu${gpu}.log"
  echo "[c1c2_sweep] GPU ${gpu} → C2 archs={${archs}} → ${log}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    .venv/bin/python -m experiments.c2_synthetic_coupled.run \
    --archs ${archs} \
    < /dev/null > "${log}" 2>&1
done

echo "[c1c2_sweep] launched 5 detached sweep procs (setsid -f); PIDs:"
pgrep -af "experiments.c[12]_" | head | tee /tmp/p_c1c2_pids.txt
echo "[c1c2_sweep] tail -f logs/c1_gpu*.log logs/c2_gpu*.log to monitor"
