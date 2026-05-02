#!/bin/bash
# Post-chain eval driver — uses Han's (a)-style sharing.
#
# Polls for chain script (PID 142153) to finish, then runs the standard
# steering pipeline ONCE with all 8 archs as a single batch:
#   select_features → diagnose_z_magnitudes → intervene_paper_clamp_normalised → grade
#
# Each pipeline step shares the expensive Gemma-2-2b forward across archs:
#   - select_features: one Gemma capture, disk-cached, encode each SAE per-arch
#   - diagnose: one Gemma capture (in-process share_acts mechanism)
#   - intervene: one Gemma load, swap hooks per arch
#   - grade: API-parallelized via --n-workers
#
# Right-edge protocol only (cheapest). Per-position can come later if RE shows
# any TXC win above prereg threshold.
#
# Logs to /tmp/eval_t10_chain.log
set -e
cd /workspace/temp_xc
export TQDM_DISABLE=1

LOG=/tmp/eval_t10_chain.log
echo "=== Eval driver started $(date -u) ===" >> $LOG

# Wait for chain to fully finish
echo "=== Waiting for chain (PID 142153) ===" >> $LOG
while kill -0 142153 2>/dev/null; do
  sleep 60
done
echo "=== Chain finished $(date -u) ===" >> $LOG

# All T=10 deadzone-escape arch_ids the chain produces
ARCHS=(
  "txc_h8_t10_kpos20_shifts10"
  "txc_h8_t10_kpos20_shifts2"
  "subseq_h8_tmax10_tsamp5_kpos20_shifts2_ctg"
  "subseq_h8_tmax10_tsamp5_kpos20_shifts2_gauss_s1.5_3.0_g2"
  "spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_indep_uniform_contr"
  "spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_nested_uniform_contr"
  "spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_indep_gauss_s1.5_3.0_g2_contr"
  "spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_nested_gauss_s1.5_3.0_g2_contr"
)

# Filter out archs whose ckpts didn't land
PRESENT=()
for arch in "${ARCHS[@]}"; do
  ckpt=/workspace/temp_xc/experiments/phase7_unification/results/ckpts/${arch}__seed42.pt
  if [ -f "$ckpt" ]; then
    PRESENT+=("$arch")
  else
    echo "=== [$arch] SKIP — ckpt not found ===" >> $LOG
  fi
done
echo "=== Eval batch: ${#PRESENT[@]} archs present ===" >> $LOG

# Step 1 — select_features (shared Gemma, disk cache)
echo "=== select_features START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.select_features \
  --archs "${PRESENT[@]}" --seed 42 >> $LOG 2>&1
echo "=== select_features DONE $(date -u) ===" >> $LOG

# Step 2 — diagnose_z_magnitudes (shared L12 acts via share_acts mechanism)
echo "=== diagnose_z_magnitudes START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.diagnose_z_magnitudes \
  --archs "${PRESENT[@]}" --seed 42 >> $LOG 2>&1
echo "=== diagnose_z_magnitudes DONE $(date -u) ===" >> $LOG

# Step 3 — intervene_paper_clamp_normalised — multi-process parallel (b)-style
#
# Each process is INDEPENDENT (own CUDA context), B=7 strengths per arch
# unchanged → bit-parity preserved vs sequential. GPU just timeshares.
# Memory: 5GB Gemma + ~7GB SAE/proc → 2 procs × ~14GB ≈ 28GB on 48GB A40
# (3+ procs would OOM on A40 under transient peaks). On H100 80GB this can
# be raised to 4-5.
#
# Within each process we still benefit from (a)-style shared Gemma across
# the arch sub-list it owns.
N_GROUPS=${N_GROUPS:-2}
echo "=== intervene_paper_clamp_normalised START $(date -u) (N_GROUPS=$N_GROUPS) ===" >> $LOG
N=${#PRESENT[@]}
PER=$(( (N + N_GROUPS - 1) / N_GROUPS ))   # ceil(N/N_GROUPS)
declare -a PIDS=()
for ((g=0; g<N_GROUPS; g++)); do
  off=$((g * PER))
  if [ $off -ge $N ]; then break; fi
  GROUP=("${PRESENT[@]:$off:$PER}")
  echo "  group $g: ${GROUP[*]}" >> $LOG
  .venv/bin/python -m experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_normalised \
    --archs "${GROUP[@]}" --seed 42 >> /tmp/intervene_g${g}.log 2>&1 &
  PIDS+=("$!")
done
echo "  spawned ${#PIDS[@]} parallel intervene processes: ${PIDS[*]}" >> $LOG
# Wait for all groups; collect statuses
ALL_OK=1
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    ALL_OK=0
    echo "  intervene PID $pid FAILED" >> $LOG
  fi
done
# Concat group logs into main log for visibility
for ((g=0; g<N_GROUPS; g++)); do
  if [ -f /tmp/intervene_g${g}.log ]; then
    echo "  --- group $g log ---" >> $LOG
    cat /tmp/intervene_g${g}.log >> $LOG
  fi
done
if [ $ALL_OK -eq 0 ]; then
  echo "=== intervene_paper_clamp_normalised PARTIAL (some procs failed) $(date -u) ===" >> $LOG
else
  echo "=== intervene_paper_clamp_normalised DONE $(date -u) ===" >> $LOG
fi

# Step 4 — grade_with_sonnet (API-parallel)
echo "=== grade_with_sonnet START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.grade_with_sonnet \
  --archs "${PRESENT[@]}" \
  --subdir steering_paper_normalised >> $LOG 2>&1
echo "=== grade_with_sonnet DONE $(date -u) ===" >> $LOG

echo "=== Eval driver ALL DONE $(date -u) ===" >> $LOG
