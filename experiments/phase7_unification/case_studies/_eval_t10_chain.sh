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

# Step 3 — intervene_paper_clamp_normalised (shared subject loaded once)
echo "=== intervene_paper_clamp_normalised START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_normalised \
  --archs "${PRESENT[@]}" --seed 42 >> $LOG 2>&1
echo "=== intervene_paper_clamp_normalised DONE $(date -u) ===" >> $LOG

# Step 4 — grade_with_sonnet (API-parallel)
echo "=== grade_with_sonnet START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.grade_with_sonnet \
  --archs "${PRESENT[@]}" \
  --subdir steering_paper_normalised >> $LOG 2>&1
echo "=== grade_with_sonnet DONE $(date -u) ===" >> $LOG

echo "=== Eval driver ALL DONE $(date -u) ===" >> $LOG
