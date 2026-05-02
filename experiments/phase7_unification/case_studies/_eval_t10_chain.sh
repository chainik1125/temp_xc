#!/bin/bash
# Post-chain eval driver.
#
# Polls for chain script (PID 142153) to finish, then runs the standard
# steering pipeline on every new T=10 ckpt produced by the chain:
#   select_features → diagnose_z_magnitudes → intervene_paper_clamp_normalised → grade
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

for arch in "${ARCHS[@]}"; do
  ckpt=/workspace/temp_xc/experiments/phase7_unification/results/ckpts/${arch}__seed42.pt
  if [ ! -f "$ckpt" ]; then
    echo "=== [$arch] SKIP — ckpt not found at $ckpt ===" >> $LOG
    continue
  fi
  echo "=== [$arch] START $(date -u) ===" >> $LOG

  echo "--- select_features ---" >> $LOG
  .venv/bin/python -m experiments.phase7_unification.case_studies.steering.select_features \
    --archs "$arch" --seed 42 >> $LOG 2>&1 || { echo "[$arch] select_features FAILED" >> $LOG; continue; }

  echo "--- diagnose_z_magnitudes ---" >> $LOG
  .venv/bin/python -m experiments.phase7_unification.case_studies.steering.diagnose_z_magnitudes \
    --archs "$arch" --seed 42 >> $LOG 2>&1 || { echo "[$arch] diagnose FAILED" >> $LOG; continue; }

  echo "--- intervene_paper_clamp_normalised ---" >> $LOG
  .venv/bin/python -m experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_normalised \
    --archs "$arch" --seed 42 >> $LOG 2>&1 || { echo "[$arch] intervene FAILED" >> $LOG; continue; }

  echo "--- grade_with_sonnet ---" >> $LOG
  .venv/bin/python -m experiments.phase7_unification.case_studies.steering.grade_with_sonnet \
    --archs "$arch" \
    --subdir steering_paper_normalised >> $LOG 2>&1 || { echo "[$arch] grade FAILED" >> $LOG; continue; }

  echo "=== [$arch] DONE $(date -u) ===" >> $LOG
done

echo "=== Eval driver ALL DONE $(date -u) ===" >> $LOG
