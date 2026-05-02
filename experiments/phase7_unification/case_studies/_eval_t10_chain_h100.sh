#!/bin/bash
# H100 eval driver — gates on ckpt presence, NOT on a chain PID.
#
# Runs the steering pipeline ONCE with all 8 deadzone-escape archs as a
# batch. Step 3 intervene runs in N_GROUPS parallel processes (default 5
# for H100; can be lowered to 2 for A40). Each process is independent
# and bit-parity-preserved vs sequential intervene at B=7.
#
# Optional baselines re-eval for apples-to-apples (uncomment block at
# bottom).
set -e
cd /workspace/temp_xc
export TQDM_DISABLE=1

LOG=/tmp/eval_t10_chain_h100.log
echo "=== H100 eval driver started $(date -u) ===" >> $LOG

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

# Filter to archs whose ckpts are present on local disk (post HF-pull).
PRESENT=()
for arch in "${ARCHS[@]}"; do
  ckpt=/workspace/temp_xc/experiments/phase7_unification/results/ckpts/${arch}__seed42.pt
  if [ -f "$ckpt" ]; then
    PRESENT+=("$arch")
  else
    echo "=== [$arch] SKIP — ckpt missing ===" >> $LOG
  fi
done
echo "=== Eval batch: ${#PRESENT[@]} archs present ===" >> $LOG

# Step 1 — select_features (shared Gemma + disk cache)
echo "=== select_features START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.select_features \
  --archs "${PRESENT[@]}" --seed 42 >> $LOG 2>&1
echo "=== select_features DONE $(date -u) ===" >> $LOG

# Step 2 — diagnose_z_magnitudes
echo "=== diagnose_z_magnitudes START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.diagnose_z_magnitudes \
  --archs "${PRESENT[@]}" --seed 42 >> $LOG 2>&1
echo "=== diagnose_z_magnitudes DONE $(date -u) ===" >> $LOG

# Step 3 — intervene_paper_clamp_normalised — multi-process parallel
N_GROUPS=${N_GROUPS:-5}
echo "=== intervene START $(date -u) (N_GROUPS=$N_GROUPS) ===" >> $LOG
N=${#PRESENT[@]}
PER=$(( (N + N_GROUPS - 1) / N_GROUPS ))
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
ALL_OK=1
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    ALL_OK=0
    echo "  intervene PID $pid FAILED" >> $LOG
  fi
done
for ((g=0; g<N_GROUPS; g++)); do
  if [ -f /tmp/intervene_g${g}.log ]; then
    echo "  --- group $g log ---" >> $LOG
    cat /tmp/intervene_g${g}.log >> $LOG
  fi
done
[ $ALL_OK -eq 0 ] && echo "=== intervene PARTIAL $(date -u) ===" >> $LOG \
                  || echo "=== intervene DONE $(date -u) ===" >> $LOG

# Step 4 — grade_with_sonnet (API parallel) for RE/PP
echo "=== grade RE/PP START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.grade_with_sonnet \
  --archs "${PRESENT[@]}" \
  --subdir steering_paper_normalised >> $LOG 2>&1
echo "=== grade RE/PP DONE $(date -u) ===" >> $LOG

# ---- V7 tiled-broadcast pass (sequential, after RE/PP) ----
# Y's commit b42f9770: stride-T non-overlapping blocks, single uniform δ per
# block. Attention-invariant within block — predicted to dominate at T≥3.
# Reuses select_features + z magnitudes from the RE/PP pass above.
Z_MAG=/workspace/temp_xc/experiments/phase7_unification/results/case_studies/diagnostics/z_orig_magnitudes.json

echo "=== intervene V7 START $(date -u) (N_GROUPS=$N_GROUPS) ===" >> $LOG
declare -a PIDS_V7=()
for ((g=0; g<N_GROUPS; g++)); do
  off=$((g * PER))
  if [ $off -ge $N ]; then break; fi
  GROUP=("${PRESENT[@]:$off:$PER}")
  echo "  V7 group $g: ${GROUP[*]}" >> $LOG
  .venv/bin/python -m experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_window_tiled_broadcast \
    --archs "${GROUP[@]}" --seed 42 --normalised --z-mag "$Z_MAG" \
    >> /tmp/intervene_v7_g${g}.log 2>&1 &
  PIDS_V7+=("$!")
done
echo "  V7 spawned ${#PIDS_V7[@]} parallel processes: ${PIDS_V7[*]}" >> $LOG
ALL_OK_V7=1
for pid in "${PIDS_V7[@]}"; do
  if ! wait "$pid"; then
    ALL_OK_V7=0
    echo "  V7 intervene PID $pid FAILED" >> $LOG
  fi
done
for ((g=0; g<N_GROUPS; g++)); do
  if [ -f /tmp/intervene_v7_g${g}.log ]; then
    echo "  --- V7 group $g log ---" >> $LOG
    cat /tmp/intervene_v7_g${g}.log >> $LOG
  fi
done
[ $ALL_OK_V7 -eq 0 ] && echo "=== intervene V7 PARTIAL $(date -u) ===" >> $LOG \
                     || echo "=== intervene V7 DONE $(date -u) ===" >> $LOG

echo "=== grade V7 START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.steering.grade_with_sonnet \
  --archs "${PRESENT[@]}" \
  --subdir steering_paper_window_tiled_broadcast >> $LOG 2>&1
echo "=== grade V7 DONE $(date -u) ===" >> $LOG

echo "=== H100 eval driver ALL DONE $(date -u) ===" >> $LOG

# OPTIONAL — apples-to-apples baseline re-eval on H100. Uncomment to run.
# BASELINES=(
#   "tsae_paper_k20" "topk_sae"
#   "txc_h8_t2_kpos20_shifts2"
#   "txc_maxpool_h8_t2_kpos20_shifts2"
#   "txc_contrastive_h8_t2_kpos20_shifts2"
#   "txc_bare_antidead_t3_kpos20"
#   "agentic_txc_02_kpos20"
# )
# .venv/bin/python -m experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_normalised \
#   --archs "${BASELINES[@]}" --seed 42 --force
# .venv/bin/python -m experiments.phase7_unification.case_studies.steering.grade_with_sonnet \
#   --archs "${BASELINES[@]}" --subdir steering_paper_normalised
