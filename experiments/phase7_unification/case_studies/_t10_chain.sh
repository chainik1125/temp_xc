#!/bin/bash
# Chain: train deadzone-escape T=10 variants after current T=10 shifts=(10,) finishes.
#
# Order (cheapest/most-direct first; abort propagates via set -e):
#   1. T=10 H8 shifts=(2,)                    — isolate contrastive-strength lever
#   2. SubseqH8  T_max=10 t_sample=5 shifts=(2,) contiguous — Han's chunk hypothesis
#   3. SubseqH8  T_max=10 t_sample=5 shifts=(2,) gaussian   — Han's spatial-locality hypothesis
#   4. SpatialMatryoshkaH8 T=10 shifts=(2,) indep uniform   — random-subset Matryoshka
#   5. SpatialMatryoshkaH8 T=10 shifts=(2,) nested uniform  — nested random-subset Matryoshka
#   6. SpatialMatryoshkaH8 T=10 shifts=(2,) indep gaussian  — Gaussian-splat random-subset
#   7. SpatialMatryoshkaH8 T=10 shifts=(2,) nested gaussian — combined
#
# Each step appends its log to /tmp/t10_chain.log; we tail status periodically.
set -e
cd /workspace/temp_xc
export TQDM_DISABLE=1

LOG=/tmp/t10_chain.log
echo "=== Chain started $(date -u) ===" >> $LOG

# Wait for current T=10 shifts=(10,) training to finish
echo "=== Waiting for current T=10 shifts=(10,) training (PID 138895) to finish ===" >> $LOG
while kill -0 138895 2>/dev/null; do
  sleep 30
done
echo "=== T=10 shifts=(10,) finished $(date -u) ===" >> $LOG

# 1. T=10 H8 shifts=(2,) — weaker contrastive constraint
echo "=== [1/7] T=10 H8 shifts=(2,) sd=42 START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_h8_shifts \
  --T 10 --shifts 2 --seed 42 --no-hf-push >> $LOG 2>&1
echo "=== [1/7] DONE $(date -u) ===" >> $LOG

# 2. SubseqH8 contiguous — random-chunk
echo "=== [2/7] SubseqH8 t_max=10 t_samp=5 shifts=(2,) contiguous START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_subseq_h8 \
  --T-max 10 --t-sample 5 --shifts 2 --seed 42 --sampling-mode contiguous --no-hf-push >> $LOG 2>&1
echo "=== [2/7] DONE $(date -u) ===" >> $LOG

# 3. SubseqH8 gaussian — Han's spatial-locality hypothesis
echo "=== [3/7] SubseqH8 t_max=10 t_samp=5 shifts=(2,) gaussian START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_subseq_h8 \
  --T-max 10 --t-sample 5 --shifts 2 --seed 42 \
  --sampling-mode gaussian --sigma-lo 1.5 --sigma-hi 3.0 --n-gaussians 2 --no-hf-push >> $LOG 2>&1
echo "=== [3/7] DONE $(date -u) ===" >> $LOG

# 4. SpatialMatryoshkaH8 indep uniform
echo "=== [4/7] SpatialMatryH8 T=10 shifts=(2,) indep uniform START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \
  --T 10 --shifts 2 --seed 42 --subset-mode uniform --no-hf-push >> $LOG 2>&1
echo "=== [4/7] DONE $(date -u) ===" >> $LOG

# 5. SpatialMatryoshkaH8 nested uniform
echo "=== [5/7] SpatialMatryH8 T=10 shifts=(2,) nested uniform START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \
  --T 10 --shifts 2 --seed 42 --subset-mode uniform --nested --no-hf-push >> $LOG 2>&1
echo "=== [5/7] DONE $(date -u) ===" >> $LOG

# 6. SpatialMatryoshkaH8 indep gaussian
echo "=== [6/7] SpatialMatryH8 T=10 shifts=(2,) indep gaussian START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \
  --T 10 --shifts 2 --seed 42 \
  --subset-mode gaussian --sigma-lo 1.5 --sigma-hi 3.0 --n-gaussians 2 --no-hf-push >> $LOG 2>&1
echo "=== [6/7] DONE $(date -u) ===" >> $LOG

# 7. SpatialMatryoshkaH8 nested gaussian
echo "=== [7/7] SpatialMatryH8 T=10 shifts=(2,) nested gaussian START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \
  --T 10 --shifts 2 --seed 42 \
  --subset-mode gaussian --sigma-lo 1.5 --sigma-hi 3.0 --n-gaussians 2 --nested --no-hf-push >> $LOG 2>&1
echo "=== [7/7] DONE $(date -u) ===" >> $LOG

echo "=== Chain ALL DONE $(date -u) ===" >> $LOG
