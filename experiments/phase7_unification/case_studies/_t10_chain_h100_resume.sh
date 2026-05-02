#!/bin/bash
# H100 resume — runs the 3 remaining deadzone-escape archs (steps 5-7).
# No PID-polling: starts immediately. Steps 0-4 should already have ckpts
# pulled from HF before running this.
#
# Pre-conditions on H100 pod:
#   1. /workspace tokens restored: GH_TOKEN HF_TOKEN ANTHROPIC_API_KEY
#      bash /workspace/temp_xc/scripts/runpod_phase7_bootstrap.sh
#   2. uv sync done (via restart_recovery.sh)
#   3. Activation cache pulled from HF data-repo:
#      .venv/bin/python -c "from huggingface_hub import snapshot_download; \\
#          snapshot_download(repo_id='han1823123123/txcdr-base-data', \\
#          repo_type='dataset', local_dir='/workspace/temp_xc/data/cached_activations/gemma-2-2b/fineweb', \\
#          allow_patterns='activation_cache/*')"
#      Then mv contents up one level so layer_specs.json + resid_L12.npy + token_ids.npy are direct children.
#   4. Trained T=10 ckpts pulled from HF ckpt-repo (5 archs already trained on A40).
#
# Outputs to /tmp/h100_chain.log; archs auto-push to HF (no --no-hf-push).
set -e
cd /workspace/temp_xc
export TQDM_DISABLE=1

LOG=/tmp/h100_chain.log
echo "=== H100 chain resume started $(date -u) ===" >> $LOG

# Step 5 — SpatialMatryH8 nested uniform
echo "=== [5/7] SpatialMatryH8 T=10 shifts=(2,) nested uniform START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \
  --T 10 --shifts 2 --seed 42 --subset-mode uniform --nested >> $LOG 2>&1
echo "=== [5/7] DONE $(date -u) ===" >> $LOG

# Step 6 — SpatialMatryH8 indep gaussian
echo "=== [6/7] SpatialMatryH8 T=10 shifts=(2,) indep gaussian START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \
  --T 10 --shifts 2 --seed 42 \
  --subset-mode gaussian --sigma-lo 1.5 --sigma-hi 3.0 --n-gaussians 2 >> $LOG 2>&1
echo "=== [6/7] DONE $(date -u) ===" >> $LOG

# Step 7 — SpatialMatryH8 nested gaussian
echo "=== [7/7] SpatialMatryH8 T=10 shifts=(2,) nested gaussian START $(date -u) ===" >> $LOG
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \
  --T 10 --shifts 2 --seed 42 \
  --subset-mode gaussian --sigma-lo 1.5 --sigma-hi 3.0 --n-gaussians 2 --nested >> $LOG 2>&1
echo "=== [7/7] DONE $(date -u) ===" >> $LOG

echo "=== H100 chain ALL DONE $(date -u) ===" >> $LOG
