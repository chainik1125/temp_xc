#!/usr/bin/env bash
# Chain: build probe_cache → rebuild S32 → run probing.
# Triggered MANUALLY after the seed=1 recovery training finishes (i.e., when
# logs/seed1_recovery.log contains "seed=1 recovery done").
#
# We pull all 36 probing-task datasets through Gemma-2-2b base (L12 anchor +
# L10..L14 stack), cache to disk, then per-example-slice the resulting
# 128-token cache to a left-aligned 32-frame cache (Agent A's URGENT fix).
# Finally run the headline probing pass on every (run_id, task) in the index.
#
# Pulling the fix from `han-phase7-unification` (already merged into this
# branch) means USE_S32_CACHE=True at the top of run_probing_phase7.py, so
# the probing code reads from probe_cache_S32/ automatically.

set -e
cd /workspace/temp_xc
mkdir -p logs

export HF_HOME=/workspace/hf_cache
export HF_TOKEN=$(cat /workspace/.tokens/hf_token)
export TQDM_DISABLE=1
export PHASE7_REPO=/workspace/temp_xc
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

LOG=logs/probing_seed1_chain.log
echo "starting probing chain at $(date -Is)" > "$LOG"

# 0. Sanity: GPU must be empty (recovery training must have ended).
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$GPU_USED" -gt 1000 ]; then
    echo "ERROR: GPU still has $GPU_USED MiB used; recovery training likely still running. Aborting." | tee -a "$LOG"
    exit 2
fi

# 1. Build probe_cache (3800 examples × 36 tasks × Gemma forward).
#    Output: experiments/phase7_unification/results/probe_cache/<task>/{acts_anchor,acts_mlc,acts_mlc_tail}.npz
echo "" >> "$LOG"
echo "=== 1. build_probe_cache_phase7 — at $(date -Is) ===" >> "$LOG"
/workspace/temp_xc/.venv/bin/python -u \
    -m experiments.phase7_unification.build_probe_cache_phase7 \
    >> "$LOG" 2>&1

# 2. Slice 128-tail cache to left-aligned 32-frame cache.
echo "" >> "$LOG"
echo "=== 2. rebuild_probe_cache_s32 — at $(date -Is) ===" >> "$LOG"
/workspace/temp_xc/.venv/bin/python -u \
    -m experiments.phase7_unification.rebuild_probe_cache_s32 \
    >> "$LOG" 2>&1

# 3. Run the headline probing pass — uses USE_S32_CACHE=True automatically.
echo "" >> "$LOG"
echo "=== 3. run_probing_phase7 --headline — at $(date -Is) ===" >> "$LOG"
/workspace/temp_xc/.venv/bin/python -u \
    -m experiments.phase7_unification.run_probing_phase7 --headline \
    >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "probing chain done at $(date -Is)" >> "$LOG"
