#!/bin/bash
# Sequential IT-side training + probing pipeline.
# Run from repo root.
#
# Stages:
#  1. Train seed 42 (9 A40_ok archs, ~5.5 hr)
#  2. Probe seed 42 ckpts on IT probe cache (~30 min)
#  3. Train seed 1 (~5.5 hr)
#  4. Probe seed 1 ckpts (~30 min)
#  5. Re-render leaderboard
#
# Stops on first non-zero exit (set -e). Each stage logs to its own
# file under logs/.
#
# To resume after a partial failure, re-launch with stages already
# done short-circuiting via the runtime's "ckpt exists, skip training"
# logic in train_phase7 (NOT IMPLEMENTED yet — manual remove of
# completed archs from it_a40_ok_archs.txt is the workaround).

set -e

cd /workspace/temp_xc

export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export HF_HOME=/workspace/hf_cache
export UV_LINK_MODE=copy

ARCHS=experiments/phase7_unification/it_a40_ok_archs.txt
CACHE_DIR=experiments/phase7_unification/results/probe_cache_S32_it
SUBJ=google/gemma-2-2b-it

mkdir -p /workspace/temp_xc/logs

echo "=============================="
echo "Stage 1: train seed 42"
echo "=============================="
.venv/bin/python -m experiments.phase7_unification.train_phase7_it \
  --canonical --seed 42 --archs "$ARCHS" \
  2>&1 | tee /workspace/temp_xc/logs/train_it_seed42.log

echo ""
echo "=============================="
echo "Stage 2: probe seed 42 ckpts"
echo "=============================="
.venv/bin/python -m experiments.phase7_unification.run_probing_phase7 \
  --headline \
  --probe_cache_dir "$CACHE_DIR" \
  --subject_model_filter "$SUBJ" \
  2>&1 | tee /workspace/temp_xc/logs/probe_it_seed42.log

echo ""
echo "=============================="
echo "Stage 3: train seed 1"
echo "=============================="
.venv/bin/python -m experiments.phase7_unification.train_phase7_it \
  --canonical --seed 1 --archs "$ARCHS" \
  2>&1 | tee /workspace/temp_xc/logs/train_it_seed1.log

echo ""
echo "=============================="
echo "Stage 4: probe seed 1 ckpts"
echo "=============================="
.venv/bin/python -m experiments.phase7_unification.run_probing_phase7 \
  --headline \
  --probe_cache_dir "$CACHE_DIR" \
  --subject_model_filter "$SUBJ" \
  2>&1 | tee /workspace/temp_xc/logs/probe_it_seed1.log

echo ""
echo "=============================="
echo "Stage 5: render leaderboard"
echo "=============================="
.venv/bin/python -m experiments.phase7_unification.build_leaderboard_2seed \
  --subject-model "$SUBJ" \
  2>&1 | tee /workspace/temp_xc/logs/leaderboard_it.log

echo ""
echo "=== IT pipeline complete ==="
