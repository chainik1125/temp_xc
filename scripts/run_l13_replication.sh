#!/bin/bash
# Replication sweep scaffold: resid_L13 on Gemma-2-2B-IT, k=50, unshuffled.
# Trains Stacked + Crosscoder + TFA-pos on already-cached L13 activations.
# Expected runtime on A40: 2-3h for all three arches.
#
# This script only does TRAINING. The analysis pipeline
# (scan_features, temporal_spread, tfa_pred_novel_split, feature_match,
# high_span_comparison, content_based_match, explain_features,
# plot_autointerp_summary) is hardcoded to resid_L25 k=50 in several
# places. To run against L13 you must:
#   1. Let this script finish → produces ckpts at
#      results/nlp_sweep/gemma/ckpts/*__resid_L13__k50__seed42.pt
#   2. Refactor each analysis script to accept --layer-key resid_L13
#      (and/or duplicate the hard-coded constants in
#      plot_autointerp_summary.py and high_span_comparison.py).
#   3. Then rerun the pipeline against the new ckpts.

set -euo pipefail

export TQDM_DISABLE=1
export PYTHONPATH=/workspace/temp_xc
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

PYTHON="${PYTHON:-/workspace/temp_xc/.venv/bin/python}"
RESULTS_DIR="results/nlp_sweep"

mkdir -p "$RESULTS_DIR/gemma" logs

echo "============================================================"
echo "  L13 REPLICATION SWEEP — $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

$PYTHON -u -m src.bench.sweep \
    --dataset-type cached_activations \
    --model-name gemma-2-2b-it \
    --cached-dataset fineweb \
    --cached-layer-key resid_L13 \
    --models tfa_pos stacked_sae crosscoder \
    --k 50 \
    --T 5 \
    --steps 10000 \
    --expansion-factor 8 \
    --tfa-bottleneck-factor 8 \
    --tfa-batch-size 16 \
    --results-dir "${RESULTS_DIR}/gemma"

echo ""
echo "============================================================"
echo "  TRAINING COMPLETE — $(date)"
echo "  Next: refactor analysis scripts to --layer-key resid_L13"
echo "  See scripts/run_l13_replication.sh header for details."
echo "============================================================"
