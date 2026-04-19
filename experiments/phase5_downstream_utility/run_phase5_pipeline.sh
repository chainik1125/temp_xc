#!/bin/bash
# Full Phase 5.1 pipeline launcher.
#
# Assumes the MLC-layer cache is complete (L11, L12, L13, L14, L15 present
# in data/cached_activations/gemma-2-2b-it/fineweb/).
#
# Pipeline:
#   1. Build per-task probing activation cache (~15 min).
#   2. Train all 10 primary architectures at seed 42 (~2-3 h).
#   3. Run sparse-probing + baselines over all ckpts (~30-60 min).
#   4. Aggregate + make headline plots.
#
# Logs stream to /workspace/temp_xc/logs/phase5_*.log.

set -e
cd /workspace/temp_xc
export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export UV_LINK_MODE=copy
export PYTHONPATH=/workspace/temp_xc

echo "=== [1/4] BUILD PROBE ACTIVATION CACHE (+ cross-token for 5.4) ==="
.venv/bin/python experiments/phase5_downstream_utility/probing/build_probe_cache.py \
    --include-crosstoken \
    2>&1 | tee logs/phase5_probe_cache.log

echo "=== [2/4] TRAIN PRIMARY ARCHS (seed 42) ==="
.venv/bin/python experiments/phase5_downstream_utility/train_primary_archs.py \
    --seeds 42 --max-steps 25000 \
    2>&1 | tee logs/phase5_train.log

echo "=== [3/4] RUN SPARSE PROBING ==="
.venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
    2>&1 | tee logs/phase5_probing.log

echo "=== [4/4] AGGREGATE + PLOT ==="
.venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py \
    2>&1 | tee logs/phase5_plots.log

echo "=== PIPELINE COMPLETE ==="
ls -lh experiments/phase5_downstream_utility/results/plots/
