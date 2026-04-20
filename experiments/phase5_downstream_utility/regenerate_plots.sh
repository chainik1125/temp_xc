#!/usr/bin/env bash
# Post-probing regeneration: plots + SVD + commit.
# Run after full_window probing completes.
set -u
cd /workspace/temp_xc
export HF_HOME=/workspace/hf_cache
export TQDM_DISABLE=1
export UV_LINK_MODE=copy
export PYTHONPATH=/workspace/temp_xc

LOG=/workspace/temp_xc/logs/overnight/final_plots.log
echo "=== make_headline_plot ===" > "$LOG"
.venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
echo "=== plot_training_curves ===" >> "$LOG"
.venv/bin/python experiments/phase5_downstream_utility/plots/plot_training_curves.py >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
echo "=== analyze_decoder_svd ===" >> "$LOG"
.venv/bin/python experiments/phase5_downstream_utility/analyze_decoder_svd.py >> "$LOG" 2>&1
echo "exit=$?" >> "$LOG"

ls experiments/phase5_downstream_utility/results/plots/ | sort >> "$LOG"
wc -l experiments/phase5_downstream_utility/results/probing_results.jsonl >> "$LOG"
echo "DONE"
