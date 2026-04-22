#!/bin/bash
# run_ablation.sh — one-shot orchestrator.
#
# Steps:
#   1. Vendor Dmitry's sae_day code via git archive (./setup_vendor.sh).
#   2. Run the 5δ × 4arch ablation through his driver + our matsae patch.
#   3. Compute Bayes-optimal R²_max per δ (mirrors his compute_r2_ceiling.py).
#   4. Plot gap-recovery (R²/R²_max) to plots/*.pdf.
#   5. Copy plots into docs/aniket/experiments/mess3_mat_ablation/plots/
#      for the writeup.
#
# Usage:
#   bash run_ablation.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Step 1: vendor Dmitry's code if not already present.
if [[ ! -f vendor/COMMIT_SHA ]]; then
    echo "[info] stage=setup_vendor | status=start"
    bash setup_vendor.sh
else
    echo "[info] stage=setup_vendor | status=cached | pin=$(head -1 vendor/COMMIT_SHA)"
fi

# Dmitry's code assumes `simplexity` + deps from his [separation-scaling] extra.
# If unavailable, document the missing extra and stop.
if ! "$REPO_ROOT/.venv/bin/python" -c "import simplexity" >/dev/null 2>&1; then
    echo "[info] simplexity missing — attempting uv sync --extra separation-scaling"
    (cd "$REPO_ROOT" && uv sync --extra separation-scaling)
fi

# Step 2: run driver with our matsae dispatch.
echo "[info] stage=run_driver | status=start | config=config_ablation.yaml"
export PYTHONPATH="$HERE:$HERE/vendor/src:$HERE/vendor/experiments/standard_hmm:$HERE/vendor/experiments/transformer_standard_hmm:$HERE/vendor/experiments/transformer_nonergodic"
"$REPO_ROOT/.venv/bin/python" run_ablation.py --config config_ablation.yaml 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M).log

# Step 3: Bayes ceiling — reuse Dmitry's computer as-is.
echo "[info] stage=compute_r2_ceiling | status=start"
"$REPO_ROOT/.venv/bin/python" -c "
import sys
sys.path.insert(0, '$HERE/vendor/experiments/transformer_nonergodic')
# Dmitry's compute_r2_ceiling.py writes its own r2_ceiling.json. We run it
# against the cells we just generated.
import subprocess
subprocess.run([
    '$REPO_ROOT/.venv/bin/python',
    '$HERE/vendor/../compute_r2_ceiling.py',  # lives under separation_scaling root
], check=False)
" || echo "[warn] r2_ceiling: falling back on manual ceiling computation in plot_ablation.py"

# Step 4: plots.
echo "[info] stage=plot | status=start"
"$REPO_ROOT/.venv/bin/python" plot_ablation.py --results-root results --out-dir plots

# Step 5: copy final PDFs into docs for the writeup.
DOCS_PLOTS="$REPO_ROOT/docs/aniket/experiments/mess3_mat_ablation/plots"
mkdir -p "$DOCS_PLOTS"
cp -v plots/*.pdf "$DOCS_PLOTS/" 2>/dev/null || echo "[warn] no plots produced"

echo "[done] mess3_mat_ablation complete."
echo "  JSONs:   $HERE/results/"
echo "  Plots:   $DOCS_PLOTS/"
echo "  Summary doc to be written: $REPO_ROOT/docs/aniket/experiments/mess3_mat_ablation/summary.md"
