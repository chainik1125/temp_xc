#!/bin/bash
# run_ablation.sh — one-shot orchestrator.
#
# Uses the Python interpreter currently on PATH. Dmitry's pipeline
# requires Python ≥ 3.12 (simplexity dep); our repo-root .venv is 3.11,
# so DO NOT use the repo-root venv for this experiment. Instead:
#
#   cd experiments/mess3_mat_ablation
#   uv venv --python 3.12 .venv
#   source .venv/bin/activate
#   uv pip install torch numpy scikit-learn matplotlib pyyaml transformer_lens \
#       "jax[cpu]" jaxlib \
#       "git+https://github.com/Astera-org/simplexity.git@review/nonergodic-pr172"
#   bash run_ablation.sh
#
# Steps:
#   1. Vendor Dmitry's sae_day code via git archive (./setup_vendor.sh).
#   2. Run the 5δ × 4arch ablation through his driver + our matsae patch.
#   3. Compute Bayes-optimal R²_max per δ (mirrors his compute_r2_ceiling.py).
#   4. Plot gap-recovery (R²/R²_max) to plots/*.pdf.
#   5. Copy plots into docs/aniket/experiments/mess3_mat_ablation/plots/.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Resolve the Python interpreter. Priority:
#   1. MESS3_PYTHON env override (explicit)
#   2. `python` from PATH — honors whatever venv is currently activated
#   3. hard fail (don't silently use /usr/bin/python3 which has no deps)
if [[ -n "${MESS3_PYTHON:-}" ]]; then
    PY="$MESS3_PYTHON"
elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
else
    echo "[error] no python found on PATH; activate the mess3 venv first" >&2
    exit 2
fi
echo "[info] python=$PY"
$PY -c "import sys; assert sys.version_info >= (3,12), f'python {sys.version_info[:2]} < 3.12; simplexity wants ≥3.12'" || {
    echo "[error] simplexity requires Python ≥ 3.12. Create a 3.12 venv — see header comment." >&2
    exit 2
}

# Step 1: vendor Dmitry's code if not already present.
if [[ ! -f vendor/COMMIT_SHA ]]; then
    echo "[info] stage=setup_vendor | status=start"
    bash setup_vendor.sh
else
    echo "[info] stage=setup_vendor | status=cached | pin=$(head -1 vendor/COMMIT_SHA)"
fi

# Step 2: verify simplexity importable.
if ! $PY -c "import simplexity" >/dev/null 2>&1; then
    echo "[error] simplexity not importable from $PY. Install with:" >&2
    echo "    uv pip install 'git+https://github.com/Astera-org/simplexity.git@review/nonergodic-pr172'" >&2
    exit 2
fi

# Step 3: run driver with our matsae dispatch.
echo "[info] stage=run_driver | status=start | config=config_ablation.yaml"
export PYTHONPATH="$HERE:$HERE/vendor/src:$HERE/vendor/experiments/standard_hmm:$HERE/vendor/experiments/transformer_standard_hmm:$HERE/vendor/experiments/transformer_nonergodic"
$PY run_ablation.py --config config_ablation.yaml 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M).log

# Step 4: Bayes ceiling — reuse Dmitry's compute_r2_ceiling as-is.
# His script lives one level up from his vendor/, but we don't have that
# structure. Skip for now; plot_ablation.py falls back to raw-R² axis
# when r2_ceiling.json is absent.
echo "[info] stage=compute_r2_ceiling | status=skipped (plot falls back to raw-R² axis)"

# Step 5: plots.
echo "[info] stage=plot | status=start"
$PY plot_ablation.py --results-root results --out-dir plots

# Step 6: copy final PDFs into docs for the writeup.
DOCS_PLOTS="$REPO_ROOT/docs/aniket/experiments/mess3_mat_ablation/plots"
mkdir -p "$DOCS_PLOTS"
cp -v plots/*.pdf "$DOCS_PLOTS/" 2>/dev/null || echo "[warn] no plots produced"

echo "[done] mess3_mat_ablation complete."
echo "  JSONs:   $HERE/results/"
echo "  Plots:   $DOCS_PLOTS/"
echo "  Summary doc to be written: $REPO_ROOT/docs/aniket/experiments/mess3_mat_ablation/summary.md"
