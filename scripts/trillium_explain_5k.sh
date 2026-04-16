#!/bin/bash
# trillium_explain_5k.sh — Run the autointerp EXPLAIN phase on ~5000 features
# per checkpoint. Login-node only (needs outbound HTTPS to api.anthropic.com).
#
# This reads the feat_*.json files written by trillium_scan_5k_sbatch.sh,
# filters to those with empty explanation fields, and calls Claude Haiku for
# each. With 5000 features per checkpoint × 4 checkpoints = 20,000 Claude
# calls total.
#
# Cost estimate (Claude Haiku 4.5, ~$1/MTok in, $5/MTok out):
#   per feature: ~2k input + ~500 output tokens ≈ $0.004
#   per checkpoint: 5000 × $0.004 ≈ $20
#   total (4 checkpoints): ~$80
#
#   BUT: ResultsStore has resume support — features already explained
#   (from the earlier 30-feature run) are skipped. So the actual cost is
#   (5000 - 30) × 4 × $0.004 ≈ $79. Round to ~$80.
#
#   If this is too expensive, set TOP_FEATURES=500 in the scan script
#   and re-scan first. 500 features = ~$8 total.
#
# The explain phase uses ~zero CPU (just sits waiting on HTTPS responses)
# so it fits within the login node's CPU-time cap as long as the network
# calls don't trigger MKL parallelism. If it gets killed anyway, re-run
# and it resumes from where it left off (ResultsStore skips existing).
#
# Usage:
#   bash scripts/trillium_explain_5k.sh
#
# Then:
#   bash scripts/trillium_sbatch_fmap_5k.sh

set -euo pipefail

HOST=$(hostname)
if [[ "$HOST" != trig-login* ]]; then
    echo "FAIL: run this from the login node (you are on '$HOST')."
    exit 1
fi

source "$SCRATCH/temp_xc/scripts/trillium_activate.sh"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "FAIL: ANTHROPIC_API_KEY not set in environment."
    exit 2
fi

cd "$SCRATCH/temp_xc"

# Limit Python's thread pool to 1 to minimize CPU-time consumption.
# The explain phase is I/O-bound (waiting on HTTPS), not CPU-bound.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

run_explain() {
    local LABEL="$1"
    local REPORT_DIR="$2"

    local RESULTS="$REPORT_DIR/autointerp/$LABEL"
    if [ ! -d "$RESULTS" ]; then
        echo ">> SKIP $LABEL — no scan output: $RESULTS"
        return 0
    fi

    local TOTAL
    TOTAL=$(find "$RESULTS" -name 'feat_*.json' | wc -l)
    local PENDING
    PENDING=$(python3 -c "
import json, pathlib
d = pathlib.Path('$RESULTS')
n = sum(1 for f in d.glob('feat_*.json')
        if not json.loads(f.read_text()).get('explanation'))
print(n)
" 2>/dev/null || echo "$TOTAL")

    if [ "$PENDING" -eq 0 ]; then
        echo ">> [$LABEL] all $TOTAL features already explained — skipping"
        return 0
    fi

    echo ""
    echo ">> [$LABEL] explain phase ($PENDING / $TOTAL features pending)"
    echo "   est cost: ~\$$(python3 -c "print(f'{$PENDING * 0.004:.2f}')")"
    python -m temporal_crosscoders.NLP.autointerp \
        --phase explain \
        --label "$LABEL" \
        --output-dir "$REPORT_DIR" \
        --explain-model claude-haiku-4-5-20251001 \
        --no-harm
}

echo "=== explain 5k (login-node) ==="
echo "  OMP/MKL threads pinned to 1 to minimize CPU-time usage"
echo ""

run_explain step1-unshuffled reports/step1-gemma-replication
run_explain step1-shuffled   reports/step1-gemma-replication
run_explain step2-unshuffled reports/step2-deepseek-reasoning
run_explain step2-shuffled   reports/step2-deepseek-reasoning

echo ""
echo "=== explain done ==="
echo "Next: bash scripts/trillium_sbatch_fmap_5k.sh"
