#!/bin/bash
# trillium_explain.sh — Run the autointerp EXPLAIN phase on all 4 autointerp
# dirs produced by trillium_scan_sbatch.sh. Reads existing feat_*.json files
# and calls Claude to fill in the explanation field. Must run on the LOGIN
# NODE (needs outbound HTTPS to api.anthropic.com). Uses ~zero CPU — just
# waits on network — so it fits well inside the login CPU-time cap.
#
# Cost: 30 features per label × 4 labels × (~2k in + ~500 out tokens) ≈ $0.56
# with Claude Haiku 4.5.
#
# Usage:
#   bash scripts/trillium_explain.sh
#
# Then:
#   bash scripts/trillium_sbatch_fmap_labeled.sh

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

run_explain() {
    local LABEL="$1"
    local REPORT_DIR="$2"

    local RESULTS="$REPORT_DIR/autointerp/$LABEL"
    if [ ! -d "$RESULTS" ]; then
        echo ">> SKIP $LABEL — no scan output: $RESULTS"
        return 0
    fi

    local COUNT
    COUNT=$(find "$RESULTS" -name 'feat_*.json' | wc -l)
    if [ "$COUNT" -eq 0 ]; then
        echo ">> SKIP $LABEL — no feat_*.json files in $RESULTS"
        return 0
    fi

    echo ""
    echo ">> [$LABEL] explain phase ($COUNT features)"
    python -m temporal_crosscoders.NLP.autointerp \
        --phase explain \
        --label "$LABEL" \
        --output-dir "$REPORT_DIR" \
        --explain-model claude-haiku-4-5-20251001 \
        --no-harm
}

echo "=== explain phase (login-node) ==="
echo "  cost est: ~\$0.56 total across 4 labels"
echo ""

run_explain step1-unshuffled reports/step1-gemma-replication
run_explain step1-shuffled   reports/step1-gemma-replication
run_explain step2-unshuffled reports/step2-deepseek-reasoning
run_explain step2-shuffled   reports/step2-deepseek-reasoning

echo ""
echo "=== explain done ==="
echo "Next: bash scripts/trillium_sbatch_fmap_labeled.sh"
