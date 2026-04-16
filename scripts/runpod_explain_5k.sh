#!/bin/bash
# runpod_explain_5k.sh — Autointerp EXPLAIN phase: read scan-phase
# feat_*.json files and fill in explanations via Claude Haiku 4.5.
# RunPod version of trillium_explain_5k.sh — no login-node restriction.
#
# Resume-safe: features that already have non-empty explanations are
# skipped. Cost: ~$0.004 per feature × pending count.
#
# Usage:
#   bash scripts/runpod_explain_5k.sh
#
# Then:
#   bash scripts/runpod_fmap_5k.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
source scripts/runpod_activate.sh

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "FAIL: ANTHROPIC_API_KEY not set. Edit .env and re-source runpod_activate.sh."
    exit 1
fi

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
    echo ">> [$LABEL] explain phase ($PENDING / $TOTAL pending)"
    echo "   est cost: ~\$$(python3 -c "print(f'{$PENDING * 0.004:.2f}')")"
    python -m temporal_crosscoders.NLP.autointerp \
        --phase explain \
        --label "$LABEL" \
        --output-dir "$REPORT_DIR" \
        --explain-model claude-haiku-4-5-20251001 \
        --no-harm
}

echo "=== explain 5k (runpod) ==="

run_explain step1-unshuffled reports/step1-gemma-replication
run_explain step1-shuffled   reports/step1-gemma-replication
run_explain step2-unshuffled reports/step2-deepseek-reasoning
run_explain step2-shuffled   reports/step2-deepseek-reasoning

echo ""
echo "=== explain done ==="
echo "Next: bash scripts/runpod_fmap_5k.sh"
