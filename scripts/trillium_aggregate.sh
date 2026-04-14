#!/bin/bash
# trillium_aggregate.sh — Run the Slack-ready report generator over one or
# more results dirs. No GPU needed.
#
#   bash scripts/trillium_aggregate.sh                                      # results/ → reports/latest/
#   bash scripts/trillium_aggregate.sh results/nlp/deepseek-gsm8k reports/day1/
set -euo pipefail

ROOT="${1:-results/}"
OUT="${2:-reports/latest/}"

cd "$SCRATCH/temp_xc"
source scripts/trillium_activate.sh

python scripts/aggregate_results.py --root "$ROOT" --out "$OUT"
echo ""
echo "Report: $OUT/report.md"
ls -lh "$OUT"
