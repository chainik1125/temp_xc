#!/usr/bin/env bash
# Chain the full safety pipeline. Sources .env for ANTHROPIC_API_KEY (best-
# effort; the Gemma fallback is automatic if the key is missing/invalid).
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

if [[ -f safety_research/.env ]]; then
  set -a; . safety_research/.env; set +a
fi

LOG_DIR=safety_research/results
mkdir -p "$LOG_DIR"

echo "=== [1/4] training ==="
uv run --no-sync python safety_research/scripts/train_three_arms.py \
  2>&1 | tee "$LOG_DIR/train.log"

echo "=== [2/4] autointerp ==="
uv run --no-sync python safety_research/scripts/run_autointerp.py \
  2>&1 | tee "$LOG_DIR/autointerp.log"

echo "=== [3/4] umap meta ==="
uv run --no-sync python safety_research/scripts/umap_meta.py \
  2>&1 | tee "$LOG_DIR/umap.log"

echo "=== [4/4] safety eval ==="
uv run --no-sync python safety_research/scripts/safety_eval.py \
  2>&1 | tee "$LOG_DIR/safety.log"

uv run --no-sync python safety_research/scripts/build_report.py
echo "DONE — see safety_research/REPORT.md"
