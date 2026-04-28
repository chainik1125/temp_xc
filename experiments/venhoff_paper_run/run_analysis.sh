#!/bin/bash
# One-shot wrapper around analyze_grid.py for the Venhoff paper-budget run.
#
# Reads the per-arch benchmark JSONs that the for-loop sweep
# (`for arch in sae tempxc mlc; do ... cp benchmark_results...{arch}.json`)
# saved into vendor/.../hybrid/results/, computes per-cell Gap Recovery
# from per-task model answers (re-graded via math_verify), and emits:
#
#   results/summary.csv             — best-cell GR per arch + Venhoff baseline
#   results/gap_recovery_heatmap.png — (coef × window) heatmap per arch
#   results/analysis.json           — full structured dump for notebooks
#
# Usage (scans both single-cell and grid runs by default):
#   bash experiments/venhoff_paper_run/run_analysis.sh
#
# Override the set of filename suffixes scanned. Use '' for the bare
# (single-cell) JSONs and '_grid' for the grid-guardrail JSONs:
#   SUFFIXES="_grid" bash experiments/venhoff_paper_run/run_analysis.sh
#
# Override JSON source dir:
#   JSON_DIR=/some/other/path bash experiments/venhoff_paper_run/run_analysis.sh

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

JSON_DIR="${JSON_DIR:-/workspace/spar-temporal-crosscoders/vendor/thinking-llms-interp/hybrid/results}"
OUT_DIR="${OUT_DIR:-experiments/venhoff_paper_run/results}"
SUFFIXES="${SUFFIXES:-"" _grid}"
BASE_SHORT="${BASE_SHORT:-llama-3.1-8b}"
DATASET="${DATASET:-math500}"

PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    PY="python"
fi

# shellcheck disable=SC2086
"$PY" experiments/venhoff_paper_run/analyze_grid.py \
    --json-dir "$JSON_DIR" \
    --out-dir "$OUT_DIR" \
    --base-short "$BASE_SHORT" \
    --dataset "$DATASET" \
    --suffixes $SUFFIXES
