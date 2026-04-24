#!/usr/bin/env bash
# One-shot end-to-end scrappy autoresearch:
#   1. Run Phase 0 bootstrap if not already cached.
#   2. Run the full curated + screening batch via run_autoresearch.sh.
#
# Single bash command to fire an overnight pod. Output ledger lands at
# experiments/venhoff_scrappy/results/autoresearch_index.jsonl; per-cycle
# grade JSON + hybrid outputs land at results/cycles/<cand>/.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SCRAPPY=experiments/venhoff_scrappy
PHASE0_ROOT="$SCRAPPY/results/phase0"

if ! find "$PHASE0_ROOT" -name "activations_path3.pkl" 2>/dev/null | grep -q .; then
    echo "[info] Phase 0 cache empty — running phase0_bootstrap.sh (~20 min)"
    bash scripts/phase0_bootstrap.sh
else
    echo "[info] Phase 0 cache present at $PHASE0_ROOT — skipping bootstrap"
fi

# Curated batch (4 cycles, sets baseline + smoke-tests arch diversity):
CURATED=(baseline_sae baseline_tempxc baseline_mlc)

# Screening factorial: arch × n_clusters (9 cycles; ~90 min at 10 min/cycle):
SCREENING=(
    sae_nclusters5  sae_nclusters10  sae_nclusters15
    tempxc_nclusters5 tempxc_nclusters10 tempxc_nclusters15
    mlc_nclusters5  mlc_nclusters10  mlc_nclusters15
)

bash "$SCRAPPY/run_autoresearch.sh" "${CURATED[@]}" "${SCREENING[@]}"
