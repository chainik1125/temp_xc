#!/usr/bin/env bash
# Phase 6.2 autoresearch loop. Runs up to 6 sequential cycles.
#
# For now uses a simple priority-ordered selector (C3 highest, then
# C5, C6, C1, C2, C4). A Claude-Sonnet adaptive proposer can replace
# `_select_next_candidate` later.
#
# Usage:
#   bash experiments/phase6_2_autoresearch/run_phase62_loop.sh [MAX_CYCLES]
# Default MAX_CYCLES=6.

set -euo pipefail
cd /workspace/temp_xc
source /workspace/temp_xc/.envrc

MAX_CYCLES="${1:-6}"
RESULTS_DIR="experiments/phase6_2_autoresearch/results"
mkdir -p "$RESULTS_DIR" logs

# Priority-ordered candidate list. Implementation-ready first (C5, C6),
# then the ones that need new arch classes (C1-C4). An agent-based
# proposer would reorder this dynamically based on prior results.
PRIORITY_ORDER=("C5" "C6" "C1" "C2" "C3" "C4")

_already_done() {
  local cid="$1"
  [ -f "$RESULTS_DIR/phase62_results.jsonl" ] || return 1
  grep -q "\"candidate_id\": \"${cid}\"" "$RESULTS_DIR/phase62_results.jsonl" \
    2>/dev/null
}

_next_candidate() {
  for cid in "${PRIORITY_ORDER[@]}"; do
    if ! _already_done "$cid"; then
      # Check if implemented
      local impl
      impl=$(.venv/bin/python - <<EOF
import sys
sys.path.insert(0, 'experiments/phase6_2_autoresearch')
from candidates import by_id
print(int(by_id("$cid").implemented))
EOF
)
      if [ "$impl" = "1" ]; then
        echo "$cid"
        return 0
      else
        echo "[skip] $cid not implemented; move on" >&2
      fi
    fi
  done
  return 1
}

echo "============================================================"
echo "Phase 6.2 autoresearch loop (max ${MAX_CYCLES} cycles)"
echo "============================================================"

for i in $(seq 1 "$MAX_CYCLES"); do
  CID=$(_next_candidate) || { echo "no more candidates to run"; break; }
  echo
  echo "=== cycle $i / $MAX_CYCLES : $CID ==="
  bash experiments/phase6_2_autoresearch/run_phase62_cycle.sh "$CID" \
    > "logs/phase62_cycle${i}_${CID}.log" 2>&1 || {
      echo "[fail] $CID cycle crashed; see logs/phase62_cycle${i}_${CID}.log"
      continue
    }

  # Early-stop: check if any random x/32 ≥ 10 (matches tsae_paper within 2)
  BEST_RANDOM=$(.venv/bin/python - <<EOF
import json, pathlib
best = 0
p = pathlib.Path("$RESULTS_DIR/phase62_results.jsonl")
for line in p.read_text().splitlines() if p.exists() else []:
    d = json.loads(line)
    r = d["metrics"].get("random", {}).get("semantic_count", 0)
    if r > best: best = r
print(best)
EOF
)
  echo "[loop] best random so far: $BEST_RANDOM / 32"
  if [ "$BEST_RANDOM" -ge 10 ]; then
    echo "[loop] early-stop: random target reached ($BEST_RANDOM ≥ 10)"
    break
  fi
done

echo
echo "=== loop DONE: $(date -u) ==="
echo "results: $RESULTS_DIR/phase62_results.jsonl"
.venv/bin/python experiments/phase6_2_autoresearch/summarise_phase62.py \
  2>&1 || echo "(summarise_phase62.py not yet implemented; see jsonl)"
