#!/usr/bin/env bash
# Stage B v2 pipeline — extends the original paper-budget pipeline with:
#   - Sonnet 4.6 0-3 coherence grader (replaces max-run-≤-2 floor that
#     missed sentence-level "Wait, I'm not. / Wait, I'm not." degeneration)
#   - rank ∈ {meandiff, tstat, ratio} as a new hill-climb axis
#   - hill-climb optimises primary_kw_at_sonnet_coh, not primary_kw_at_coh
#
# Assumes the v1 pipeline has already run cache + train + mine + B1 + B2 for
# all 12 sweep cells and at least the iter-1/iter-2 hill-climb cells.
# Idempotent — Sonnet grades are resumable; hill-climb state can be reset.

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="experiments.ward_backtracking_txc"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
echo "[v2-pipeline] NUM_GPUS=${NUM_GPUS}"

# ─── Phase A: Sonnet-grade all existing per-cell B1 generations ─────────
echo
echo "================================================================"
echo "  PHASE A — Sonnet 4.6 coherence grader on existing B1 data"
echo "================================================================"
python -m $ROOT.regrade_cells --grade --concurrency 12

# Also grade the canonical merged B1 JSON (so rank_phase1 sees Sonnet too)
echo
echo "[v2-pipeline] grading canonical b1_steering_results.json"
python -m $ROOT.grade_sonnet \
    --b1-json results/ward_backtracking_txc/steering/b1_steering_results.json \
    --concurrency 12

# ─── Phase B: re-rank Phase 1 sweep cells under Sonnet metric ───────────
echo
echo "================================================================"
echo "  PHASE B — re-rank Phase 1 cells by Sonnet coherence floor"
echo "================================================================"
python -m $ROOT.rank_phase1 \
    --metric-key primary_kw_at_sonnet_coh \
    --out-name rank_phase1_sonnet.json

# ─── Phase C: reset hillclimb state to seed from Sonnet winner ──────────
echo
echo "================================================================"
echo "  PHASE C — reset hill-climb state to seed from Sonnet winner"
echo "================================================================"
python -c "
import json
from pathlib import Path
ranks = json.loads(Path('results/ward_backtracking_txc/rank_phase1_sonnet.json').read_text())
state = {'current_best': None, 'iteration': 0, 'history': [], 'evaluated': {}}
sp = Path('results/ward_backtracking_txc/hillclimb_state_v2.json')
sp.write_text(json.dumps(state, indent=2))
print(f'[reset] sonnet winner = {ranks[\"winner_cell_id\"]} primary_sonnet={ranks[\"winner_metric\"].get(\"primary_kw_at_sonnet_coh\", 0):.4f}')
"

# ─── Phase D: greedy hill-climb under Sonnet metric ─────────────────────
echo
echo "================================================================"
echo "  PHASE D — greedy hill-climb (5-axis) under Sonnet metric"
echo "================================================================"
# Use a sibling state file to keep v1's state intact for the writeup.
HILL_STATE_v2="results/ward_backtracking_txc/hillclimb_state_v2.json"
# The hill_climb.py module uses paths.root/hillclimb_state.json by convention;
# rename around it so we don't trample v1.
cp -f results/ward_backtracking_txc/hillclimb_state.json \
      results/ward_backtracking_txc/hillclimb_state_v1_legacy.json 2>/dev/null || true
cp -f $HILL_STATE_v2 results/ward_backtracking_txc/hillclimb_state.json

python -m $ROOT.hill_climb \
    --num-gpus "${NUM_GPUS}" \
    --max-iter "${MAX_ITER:-4}" \
    --improvement-threshold "${IMPROVEMENT_THRESHOLD:-0.05}" \
    --start-from "$(python -c "import json; print(json.load(open('results/ward_backtracking_txc/rank_phase1_sonnet.json'))['winner_cell_id'])")" \
    --metric-key primary_kw_at_sonnet_coh

# Save the v2 state separately
cp -f results/ward_backtracking_txc/hillclimb_state.json $HILL_STATE_v2

# ─── Phase E: regrade all NEW cells the hill-climb produced ─────────────
echo
echo "================================================================"
echo "  PHASE E — Sonnet-grade any new hill-climb cells"
echo "================================================================"
python -m $ROOT.regrade_cells --grade --concurrency 12

# ─── Phase F: refresh plots with the v2 state ──────────────────────────
echo
echo "================================================================"
echo "  PHASE F — refresh plots"
echo "================================================================"
python -m $ROOT.plot.steering_comparison_bars
python -m $ROOT.plot.coherence
python -m $ROOT.plot.feature_firing_heatmap
python -m $ROOT.plot.per_offset_firing
python -m $ROOT.plot.cosine_matrix
python -m $ROOT.plot.b2_difference_area
python -m $ROOT.plot.text_examples

echo
echo "[v2-pipeline] DONE."
echo "  v1 state:      results/ward_backtracking_txc/hillclimb_state_v1_legacy.json"
echo "  v2 state:      results/ward_backtracking_txc/hillclimb_state_v2.json"
echo "  Sonnet rank:   results/ward_backtracking_txc/rank_phase1_sonnet.json"
echo "  v2 winner:     $(python -c "import json; s=json.load(open('results/ward_backtracking_txc/hillclimb_state_v2.json')); print(s['current_best']['cell_id'] if s['current_best'] else 'none')")"
