#!/usr/bin/env bash
# v2-extend: re-run hill-climb with the extended arch_list (txc_h8, txc_h13)
# starting from the existing hillclimb_state_v2.json. Preserves all
# previously-evaluated cells; only NEW arch combinations get trained.
#
# Run AFTER run_v2_pipeline.sh completes naturally.
#
# Each new H8/H13 cell costs:
#   - train: ~10 min (similar TopK params; multi-distance forward 1.5× cost)
#   - mine:  ~30 s
#   - B1:    ~30 min (8 sources for window archs)
# So one cell ≈ 40 min wall on H100.
#
# Hill-climb iter 1 from any existing winner will propose 5 new arch swaps.
# Of those, 2 are H8/H13 — fresh evaluation; ~80 min for batch 1 across 2 GPUs.

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="experiments.ward_backtracking_txc"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

# Step 1: ensure ALL cells with B1 results are graded under Sonnet (catches
# lowk TXC cells trained outside the v2 wrapper, hill-climb-evaluated cells
# without grades, etc.). Idempotent — already-graded cells skip via grade_sonnet
# resume logic.
echo "=== v2-extend Phase 0 — Sonnet-grade ALL per-cell B1 files ==="
python -m $ROOT.regrade_cells --grade --concurrency 12

# Step 2: re-rank ALL cells (not just Phase-1 sweep) under Sonnet metric,
# emit a fresh leaderboard that includes lowk TXC + iter-1 hill-climb cells.
# rank_phase1.py was designed for the canonical sweep; we need a broader ranker.
echo "=== v2-extend Phase 0.5 — global Sonnet leaderboard across ALL cell_metrics ==="
python -c "
import json
from pathlib import Path
md = Path('results/ward_backtracking_txc/cell_metrics')
rows = []
for p in sorted(md.glob('*.json')):
    m = json.loads(p.read_text())
    if 'primary_kw_at_sonnet_coh' not in m:
        # Sonnet not graded for this cell yet — skip (treat as 0).
        continue
    rows.append(m)
rows.sort(key=lambda m: -m.get('primary_kw_at_sonnet_coh', 0.0))
print(f'[global-sonnet-rank] {len(rows)} cells with Sonnet metric')
print('Top 8:')
for m in rows[:8]:
    print(f\"  {m['cell_id']:55s} sonnet={m['primary_kw_at_sonnet_coh']:.4f} legacy={m['primary_kw_at_coh']:.4f} mag={m['best_magnitude']:+.0f} frac_coh_s={m.get('frac_coh_sonnet', 0):.2f}\")
out = {'winner_cell_id': rows[0]['cell_id'] if rows else None,
       'winner_metric': rows[0] if rows else None,
       'metric_key': 'primary_kw_at_sonnet_coh',
       'leaderboard': rows,
       'n_cells': len(rows)}
Path('results/ward_backtracking_txc/rank_global_sonnet.json').write_text(json.dumps(out, indent=2))
print(f'[saved] rank_global_sonnet.json — global winner: {out[\"winner_cell_id\"]}')
"

# Step 3: reset hill-climb state to seed from the GLOBAL Sonnet winner.
# Preserve the v2 evaluated map so cells we already ran get [reuse].
echo "=== v2-extend Phase 0.75 — reset hill-climb state to global Sonnet winner ==="
python -c "
import json
from pathlib import Path
v2 = Path('results/ward_backtracking_txc/hillclimb_state_v2.json')
canonical = Path('results/ward_backtracking_txc/hillclimb_state.json')
rank = Path('results/ward_backtracking_txc/rank_global_sonnet.json')

# Start from v2 state (preserves evaluated) if available, else fresh.
if v2.exists():
    s = json.loads(v2.read_text())
    print(f'[v2-extend] preserved {len(s[\"evaluated\"])} evaluated cells from v2 state')
else:
    s = {'current_best': None, 'iteration': 0, 'history': [], 'evaluated': {}}

# Pull in metrics for ALL cells in cell_metrics/ (lowk + post-iter-1) so they
# show up as [reuse] when seen as neighbors.
md = Path('results/ward_backtracking_txc/cell_metrics')
for p in md.glob('*.json'):
    m = json.loads(p.read_text())
    s['evaluated'].setdefault(m['cell_id'], m)

# Reset iteration + reseed current_best to global Sonnet winner.
ranks = json.loads(rank.read_text())
winner = ranks['winner_cell_id']
s['iteration'] = 0
if winner:
    s['current_best'] = {'cell_id': winner, 'metric': ranks['winner_metric']}
    print(f'[v2-extend] new anchor: {winner} sonnet={ranks[\"winner_metric\"][\"primary_kw_at_sonnet_coh\"]:.4f}')

canonical.write_text(json.dumps(s, indent=2))
"

echo "=== v2-extend Phase D' — hill-climb with extended arch_list (H8 + H13) ==="
python -m $ROOT.hill_climb \
    --num-gpus "${NUM_GPUS}" \
    --max-iter "${MAX_ITER:-4}" \
    --improvement-threshold "${IMPROVEMENT_THRESHOLD:-0.05}" \
    --metric-key primary_kw_at_sonnet_coh

# Snapshot the extended state.
cp -f results/ward_backtracking_txc/hillclimb_state.json \
      results/ward_backtracking_txc/hillclimb_state_v2_extended.json

echo "=== v2-extend Phase E' — Sonnet-grade any new H8/H13 cells ==="
python -m $ROOT.regrade_cells --grade --concurrency 12

echo "=== v2-extend Phase F' — refresh plots ==="
python -m $ROOT.plot.steering_comparison_bars
python -m $ROOT.plot.coherence
python -m $ROOT.plot.feature_firing_heatmap
python -m $ROOT.plot.per_offset_firing
python -m $ROOT.plot.cosine_matrix
python -m $ROOT.plot.b2_difference_area || true
python -m $ROOT.plot.text_examples

echo
echo "[v2-extend] DONE."
echo "  state:    results/ward_backtracking_txc/hillclimb_state_v2_extended.json"
echo "  winner:   $(python -c "import json; s=json.load(open('results/ward_backtracking_txc/hillclimb_state_v2_extended.json')); print(s['current_best']['cell_id'] if s['current_best'] else 'none')")"
