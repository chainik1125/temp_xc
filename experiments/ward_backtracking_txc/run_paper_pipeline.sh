#!/usr/bin/env bash
# Stage B paper-budget pipeline:
#
#   Phase 1   — sweep:           cache, train+mine+B1+B2 the 12-cell grid (4 archs × 3 hookpoints)
#   Phase 1.5 — rank:             pick winner cell from Phase 1 B1 + coherence
#   Phase 2   — hill-climb:       greedy local search from winner, ≤4 iterations
#   Phase 6   — plots + writeup:  refresh all images_b/*.png from final state
#
# Multi-GPU: auto-detected via nvidia-smi -L; falls back to 1.
#
# Idempotent: every phase script skips already-completed work; re-running
# the pipeline picks up from the last incomplete cell.

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="experiments.ward_backtracking_txc"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "${NUM_GPUS}" -lt 1 ]; then
    echo "[fatal] no GPUs detected" >&2; exit 1
fi
echo "[paper-pipeline] NUM_GPUS=${NUM_GPUS}"

# ─── Phase 1: sweep (cache + train + mine + B1 + B2) ────────────────────────
echo
echo "================================================================"
echo "  PHASE 1 — sweep (4 archs × 3 hookpoints = 12 cells)"
echo "================================================================"
bash "$(dirname "$0")/run_grid_2gpu.sh"

# ─── Phase 1.5: rank cells, pick winner ─────────────────────────────────────
echo
echo "================================================================"
echo "  PHASE 1.5 — rank Phase 1 cells, pick winner"
echo "================================================================"
python -m $ROOT.rank_phase1

# ─── Phase 2: greedy hill-climb ─────────────────────────────────────────────
echo
echo "================================================================"
echo "  PHASE 2 — greedy hill-climb from Phase 1 winner"
echo "================================================================"
python -m $ROOT.hill_climb --max-iter "${MAX_ITER:-4}" \
    --num-gpus "${NUM_GPUS}" \
    --improvement-threshold "${IMPROVEMENT_THRESHOLD:-0.05}"

# ─── Phase 1.6: merge per-cell B1 results into the canonical JSON ──────────
# (so plot scripts see all sweep + hill-climb sources in one file)
echo
echo "================================================================"
echo "  Merge per-cell B1 results into canonical JSON for plotting"
echo "================================================================"
python -c "
from pathlib import Path
import yaml
from experiments.ward_backtracking_txc import metrics
cfg = yaml.safe_load(open('experiments/ward_backtracking_txc/config.yaml'))
canonical = Path(cfg['paths']['steering'])
per_cell_dir = Path(cfg['paths']['root']) / 'steering_per_cell'
shards = sorted(per_cell_dir.glob('b1__*.json'))
n = metrics.merge_b1_jsons([canonical] + shards, canonical)
print(f'merged {len(shards)} per-cell shards + canonical into {canonical} ({n} unique rows)')
"

# ─── Phase 6: refresh plots ────────────────────────────────────────────────
echo
echo "================================================================"
echo "  PHASE 6 — refresh all plots"
echo "================================================================"
python -m $ROOT.plot.training_curves
python -m $ROOT.plot.feature_firing_heatmap
python -m $ROOT.plot.steering_comparison_bars
python -m $ROOT.plot.per_offset_firing
python -m $ROOT.plot.cosine_matrix
python -m $ROOT.plot.sentence_act_distributions
python -m $ROOT.plot.text_examples
python -m $ROOT.plot.b2_difference_area
python -m $ROOT.plot.coherence
python -m $ROOT.plot.decoder_umap || true
python -m $ROOT.plot.decoder_umap_x_umap || true

echo
echo "[paper-pipeline] done."
echo "  Phase 1 leaderboard: results/ward_backtracking_txc/rank_phase1.json"
echo "  Hill-climb history:  results/ward_backtracking_txc/hillclimb_state.json"
echo "  Final winner cell:   $(python -c "import json; s = json.load(open('results/ward_backtracking_txc/hillclimb_state.json')); print(s['current_best']['cell_id'])")"
echo "  Plots in:            docs/aniket/experiments/ward_backtracking/images_b/"
