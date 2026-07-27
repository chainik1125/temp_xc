"""λ̂ shuffle-overlay RETRAIN grid (SHUFFLE_OVERLAY_CARD.md § 2).

The Stage-2 design restricted to the directive's arms — claiming arm
txc_batchtopk_post × T ∈ {2,4,8,16} + per-token anchors
(batchtopk_sae, tsae @ T=1), seeds {1,2,42} — with hyperparameters
inherited BY CONSTRUCTION: cells come from `design.uniform_cells`
called with `run_stage2.py`'s exact arguments (same F-anchor, k_pos,
n_steps, eval L, corpus-sized buffer), so every train_key matches the
quoted panel's cell identity. `eval_extra.retrain_tag` namespaces the
eval_key: fresh leaderboard rows, no cache collisions (grid.py's
documented mechanism); checkpoints persist locally for the overlay.

Run:  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m \
        experiments.explorations.task_hunt.lambda_intensity.run_shuffle_overlay_retrain [workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

from experiments.explorations.task_hunt.lambda_intensity.run_stage2 import (
    BUFFER_TOKENS,
    D_SAE,
    DS_DEFAULT,
    EVAL_L,
    K_POS,
    N_STEPS,
    WINDOW_TS,
)

HERE = Path(__file__).resolve().parent
RETRAIN_TAG = "lam_shuf_overlay_r1"

# The directive's three arms (card § 2) — a strict subset of
# run_stage2.PANEL; token archs sit at T=1 by the design's arch_t_list.
ARMS = (
    ("batchtopk_sae", "token"),
    ("tsae", "token"),
    ("txc_batchtopk_post", "post"),
)


def cells():
    cs = design.uniform_cells(
        DS_DEFAULT, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE],
        k_pos_sweep=K_POS, archs=ARMS, window_ts=WINDOW_TS, L=EVAL_L,
        untrained=False, log=print)
    for c in cs:
        c["buffer_tokens"] = BUFFER_TOKENS          # run_stage2's corpus-sized buffer
        c["eval_extra"] = {"retrain_tag": RETRAIN_TAG}
    assert len(cs) == 18, f"card § 2 grid is 18 cells, built {len(cs)}"
    return cs


def _describe(res):
    m = res["metrics"]
    return (f"λ={m.get('lambda_recovery', float('nan')):.3f} "
            f"chance={m.get('lambda_chance', float('nan')):+.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out = HERE / "results" / "shuffle_overlay_retrain.json"
    grid.run_pool(cells(), out, max_workers=workers, describe=_describe,
                  tag=f"shufoverlay/{DS_DEFAULT}")


if __name__ == "__main__":
    main()
