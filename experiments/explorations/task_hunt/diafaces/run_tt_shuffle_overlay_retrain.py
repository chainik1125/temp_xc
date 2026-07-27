"""ttrend shuffle-overlay RETRAIN grid (TT_SHUFFLE_OVERLAY_CARD.md § 2).

The quoted v2 tt panel restricted to the directive's arms — claiming
arm txc_batchtopk_post × T ∈ {2,4,8,16,32} + per-token anchors
(batchtopk_sae, tsae @ T=1), seeds {1,2,42} — hyperparameters frozen
to the values every quoted payload row records (d2048/k8/8000 steps/
buffer 524288/eval L 32). `eval_extra.retrain_tag` namespaces the
eval_key (fresh rows, no collisions); checkpoints persist locally
for the overlay.

Run:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
        experiments.explorations.task_hunt.diafaces.run_tt_shuffle_overlay_retrain [workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

HERE = Path(__file__).resolve().parent
DS = "dial_real_ttrend_gpt2_l7"
D_SAE = 2048
K_POS = (8,)
WINDOW_TS = (2, 4, 8, 16, 32)
EVAL_L = 32
N_STEPS = 8_000
BUFFER_TOKENS = 524_288          # the quoted panel's corpus-sized buffer
RETRAIN_TAG = "tt_shuf_overlay_r1"

ARMS = (
    ("batchtopk_sae", "token"),
    ("tsae", "token"),
    ("txc_batchtopk_post", "post"),
)


def cells():
    cs = design.uniform_cells(
        DS, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE], k_pos_sweep=K_POS,
        archs=ARMS, window_ts=WINDOW_TS, L=EVAL_L, untrained=False,
        log=print)
    for c in cs:
        c["buffer_tokens"] = BUFFER_TOKENS
        c["eval_extra"] = {"retrain_tag": RETRAIN_TAG}
    assert len(cs) == 21, f"card § 2 grid is 21 cells, built {len(cs)}"
    return cs


def _describe(res):
    m = res["metrics"]
    return (f"r={m.get('lambda_recovery', float('nan')):.3f} "
            f"chance={m.get('lambda_chance', float('nan')):+.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out = HERE / "results" / "tt_shuffle_overlay_retrain.json"
    grid.run_pool(cells(), out, max_workers=workers, describe=_describe,
                  tag=f"ttshufoverlay/{DS}")


if __name__ == "__main__":
    main()
