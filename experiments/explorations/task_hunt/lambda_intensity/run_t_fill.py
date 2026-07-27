"""λ̂ Stage-2 T{6,10} fill — Han grid item (4), T_FILL_CARD.md.

The stage-2 pathway (`run_stage2.py`) narrowed to the missing T-points:
post arch only, window_ts=(6,10), eval_window_L=30 (T|L divisibility —
see the card's venue line), everything else the stage-2 constants.

Run (GPU 1)::

    CUDA_VISIBLE_DEVICES=1 AGENT_NAME=runpod-b TEMP_BENCH_ALLOW_DIRTY=1 \
        .venv/bin/python -m \
        experiments.explorations.task_hunt.lambda_intensity.run_t_fill [workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

from .run_stage2 import (
    BUFFER_TOKENS,
    D_SAE,
    DS_DEFAULT,
    K_POS,
    N_STEPS,
    _describe,
)

FILL_TS = (6, 10)
EVAL_L_FILL = 30          # minimal L divisible by both fill Ts (card venue line)
PANEL_POST = (("txc_batchtopk_post", "post"),)
HERE = Path(__file__).resolve().parent


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cells = design.uniform_cells(
        DS_DEFAULT, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE],
        k_pos_sweep=K_POS, archs=PANEL_POST, window_ts=FILL_TS,
        L=EVAL_L_FILL, untrained_kpos=K_POS[0], log=print)
    for c in cells:
        c["buffer_tokens"] = BUFFER_TOKENS
    assert len(cells) == 12, f"expected 12 cells (6 trained + 6 untrained), got {len(cells)}"
    assert all(c["eval_window_L"] == EVAL_L_FILL for c in cells)
    out = HERE / "results" / f"stage2_t6t10_{DS_DEFAULT}.json"
    grid.run_pool(cells, out, max_workers=workers, describe=_describe,
                  tag=f"t6t10fill/{DS_DEFAULT}")


if __name__ == "__main__":
    main()
