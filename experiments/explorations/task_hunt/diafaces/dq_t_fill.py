"""dq panel T{6,10} fill — Han grid item (5), DQ_T_FILL_CARD.md.

The λ̂ T-fill shape (`lambda_intensity/run_t_fill.py`) on the dq panel's
venue: `run_panel.py` constants + the PROBE_V2_SPEC § 2 block verbatim
on every cell (paired-columns term), eval_window_L=30 (card venue line).

Run (GPU 1)::

    CUDA_VISIBLE_DEVICES=1 AGENT_NAME=runpod-b TEMP_BENCH_ALLOW_DIRTY=1 \
        .venv/bin/python -m \
        experiments.explorations.task_hunt.diafaces.dq_t_fill [workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

from experiments.explorations.task_hunt.diafaces.run_panel import (
    BUFFER_TOKENS,
    PANEL_DS,
    V2,
)
from experiments.explorations.task_hunt.lambda_intensity.run_stage2 import (
    D_SAE,
    K_POS,
    N_STEPS,
    _describe,
)

DS = PANEL_DS["dq"]
FILL_TS = (6, 10)
EVAL_L_FILL = 30          # card venue line (T | L tiling divisibility)
PANEL_POST = (("txc_batchtopk_post", "post"),)
HERE = Path(__file__).resolve().parent


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cells = design.uniform_cells(
        DS, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE],
        k_pos_sweep=K_POS, archs=PANEL_POST, window_ts=FILL_TS,
        L=EVAL_L_FILL, untrained_kpos=K_POS[0], log=print)
    for c in cells:
        c["buffer_tokens"] = BUFFER_TOKENS
        c["eval_extra"] = V2
    assert len(cells) == 12, f"expected 12 cells, got {len(cells)}"
    assert all(c["eval_window_L"] == EVAL_L_FILL for c in cells)
    out = HERE / "results" / "dq_t6t10_fill.json"
    grid.run_pool(cells, out, max_workers=workers, describe=_describe,
                  tag=f"dq_t6t10fill/{DS}")


if __name__ == "__main__":
    main()
