"""Hedging-trend LEVEL Stage 2 — the head-to-head panel on REAL Ward
activations (task-hunt round 2, arm B; `briefings/task-hunt-r2-e.md` § 1).

Card: `card_stage2.md` (FROZEN before any cell — the killed screen card
`CARD.md` is motivation only, never confirmation). Reuses the reviewed
candidate-1 Stage-2 pattern (`lambda_intensity/run_stage2.py`) cell for cell:
5 archs × T ∈ {2, 4, 8, 16} × seeds {1, 2, 42} + untrained, single scarce
anchor d_sae = 2048 = d_in/2, eval_window_L = 32, n_steps = 8000,
buffer sized to the corpus. Datasource `ward_real_slope8_distill_l14`
(plugin `explorations.task_hunt.real_slope`, generator/R1-Distill reader at
resid_post L14 = hs15, the frozen screen layer).

**The money plot** is `lambda_recovery` (held-out Pearson r vs the frozen
slope8 grid) vs T, one line per arch.

The ONE deviation from the reviewed round-1 panel, per the r2 briefing's
matched-REALIZED-l0 requirement and runpod-d's frozen amendment convention
(`lambda_intensity/card_stage2_postmatched.md` § 2–3, code-rate reading):
`txc_batchtopk_post` runs at nominal **k_pos = 8·T** (16/32/64/128), trained
AND untrained, so its realized per-token code rate matches the rest of the
panel's ≈ 8 instead of collapsing as 8/T. Every other arch stays at nominal
k_pos = 8. Falsifier: post's untrained cells must realize l0/token = 8.00
(± 0.02) at every T, else the k·T correction is wrong and the post cells are
void (card § falsifiers).

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.confidence.run_stage2 [workers] [ds]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS_DEFAULT = "ward_real_slope8_distill_l14"
D_SAE = 2048                      # d_in/2 — the scarce anchor
K_POS = (8,)
WINDOW_TS = (2, 4, 8, 16)
EVAL_L = 32
N_STEPS = 8_000
BUFFER_TOKENS = 524_288           # ≈ the corpus (4044 × 128 = 517,632)
HERE = Path(__file__).resolve().parent

PANEL = (
    ("batchtopk_sae", "token"),
    ("tsae", "token"),
    ("stacked_batchtopk", "stacked"),
    ("txc_batchtopk_pre", "pre"),
    ("txc_batchtopk_post", "post"),
)


def _cells(ds: str):
    cells = design.uniform_cells(
        ds, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE], k_pos_sweep=K_POS,
        archs=PANEL, window_ts=WINDOW_TS, L=EVAL_L, untrained_kpos=K_POS[0],
        log=print)
    for c in cells:
        c["buffer_tokens"] = BUFFER_TOKENS
        # Budget-matched TXC-post (card § panel; k·T so realized l0 ≈ 8/token
        # instead of 8/T — the post-squash divides the window budget by T).
        if c["arch"] == "txc_batchtopk_post":
            c["k_pos"] = c["k_pos"] * c["T"]
    return cells


def _describe(res):
    m = res["metrics"]
    return (f"r={m.get('lambda_recovery', float('nan')):.3f} "
            f"chance={m.get('lambda_chance', float('nan')):+.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ds = sys.argv[2] if len(sys.argv) > 2 else DS_DEFAULT
    out = HERE / "results" / f"stage2_{ds}.json"
    cells = _cells(ds)
    grid.run_pool(cells, out, max_workers=workers, describe=_describe,
                  tag=f"stage2/{ds}")


if __name__ == "__main__":
    main()
