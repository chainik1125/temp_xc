"""Run the recipe-instruction phase-runs benchmark grid — thin config driver.

Cell enumeration + config only; the locked uniform design lives in
:mod:`explorations.synthetic.design`, the pool + canonical-runner plumbing in
:func:`explorations.synthetic.grid.run_pool`.

Uniform clean-room design (the stage-6 briefing / full-rerun template):
  archs   : batchtopk_sae, tsae (token, T=1); stacked_batchtopk, txc_batchtopk_pre,
            txc_batchtopk_post, spectral_txc (window, T∈{2,4,8})   [fair-backbone]
  d_sae   : {F//2, F, 2F} = {10, 20, 40}   (F=20)
  k_pos   : {1,2,4,8,16} meeting each arch's dict constraint (drops logged)
  seeds   : 1, 2, 42   + untrained control (n_steps=0) per (arch,T)
  eval_window_L=32, n_steps=30000. Throughput normalised (batch = 1024//T).

Run AFTER the A1–A3 freeze commits (stage-6 #3b briefing) — the primary
metric is the re-scoped `equality_residual_recovery`.

    .venv/bin/python -m experiments.explorations.synthetic.recipe_instruction_phase_runs.run_grid [max_workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS = "toy_recipe_instruction_d64"
F = 20
N_STEPS = 30_000
OUT = Path(__file__).resolve().parent / "results" / "recipe_grid_results.json"


def _cells():
    return design.uniform_cells(DS, F, N_STEPS, log=print)


def _describe(res):
    m = res["metrics"]
    return (f"phase={m.get('phase_recovery', float('nan')):.3f} "
            f"resid={m.get('equality_residual_recovery', float('nan')):+.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    grid.run_pool(_cells(), OUT, max_workers=max_workers, describe=_describe,
                  tag="recipe")


if __name__ == "__main__":
    main()
