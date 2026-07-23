"""Run the phasepair (FB-1) benchmark grid — thin config driver.

The plain locked uniform grid on the FB-1 datasource: 6 fair-backbone archs
× d_sae ∈ {50, 101, 202} (F = M = 101) × T ∈ {1,2,4,8} × dict-feasible
k_pos ∈ {1,2,4,8,16} × seeds {1,2,42} + untrained; n_steps = 30_000;
eval_window_L = 32. Frozen predictions: freqbench/cards/FB-1.md § 6
(headline: the phase-vs-power dissociation — pair_recovery ≥ sign_recovery
everywhere; additive family pair > 0, sign ≈ 0).

    .venv/bin/python -m experiments.explorations.synthetic.phasepair.run_grid [max_workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS = "toy_phasepair_M101_d24"
F = 101
N_STEPS = 30_000

OUT = Path(__file__).resolve().parent / "results" / "phasepair_grid_results.json"


WINDOW_ARCHS = tuple((a, f) for a, f in design.FAIR_BACKBONE if f != "token")


def _cells():
    cells = design.uniform_cells(DS, F, N_STEPS, log=print)
    # T=16 frontier addendum (briefings/freqbench-t16-fbc2.md): the window archs
    # one octave past the locked design (every multiband AC band is multi-index).
    cells += design.uniform_cells(DS, F, N_STEPS, archs=WINDOW_ARCHS,
                                  window_ts=(16,), log=print)
    return cells


def _describe(res):
    m = res["metrics"]
    return (f"sign={m.get('sign_recovery', float('nan')):+.3f} "
            f"pair={m.get('pair_recovery', float('nan')):+.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    grid.run_pool(_cells(), OUT, max_workers=max_workers, describe=_describe,
                  tag="phasepair")


if __name__ == "__main__":
    main()
