"""Run the multilane superposition (FB-2) benchmark grid — thin config driver.

Cell enumeration only; the locked uniform design lives in
:mod:`explorations.synthetic.design`, the pool in
:func:`explorations.synthetic.grid.run_pool`. Enumerates the uniform grid on
the FB-2 datasource plus the **matched-budget band-partition addendum**
frozen in the card (freqbench/cards/FB-2.md § 6 — the sprint-transported
multiband-vs-full headline needs `spectral_txc_full` (1-band) and
`spectral_txc_dcac` (2-band) at k_pos=1, the frequency-A6 pattern).

Uniform design: 6 fair-backbone archs × d_sae ∈ {50, 101, 202} (F = M = 101)
× T ∈ {1,2,4,8} × dict-feasible k_pos ∈ {1,2,4,8,16} × seeds {1,2,42}
+ untrained controls; n_steps = 30_000; eval_window_L = 32.

    .venv/bin/python -m experiments.explorations.synthetic.multilane.run_grid [max_workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS = "toy_multilane_circle_M101_d24"
F = 101
N_STEPS = 30_000
BAND_ARCHS = (("spectral_txc_full", "spectral"), ("spectral_txc_dcac", "spectral"))
WINDOW_ARCHS = tuple((a, f) for a, f in design.FAIR_BACKBONE if f != "token")

OUT = Path(__file__).resolve().parent / "results" / "multilane_grid_results.json"


def _cells():
    cells = design.uniform_cells(DS, F, N_STEPS, log=print)
    # band-partition addendum (frozen in the card): full/dcac, k_pos=1.
    cells += design.uniform_cells(DS, F, N_STEPS, archs=BAND_ARCHS,
                                  k_pos_sweep=(1,), log=print)
    # T=16 frontier addendum (briefings/freqbench-t16-fbc2.md): the window archs
    # + the band pair, so the multiband-vs-full margin is measured off the
    # coarse-window regime.
    cells += design.uniform_cells(DS, F, N_STEPS, archs=WINDOW_ARCHS,
                                  window_ts=(16,), log=print)
    cells += design.uniform_cells(DS, F, N_STEPS, archs=BAND_ARCHS,
                                  window_ts=(16,), k_pos_sweep=(1,), log=print)
    return cells


def _describe(res):
    m = res["metrics"]
    return (f"ml={m.get('multilane_recovery', float('nan')):+.3f} "
            f"orc={m.get('multilane_oracle', float('nan')):.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    grid.run_pool(_cells(), OUT, max_workers=max_workers, describe=_describe,
                  tag="multilane")


if __name__ == "__main__":
    main()
