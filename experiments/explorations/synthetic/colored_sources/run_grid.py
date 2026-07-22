"""Run the colored-sources (FB-3) benchmark grid — thin config driver.

Cell enumeration only; the locked uniform design lives in
:mod:`explorations.synthetic.design`, the pool in
:func:`explorations.synthetic.grid.run_pool`. The plain uniform grid on the
FB-3 datasource (no addenda): 6 fair-backbone archs × d_sae ∈ {16, 32, 64}
(F = N = 32) × T ∈ {1,2,4,8} × dict-feasible k_pos ∈ {1,2,4,8,16} × seeds
{1,2,42} + untrained controls; n_steps = 30_000; eval_window_L = 32.

The primary metric (`colored_rec_adj`) is weight-space — the eval add-on
costs nothing; the frozen predictions live in freqbench/cards/FB-3.md § 6
(headline: the W = D+1 transition — all T ≤ 2 cells provably floored).

    .venv/bin/python -m experiments.explorations.synthetic.colored_sources.run_grid [max_workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS = "toy_colored_sources_N32_D2_d32"
F = 32
N_STEPS = 30_000

OUT = Path(__file__).resolve().parent / "results" / "colored_grid_results.json"


def _cells():
    return design.uniform_cells(DS, F, N_STEPS, log=print)


def _describe(res):
    m = res["metrics"]
    return (f"rec={m.get('colored_rec_adj', float('nan')):+.3f} "
            f"eauc={m.get('eauc', float('nan')):.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    grid.run_pool(_cells(), OUT, max_workers=max_workers, describe=_describe,
                  tag="colored")


if __name__ == "__main__":
    main()
