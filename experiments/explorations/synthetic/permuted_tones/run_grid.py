"""Run the permuted-tones (FB-5) benchmark grid — thin config driver.

The plain locked uniform grid on the FB-5 datasource: 6 fair-backbone archs
× d_sae ∈ {50, 101, 202} (F = M = 101) × T ∈ {1,2,4,8} × dict-feasible
k_pos ∈ {1,2,4,8,16} × seeds {1,2,42} + untrained; n_steps = 6000 (the
frequency substrate's budget, frozen in the card); eval_window_L = 32.
Frozen predictions: freqbench/cards/FB-5.md § 6 (headline: spectral trained
BELOW txc-post at the canonical T=8 cell — the reversal of multilane; T=16
deliberately NOT included, per the briefing).

    .venv/bin/python -m experiments.explorations.synthetic.permuted_tones.run_grid [max_workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS = "toy_permuted_circle_M101_d128"
F = 101
N_STEPS = 6000

OUT = Path(__file__).resolve().parent / "results" / "permuted_grid_results.json"


def _cells():
    return design.uniform_cells(DS, F, N_STEPS, log=print)


def _describe(res):
    m = res["metrics"]
    return (f"sched={m.get('schedule_recovery', float('nan')):+.3f} "
            f"orc={m.get('schedule_oracle', float('nan')):.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    grid.run_pool(_cells(), OUT, max_workers=max_workers, describe=_describe,
                  tag="permuted")


if __name__ == "__main__":
    main()
