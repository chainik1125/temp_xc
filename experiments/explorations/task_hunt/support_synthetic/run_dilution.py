"""Item 1 — the budget-dilution receipt grid (CARD § 1).

Three TXC-pre lines on the λ̂ mirror, k_pos = 1, seeds {1, 2, 42}, L = 32,
N_STEPS = 30 000, untrained control at every line-point:

  A1 (canonical fixed budget)  d_sae = 20   T ∈ {2, 4, 8, 16}
  A2 (fixed budget, 2F)        d_sae = 40   T ∈ {2, 4, 8, 16, 32}
  B  (budget-scaled, 5·T)      d_sae = 5T   T ∈ {2, 4, 8, 16, 32}

(T = 32, d = 20) is dict-infeasible for the pooled family (k_pos·T > d_sae) —
A1 ends at 16, per the card. Cells whose exact config already sits on the
leaderboard (the bench grid's T ∈ {2, 4, 8} points) are runner cache hits:
the existing row comes back, nothing is re-appended — 0 dup keys.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.support_synthetic.run_dilution [workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS = "toy_backtracking_selfexcite_d64"
F = 20
N_STEPS = 30_000
L = 32
ARCH = (("txc_batchtopk_pre", "pre"),)
HERE = Path(__file__).resolve().parent

# line -> [(T, d_sae), ...] — exactly the card's table.
POINTS = {
    "A1": [(2, 20), (4, 20), (8, 20), (16, 20)],
    "A2": [(2, 40), (4, 40), (8, 40), (16, 40), (32, 40)],
    "B": [(2, 10), (4, 20), (8, 40), (16, 80), (32, 160)],
}


def _cells():
    seen, cells = set(), []
    for pts in POINTS.values():
        for T, d in pts:
            if (T, d) in seen:            # shared cells (B∩A1, B∩A2): run once
                continue
            seen.add((T, d))
            cells += design.uniform_cells(
                DS, F, N_STEPS, k_pos_sweep=(1,), archs=ARCH, window_ts=(T,),
                d_saes=[d], L=L, untrained=True, untrained_kpos=1, log=print)
    return cells


def _describe(res):
    m = res["metrics"]
    return (f"λ={m.get('lambda_recovery', float('nan')):.3f} "
            f"l0w={m.get('l0_per_window', float('nan')):.2f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = HERE / "results" / "dilution_grid_results.json"
    grid.run_pool(_cells(), out, max_workers=workers, describe=_describe,
                  tag="support/dilution")


if __name__ == "__main__":
    main()
