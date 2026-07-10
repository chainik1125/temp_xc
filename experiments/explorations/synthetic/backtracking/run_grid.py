"""Run the backtracking (self-exciting) benchmark grid — thin config driver.

Cell enumeration + config only; the locked uniform design lives in
:mod:`explorations.synthetic.design`, the pool + canonical-runner plumbing in
:func:`explorations.synthetic.grid.run_pool`.

Uniform clean-room design (briefings/full-rerun-and-purge.md):
  archs   : batchtopk_sae, tsae (token, T=1); stacked_batchtopk, txc_batchtopk_pre,
            txc_batchtopk_post, spectral_txc (window, T∈{2,4,8})   [fair-backbone]
  d_sae   : {F//2, F, 2F} = {10, 20, 40}   (F=20)
  k_pos   : {1,2,4,8,16} meeting each arch's dict constraint (drops logged)
  seeds   : 1, 2, 42   + untrained control (n_steps=0) per (arch,T)
  eval_window_L=32, n_steps=30000. Throughput normalised (batch = 1024//T).

    .venv/bin/python -m experiments.explorations.synthetic.backtracking.run_grid [max_workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS = "toy_backtracking_selfexcite_d64"
F = 20
N_STEPS = 30_000
OUT = Path(__file__).resolve().parent / "results" / "backtracking_grid_results.json"


def _cells():
    return design.uniform_cells(DS, F, N_STEPS, log=print)


def _describe(res):
    m = res["metrics"]
    return (f"λ={m.get('lambda_recovery', float('nan')):.3f} "
            f"eauc={m.get('eauc', float('nan')):.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    grid.run_pool(_cells(), OUT, max_workers=max_workers, describe=_describe, tag="backtracking")


if __name__ == "__main__":
    main()
