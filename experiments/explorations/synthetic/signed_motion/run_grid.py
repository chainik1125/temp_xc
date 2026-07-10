"""Run the signed-motion (order-sensitive AC) benchmark grid — thin config driver.

NEW under the uniform clean-room design: the published signed_motion figure was
built on the deprecated TopK-legacy archs (txc_base / stacked_sae / topk_sae),
which the purge removes. This re-runs the bench on the fair-backbone suite so its
`s_temp` (sign recovery) row joins the program B×A matrix. Cell enumeration lives
in :mod:`explorations.synthetic.design`; the pool in
:func:`explorations.synthetic.grid.run_pool`.

Uniform design (briefings/full-rerun-and-purge.md):
  archs   : batchtopk_sae, tsae (token, T=1); stacked_batchtopk, txc_batchtopk_pre,
            txc_batchtopk_post, spectral_txc (window, T∈{2,4,8})   [fair-backbone]
  d_sae   : {F//2, F, 2F} = {9, 19, 38}   (F=19)
  k_pos   : {1,2,4,8,16} meeting each arch's dict constraint (drops logged)
  seeds   : 1, 2, 42   + untrained control (n_steps=0) per (arch,T)
  eval_window_L=32, n_steps=10000. Throughput normalised (batch = 1024//T).

The verdict is expected to stay NEGATIVE (the #windows=2F=38 memorization
confound is structural), but the archs are now the fair-backbone suite.

    .venv/bin/python -m experiments.explorations.synthetic.signed_motion.run_grid [max_workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS = "toy_signed_motion_M19_d40"
F = 19
N_STEPS = 10_000
OUT = Path(__file__).resolve().parent / "results" / "signed_motion_grid_results.json"


def _cells():
    return design.uniform_cells(DS, F, N_STEPS, log=print)


def _describe(res):
    m = res["metrics"]
    return (f"s_temp={m.get('s_temp', float('nan')):+.3f} "
            f"eauc={m.get('eauc', float('nan')):.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    grid.run_pool(_cells(), OUT, max_workers=max_workers, describe=_describe, tag="signed_motion")


if __name__ == "__main__":
    main()
