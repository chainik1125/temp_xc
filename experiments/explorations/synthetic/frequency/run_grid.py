"""Run the cyclic-tone frequency benchmark grid — thin config driver.

Cell enumeration + config only; the locked uniform design lives in
:mod:`explorations.synthetic.design`, the pool in
:func:`explorations.synthetic.grid.run_pool`. This driver enumerates the uniform
CIRCLE grid plus the bench-specific extras kept from the frozen frequency spec:
the RANDOM null, the memorization demo, and the matched-budget band-partition
addendum (amendment A6).

Uniform clean-room design (briefings/full-rerun-and-purge.md):
  archs   : batchtopk_sae, tsae (token, T=1); stacked_batchtopk, txc_batchtopk_pre,
            txc_batchtopk_post, spectral_txc (window, T∈{2,4,8})   [fair-backbone]
  d_sae   : {F//2, F, 2F} = {50, 101, 202}   (F = alphabet M = 101; all < |Ω|·M=1010)
  k_pos   : {1,2,4,8,16} meeting each arch's dict constraint (drops logged)
  seeds   : 1, 2, 42   + untrained control (n_steps=0) per (arch,T)
  eval_window_L=32, n_steps=6000.

Extras (frequency-specific, on the same fair backbone):
  - RANDOM null: anchor d_sae=101, k_pos=1, all (arch,T), 3 seeds (geometry ablation).
  - memo demo: d_sae=2048 (> |Ω|·M) on {txc_pre, spectral} T=8, both datasources.
  - band addendum: spectral_txc_full (1-band) + spectral_txc_dcac (2-band) on the
    CIRCLE frontier + untrained, k_pos=1 (matched total budget k_win=k_pos·T).

    .venv/bin/python -m experiments.explorations.synthetic.frequency.run_grid [max_workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

CIRCLE = "toy_cyclic_circle_M101_d128"
RANDOM = "toy_cyclic_random_M101_d128"
F = 101
ANCHOR = 101
N_STEPS = 6000
MEMO_DSAE = 2048                       # > |Ω|·M = 1010 → the memorization-regime demo
MEMO_ARCHS = (("txc_batchtopk_pre", "pre"), ("spectral_txc", "spectral"))
BAND_ARCHS = (("spectral_txc_full", "spectral"), ("spectral_txc_dcac", "spectral"))
WINDOW_ARCHS = tuple((a, f) for a, f in design.FAIR_BACKBONE if f != "token")

OUT = Path(__file__).resolve().parent / "results" / "frequency_grid_results.json"


def _cells():
    cells = []
    # CIRCLE headline: the full uniform grid (frontier + k_pos sweep + untrained).
    cells += design.uniform_cells(CIRCLE, F, N_STEPS, log=print)
    # RANDOM null: anchor d_sae only, k_pos=1, all (arch,T), 3 seeds, no untrained.
    cells += design.uniform_cells(RANDOM, F, N_STEPS, d_saes=[ANCHOR],
                                  k_pos_sweep=(1,), untrained=False)
    # memorization demo (> |Ω|·M): both datasources, {txc_pre, spectral} T=8, 1 seed.
    for ds in (CIRCLE, RANDOM):
        cells += design.uniform_cells(ds, F, N_STEPS, archs=MEMO_ARCHS,
                                      window_ts=(8,), d_saes=[MEMO_DSAE],
                                      k_pos_sweep=(1,), seeds=(1,), untrained=False)
    # band-partition addendum: full/dcac on the CIRCLE frontier + untrained, k_pos=1.
    cells += design.uniform_cells(CIRCLE, F, N_STEPS, archs=BAND_ARCHS,
                                  k_pos_sweep=(1,), log=print)
    # T=16 frontier addendum (briefings/freqbench-t16-fbc2.md): the window archs
    # + the matched-budget band pair, one octave past the locked design.
    cells += design.uniform_cells(CIRCLE, F, N_STEPS, archs=WINDOW_ARCHS,
                                  window_ts=(16,), log=print)
    cells += design.uniform_cells(CIRCLE, F, N_STEPS, archs=BAND_ARCHS,
                                  window_ts=(16,), k_pos_sweep=(1,), log=print)
    return cells


def _describe(res):
    m = res["metrics"]
    return (f"vel={m.get('velocity_recovery', float('nan')):+.3f} "
            f"orc={m.get('velocity_oracle', float('nan')):.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    grid.run_pool(_cells(), OUT, max_workers=max_workers, describe=_describe, tag="frequency")


if __name__ == "__main__":
    main()
