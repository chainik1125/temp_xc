"""Run the cyclic-tone frequency benchmark grid — thin config driver.

Cell enumeration + config only; the pool + canonical-runner plumbing lives in
:func:`explorations.synthetic.grid.run_pool`. This driver enumerates the whole
grid — the main arch family AND the matched-budget band-partition addendum
(amendment A6), which used to be a second driver (``run_grid_bands.py``, now
folded in here).

Grid (frequency/bench_spec.md § 5 + amendments A1–A6 — BatchTopK fair-backbone):
  archs/T : (batchtopk_sae,1) (tsae,1)                             [per-token]
            (txc_batchtopk_pre,2/4/8/16) (txc_batchtopk_post,2/4/8/16)
            (spectral_txc,2/4/8/16)                                [crosscoders]
  d_sae   : 32, 64, 101, 256   (anchored on M=101; all < |Ω|·M=1010)
  seeds   : 1, 2, 42;   k_pos=1, eval_window_L=32, n_steps=6000

Circle (headline): full d_sae frontier + untrained control + k_pos=2 anchor.
Random (null): anchor d_sae only. Plus a d_sae=2048 (> |Ω|·M) memorization demo
on both datasources, and the band-partition addendum: {spectral_txc_full,
spectral_txc_dcac} on the circle frontier + untrained (multiband is the main
grid; full=1-band, dcac=2-band, matched total budget). Stacked is dropped (A5).

    .venv/bin/python -m experiments.explorations.synthetic.frequency.run_grid [max_workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import grid

CIRCLE = "toy_cyclic_circle_M101_d128"
RANDOM = "toy_cyclic_random_M101_d128"
L = 32
K_POS = 1
N_STEPS = 6000
D_SAES = [32, 64, 101, 256]
ANCHOR = 101
SEEDS = [1, 2, 42]
MEMO_DSAE = 2048          # > |Ω|·M = 1010 → the memorization-regime demo

ARCH_T = [("batchtopk_sae", 1), ("tsae", 1),
          ("txc_batchtopk_pre", 2), ("txc_batchtopk_pre", 4),
          ("txc_batchtopk_pre", 8), ("txc_batchtopk_pre", 16),
          ("txc_batchtopk_post", 2), ("txc_batchtopk_post", 4),
          ("txc_batchtopk_post", 8), ("txc_batchtopk_post", 16),
          ("spectral_txc", 2), ("spectral_txc", 4),
          ("spectral_txc", 8), ("spectral_txc", 16)]
MEMO_ARCH_T = [("txc_batchtopk_pre", 16), ("spectral_txc", 16)]
# Band-partition addendum (matched total budget; multiband already in ARCH_T).
BAND_ARCHS = ["spectral_txc_full", "spectral_txc_dcac"]
BAND_T = [2, 4, 8, 16]

OUT = Path(__file__).resolve().parent / "results" / "frequency_grid_results.json"


def _cell(ds, arch, T, d_sae, k_pos, seed, n_steps, kind):
    return {"ds": ds, "arch": arch, "T": T, "d_sae": d_sae, "k_pos": k_pos,
            "seed": seed, "n_steps": n_steps, "kind": kind, "eval_window_L": L}


def _cells():
    cells = []
    for seed in SEEDS:
        for arch, T in ARCH_T:
            for d_sae in D_SAES:                                    # circle frontier
                cells.append(_cell(CIRCLE, arch, T, d_sae, K_POS, seed, N_STEPS, "trained"))
            cells.append(_cell(CIRCLE, arch, T, ANCHOR, K_POS, seed, 0, "untrained"))
            cells.append(_cell(CIRCLE, arch, T, ANCHOR, 2, seed, N_STEPS, "trained"))
            cells.append(_cell(RANDOM, arch, T, ANCHOR, K_POS, seed, N_STEPS, "trained"))
    # memorization demo (> |Ω|·M): both datasources, 1 seed
    for ds in (CIRCLE, RANDOM):
        for arch, T in MEMO_ARCH_T:
            cells.append(_cell(ds, arch, T, MEMO_DSAE, K_POS, 1, N_STEPS, "memo"))
    # band-partition addendum: full/dcac on the circle frontier + untrained
    for seed in SEEDS:
        for arch in BAND_ARCHS:
            for T in BAND_T:
                for d_sae in D_SAES:
                    cells.append(_cell(CIRCLE, arch, T, d_sae, K_POS, seed, N_STEPS, "trained"))
                cells.append(_cell(CIRCLE, arch, T, ANCHOR, K_POS, seed, 0, "untrained"))
    return cells


def _describe(res):
    m = res["metrics"]
    return (f"vel={m.get('velocity_recovery', float('nan')):+.3f} "
            f"orc={m.get('velocity_oracle', float('nan')):.3f} "
            f"nmse={m.get('nmse', float('nan')):.3f}")


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    grid.run_pool(_cells(), OUT, max_workers=max_workers, describe=_describe, tag="grid")


if __name__ == "__main__":
    main()
