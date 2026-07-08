"""Run the backtracking (self-exciting) benchmark grid — thin config driver.

Cell enumeration + config only; the pool + canonical-runner plumbing lives in
:func:`explorations.synthetic.grid.run_pool`.

Grid (synthetic/backtracking/bench_spec.md § 5 — BatchTopK fair-backbone):
  archs/T : (batchtopk_sae,1) (tsae,1) (stacked_batchtopk,2/4/8)
            (txc_batchtopk_pre,2/4/8) (txc_batchtopk_post,2/4/8)         [11]
  d_sae   : 8, 16, 20, 40   (anchored on F=20; scarce {8,16,20} + over-complete 40)
  seeds   : 1, 2, 42
  k_pos=1, eval_window_L=32, n_steps=30000                  -> 132 trained cells
Plus the UNTRAINED-encoder control (n_steps=0) + the k_pos=2 anchor at d_sae=20,
both for all 11 (arch,T) × 3 seeds -> 66 cells. Total 198. Throughput normalised
(batch = 1024//T).

    .venv/bin/python -m experiments.explorations.synthetic.backtracking.run_grid [max_workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import grid

DS = "toy_backtracking_selfexcite_d64"
L = 32
K_POS = 1
N_STEPS = 30_000
D_SAES = [8, 16, 20, 40]
SEEDS = [1, 2, 42]
ARCH_T = [("batchtopk_sae", 1), ("tsae", 1),
          ("stacked_batchtopk", 2), ("stacked_batchtopk", 4), ("stacked_batchtopk", 8),
          ("txc_batchtopk_pre", 2), ("txc_batchtopk_pre", 4), ("txc_batchtopk_pre", 8),
          ("txc_batchtopk_post", 2), ("txc_batchtopk_post", 4), ("txc_batchtopk_post", 8)]

OUT = Path(__file__).resolve().parent / "results" / "backtracking_grid_results.json"


def _cell(arch, T, d_sae, k_pos, seed, n_steps, kind):
    return {"ds": DS, "arch": arch, "T": T, "d_sae": d_sae, "k_pos": k_pos,
            "seed": seed, "n_steps": n_steps, "kind": kind, "eval_window_L": L}


def _cells():
    cells = []
    for seed in SEEDS:
        for arch, T in ARCH_T:
            for d_sae in D_SAES:
                cells.append(_cell(arch, T, d_sae, K_POS, seed, N_STEPS, "trained"))
            cells.append(_cell(arch, T, 20, K_POS, seed, 0, "untrained"))
            cells.append(_cell(arch, T, 20, 2, seed, N_STEPS, "trained"))
    return cells


def _describe(res):
    m = res["metrics"]
    return (f"λ={m.get('lambda_recovery', float('nan')):.3f} "
            f"eauc={m.get('eauc', float('nan')):.3f}")


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    grid.run_pool(_cells(), OUT, max_workers=max_workers, describe=_describe, tag="grid")


if __name__ == "__main__":
    main()
