"""Loss-dissection grid — thin config driver (CARD § 4, frozen).

Four loss-only variants of the txc_batchtopk_post backbone
(txc_post_plain / _mat / _ctr / _both) on the five-bench discriminating
set, canonical slice: d_sae=F, T∈{2,4,8}, k_pos∈{1,2,4}, seeds {1,2,42}
+ automatic untrained control per (variant, T, seed). 144 cells/bench,
720 total. Anchor (txc_batchtopk_post) rows are NOT re-run — they are
read from the canonical leaderboard (135/135 verified present at freeze).

    .venv/bin/python -m experiments.explorations.synthetic.loss_dissection.run_grid [max_workers] [bench ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

HERE = Path(__file__).resolve().parent

DISSECT_ARCHS = (
    ("txc_post_plain", "post"),
    ("txc_post_mat", "post"),
    ("txc_post_ctr", "post"),
    ("txc_post_both", "post"),
)
K_POS_SLICE = (1, 2, 4)

# (bench, datasource, F, n_steps, primary metric) — CARD § 4.
BENCHES = (
    ("backtracking", "toy_backtracking_selfexcite_d64", 20, 30_000, "lambda_recovery"),
    ("frequency", "toy_cyclic_circle_M101_d128", 101, 6_000, "velocity_recovery"),
    ("phasepair", "toy_phasepair_M101_d24", 101, 30_000, "sign_recovery"),
    ("recipe_instruction_phase_runs", "toy_recipe_instruction_d64", 20, 30_000,
     "equality_residual_recovery"),
    ("multilane", "toy_multilane_circle_M101_d24", 101, 30_000, "multilane_recovery"),
)


def _cells(ds: str, F: int, n_steps: int):
    return design.uniform_cells(
        ds, F, n_steps, archs=DISSECT_ARCHS, k_pos_sweep=K_POS_SLICE,
        d_saes=[F], log=print,
    )


def _describe(primary):
    def fn(res):
        m = res["metrics"]
        return (f"{primary.split('_')[0]}={m.get(primary, float('nan')):.3f} "
                f"nmse={m.get('nmse', float('nan')):.3f} "
                f"l0t={m.get('l0_per_token', float('nan')):.2f}")
    return fn


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    only = set(sys.argv[2:])
    for bench, ds, F, n_steps, primary in BENCHES:
        if only and bench not in only:
            continue
        out = HERE / "results" / f"{bench}_dissect_grid_results.json"
        cells = _cells(ds, F, n_steps)
        grid.run_pool(cells, out, max_workers=max_workers,
                      describe=_describe(primary), tag=f"dissect:{bench}")


if __name__ == "__main__":
    main()
