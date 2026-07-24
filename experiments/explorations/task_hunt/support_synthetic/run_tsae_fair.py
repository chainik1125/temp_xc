"""Item 2 — the T-SAE temporal-knob fairness grid (CARD § 2).

Pair-distance sweep Δ ∈ {1, 2, 4, 8} (``tsae_d*`` = TSAEDelta, contract-tested
bitwise-identical to the registered ``tsae`` at Δ=1) plus the auxiliary
``tsae_a0`` (registered class, contrastive_alpha = 0). Canonical mirror budget:
d_sae = 20 (per-section), k_pos = 1, seeds {1, 2, 42}, N_STEPS = 30 000,
untrained control for all five entries (the untrained guard: Δ/α touch only
train_step, so the five entries' untrained metrics must be exactly equal per
seed). Sequence-mode cells cost ~115 s each — 15 trained cells total.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.support_synthetic.run_tsae_fair [workers]
"""

from __future__ import annotations

import sys
from pathlib import Path

from explorations.synthetic import design, grid

DS = "toy_backtracking_selfexcite_d64"
F = 20
N_STEPS = 30_000
L = 32
HERE = Path(__file__).resolve().parent

ARCHS = (
    ("tsae_d1", "token"),
    ("tsae_d2", "token"),
    ("tsae_d4", "token"),
    ("tsae_d8", "token"),
    ("tsae_a0", "token"),
)


def _cells():
    return design.uniform_cells(
        DS, F, N_STEPS, k_pos_sweep=(1,), archs=ARCHS, d_saes=[20], L=L,
        untrained=True, untrained_kpos=1, log=print)


def _describe(res):
    m = res["metrics"]
    return (f"λ={m.get('lambda_recovery', float('nan')):.3f} "
            f"l0t={m.get('l0_per_token', float('nan')):.2f}")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = HERE / "results" / "tsae_fair_grid_results.json"
    grid.run_pool(_cells(), out, max_workers=workers, describe=_describe,
                  tag="support/tsae-fair")


if __name__ == "__main__":
    main()
