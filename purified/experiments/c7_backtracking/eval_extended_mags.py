"""Re-evaluate already-trained C7 checkpoints at extended steering
magnitudes (beyond the locked ±16 grid).

Motivation: Dmitry's C6 EM finding (2026-05-05) — TXC architectures
maintain coherence at very large steering coefficients while per-token
SAEs lose coherence. Our locked C7 grid stops at ±16, where MLC's peak
$\\Delta gc$ sits at the boundary; we don't know whether MLC keeps
climbing or collapses past +16. Running the same cells at ±24, ±32
resolves the ambiguity.

This driver loads the existing checkpoint by ``train_key`` (cache hit;
no retraining) and runs a fresh ``run_arch_evaluation`` with a
user-supplied magnitude list. The resulting leaderboard row has a
distinct ``eval_key`` (the magnitude list is part of ``eval_cfg``) so
it does NOT overwrite the canonical row.

Usage::

    .venv/bin/python -m experiments.c7_backtracking.eval_extended_mags \\
        --arch txc_base --bs 1024 \\
        --magnitudes -32 -24 0 24 32

The default magnitude list adds ±24 and ±32 to the locked grid's edges
plus mag=0 for the per-question $\\Delta gc$ baseline.
"""
from __future__ import annotations

import argparse
import logging

from temp_bench import runner
from temp_bench.case_studies.backtracking import DEFAULT_PR_AUC_S_GRID
from temp_bench.schemas import TrainingConfig

from experiments.c7_backtracking.run import (
    COMPONENT,
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_eval_fn,
    my_train_fn,
)

DEFAULT_EXTENDED_MAGS = (-32.0, -24.0, 0.0, +24.0, +32.0)

log = logging.getLogger("c7.eval_extended_mags")


def main(*, arch: str, bs: int, n_steps: int, seed: int,
         magnitudes: tuple[float, ...]):
    log.info(
        "[c7.ext] arch=%s bs=%d n_steps=%d seed=%d magnitudes=%s",
        arch, bs, n_steps, seed, list(magnitudes),
    )
    runner.run_cell(
        component=COMPONENT,
        arch_name=arch,
        seed=seed,
        datasource_name=DATASOURCE,
        training_cfg=TrainingConfig(n_steps=n_steps, batch_size=bs),
        eval_cfg={
            "magnitudes": list(magnitudes),
            "cut_fraction": 0.25,
            "pr_auc_S_grid": list(DEFAULT_PR_AUC_S_GRID),
            # Disambiguator: marks this leaderboard row as the
            # extended-magnitudes follow-up eval (different eval_key
            # from the canonical ±16 grid evaluation).
            "_extended_mags": True,
        },
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=my_train_fn,
        eval_fn=my_eval_fn,
    )


def cli():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    help="Arch name (e.g. txc_base, txc_pro, mlc, "
                         "tsae_paper, topk_sae).")
    ap.add_argument("--bs", type=int, required=True,
                    help="batch_size used for the existing trained "
                         "checkpoint (256 or 1024).")
    ap.add_argument("--n-steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--magnitudes", type=float, nargs="+",
                    default=list(DEFAULT_EXTENDED_MAGS))
    args = ap.parse_args()
    raise SystemExit(main(
        arch=args.arch,
        bs=args.bs,
        n_steps=args.n_steps,
        seed=args.seed,
        magnitudes=tuple(args.magnitudes),
    ) or 0)


if __name__ == "__main__":
    cli()
