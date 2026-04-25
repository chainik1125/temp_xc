"""Fig 5/6/7 grid: three-architecture synthetic benchmark.

Sweeps {regular SAE, Stacked SAE, TXCDR} over rho x k x T matching the
midterm-report configuration (report §2.1-2.3, §3.1-3.2):

    rho in {0.0, 0.6, 0.9}
    k   in {2, 5, 10, 25}
    T   in {2, 5}

All three archs have matched window-level L0 = k * T. Feature-recovery
AUC averages decoder columns across positions for TXCDR and Stacked SAE,
matching the convention already used on temporal-bench.

Writes `results/three_arch_sweep/sweep_results.json` consumed by the
plot_fig5 / plot_fig6 / plot_fig7 scripts.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "src")

from temporal_bench.config import DataConfig, SweepConfig, TrainConfig
from temporal_bench.sweep import run_sweep


def build_config(args: argparse.Namespace) -> SweepConfig:
    data = DataConfig(
        n_features=128,
        d_model=256,
        pi=0.05,
        seed=args.data_seed,
    )
    train = TrainConfig(
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=3e-4,
        grad_clip=1.0,
        eval_every=max(args.steps // 10, 1),
        seed=args.seed,
    )
    return SweepConfig(
        models=["regular_sae", "stacked_sae", "txcdr"],
        rho_values=[0.0, 0.6, 0.9],
        k_values=[2, 5, 10, 25],
        T_values=[2, 5],
        train=train,
        data=data,
        n_seeds=args.n_seeds,
        output_dir=args.output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=65_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=str, default="results/three_arch_sweep"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = build_config(args)
    run_sweep(cfg)


if __name__ == "__main__":
    main()
