"""Sweep runner — run architecture comparisons across parameter grids.

Provides run_sweep() for programmatic use and a CLI for command-line sweeps.

Usage:
    # Full sweep (all combos):
    python -m src.bench.sweep

    # Single run:
    python -m src.bench.sweep --rho 0.9 --k 5 --T 2

    # Specific models only:
    python -m src.bench.sweep --models topk_sae crosscoder

    # Quick test:
    python -m src.bench.sweep --steps 100

    # Leaky reset:
    python -m src.bench.sweep --delta 0.5

    # Coupled features (K=10 hidden, M=20 emissions):
    python -m src.bench.sweep --coupled --K-hidden 10 --M-emission 20 --n-parents 2

    # Dry run:
    python -m src.bench.sweep --dry-run
"""

import argparse
import itertools
import json
import os
import time

import torch

from src.bench.architectures import get_default_models, ModelEntry
from src.bench.config import (
    CouplingConfig,
    DataConfig,
    MarkovConfig,
    TrainConfig,
    SweepConfig,
)
from src.bench.data import build_data_pipeline, DataPipeline
from src.bench.eval import evaluate_model, EvalResult


def _get_generator(pipeline: DataPipeline, gen_key: str):
    """Look up the right data generator from the pipeline."""
    if gen_key == "flat":
        return pipeline.gen_flat
    elif gen_key == "seq":
        return pipeline.gen_seq
    elif gen_key.startswith("window_"):
        T = int(gen_key.split("_")[1])
        return pipeline.gen_windows[T]
    else:
        raise ValueError(f"Unknown gen_key: {gen_key}")


def run_sweep(
    models: list[ModelEntry],
    data_config: DataConfig,
    train_config: TrainConfig,
    sweep_config: SweepConfig,
    device: torch.device,
    results_dir: str = "results",
    verbose: bool = True,
) -> list[dict]:
    """Run a full architecture comparison sweep.

    For each (rho, k) combination, trains and evaluates every model in
    the models list, then saves results. Supports both standard and
    coupled-feature data generation modes.

    Args:
        models: List of ModelEntry specs to compare.
        data_config: Base data configuration (rho will be overridden per sweep point).
        train_config: Training hyperparameters.
        sweep_config: Sweep grid specification.
        device: Torch device.
        results_dir: Directory for JSON result output.
        verbose: Print progress.

    Returns:
        List of result dicts, one per (model, rho, k) combination.
    """
    os.makedirs(results_dir, exist_ok=True)
    all_results = []
    is_coupled = data_config.coupling is not None

    # Collect window sizes needed across all models
    window_sizes = set()
    for entry in models:
        if entry.gen_key.startswith("window_"):
            window_sizes.add(int(entry.gen_key.split("_")[1]))

    for seed in sweep_config.seeds:
        for rho in sweep_config.rho_values:
            # Build a pipeline for this (rho, seed) combination
            rho_config = DataConfig(
                toy_model=data_config.toy_model,
                markov=MarkovConfig(
                    pi=data_config.markov.pi,
                    rho=rho,
                    delta=data_config.markov.delta,
                ),
                coupling=data_config.coupling,
                seq_len=data_config.seq_len,
                d_sae=data_config.d_sae,
                seed=seed,
                eval_n_seq=data_config.eval_n_seq,
            )

            mode_str = ""
            if is_coupled:
                c = data_config.coupling
                mode_str = f" [coupled K={c.K_hidden} M={c.M_emission}]"
            if data_config.markov.delta > 0:
                mode_str += f" [delta={data_config.markov.delta}]"
            seed_str = f" [seed={seed}]" if len(sweep_config.seeds) > 1 else ""

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"  rho = {rho}{mode_str}{seed_str}")
                print(f"{'=' * 60}")

            pipeline = build_data_pipeline(
                rho_config, device, window_sizes=list(window_sizes) or None
            )

            for k in sweep_config.k_values:
                if verbose:
                    print(f"\n  k = {k}:")

                for entry in models:
                    # Seed training for reproducibility
                    torch.manual_seed(seed)
                    t0 = time.time()

                    # Resolve training params (apply overrides)
                    batch_size = entry.training_overrides.get(
                        "batch_size", train_config.batch_size
                    )
                    lr = entry.training_overrides.get("lr", train_config.lr)
                    total_steps = entry.training_overrides.get(
                        "total_steps", train_config.total_steps
                    )

                    # Create and train
                    model = entry.spec.create(
                        d_in=data_config.toy_model.hidden_dim,
                        d_sae=data_config.d_sae,
                        k=k,
                        device=device,
                    )
                    gen_fn = _get_generator(pipeline, entry.gen_key)
                    train_log = entry.spec.train(
                        model,
                        gen_fn,
                        total_steps=total_steps,
                        batch_size=batch_size,
                        lr=lr,
                        device=device,
                        log_every=train_config.log_every,
                        grad_clip=train_config.grad_clip,
                    )

                    # Evaluate with both local and global features
                    result = evaluate_model(
                        entry.spec,
                        model,
                        pipeline.eval_hidden,
                        device,
                        true_features=pipeline.true_features,
                        global_features=pipeline.global_features,
                        seq_len=data_config.seq_len,
                    )

                    elapsed = time.time() - t0
                    row = {
                        "model": entry.name,
                        "rho": rho,
                        "k": k,
                        "seed": seed,
                        **result.to_dict(),
                        "elapsed_sec": round(elapsed, 1),
                    }
                    if data_config.markov.delta > 0:
                        row["delta"] = data_config.markov.delta
                    all_results.append(row)

                    if verbose:
                        auc_str = f"AUC={result.auc:.4f}" if result.auc is not None else ""
                        global_str = ""
                        if result.global_auc is not None:
                            global_str = f" | gAUC={result.global_auc:.4f}"
                        print(
                            f"    {entry.name:20s} | NMSE={result.nmse:.6f} | "
                            f"L0={result.l0:.2f} | {auc_str}{global_str} | {elapsed:.0f}s"
                        )

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Save per-(rho, seed) results
            tag = f"rho{rho:.1f}_seed{seed}"
            rho_path = os.path.join(results_dir, f"results_{tag}.json")
            rho_results = [
                r for r in all_results if r["rho"] == rho and r["seed"] == seed
            ]
            with open(rho_path, "w") as f:
                json.dump(rho_results, f, indent=2)

    # Save full summary
    summary_path = os.path.join(results_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  SWEEP COMPLETE | {len(all_results)} runs")
        print(f"  Results: {summary_path}")
        print(f"{'=' * 60}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run architecture comparison sweep"
    )
    parser.add_argument("--rho", type=float, nargs="+", default=None,
                        help="Rho values to sweep (default: [0.0, 0.6, 0.9])")
    parser.add_argument("--k", type=int, nargs="+", default=None,
                        help="K values to sweep (default: [2, 5, 10, 25])")
    parser.add_argument("--T", type=int, nargs="+", default=None,
                        help="Window sizes (default: [2, 5])")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Filter to specific model names")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training steps")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to sweep (default: [42])")
    parser.add_argument("--results-dir", type=str, default="results/bench",
                        help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sweep plan without training")

    # HMM extensions
    parser.add_argument("--delta", type=float, default=0.0,
                        help="Leaky reset parameter (0 = standard)")
    parser.add_argument("--coupled", action="store_true",
                        help="Enable coupled-feature mode")
    parser.add_argument("--K-hidden", type=int, default=10,
                        help="Number of hidden states (coupled mode)")
    parser.add_argument("--M-emission", type=int, default=20,
                        help="Number of emission features (coupled mode)")
    parser.add_argument("--n-parents", type=int, default=2,
                        help="Parents per emission (coupled mode)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sweep_config = SweepConfig(
        k_values=args.k or [2, 5, 10, 25],
        rho_values=args.rho or [0.0, 0.6, 0.9],
        T_values=args.T or [2, 5],
        seeds=args.seeds or [42],
    )
    train_config = TrainConfig(
        total_steps=args.steps or 30_000,
    )

    coupling = None
    if args.coupled:
        coupling = CouplingConfig(
            K_hidden=args.K_hidden,
            M_emission=args.M_emission,
            n_parents=args.n_parents,
        )

    data_config = DataConfig(
        markov=MarkovConfig(delta=args.delta),
        coupling=coupling,
        d_sae=args.M_emission if args.coupled else 128,
    )

    models = get_default_models(sweep_config.T_values)

    # Filter models if requested
    if args.models:
        models = [m for m in models if any(
            f.lower() in m.name.lower() for f in args.models
        )]

    # Build job list for display
    jobs = list(itertools.product(
        sweep_config.seeds, sweep_config.rho_values, sweep_config.k_values, models
    ))

    mode_info = "standard"
    if args.coupled:
        mode_info = f"coupled (K={args.K_hidden}, M={args.M_emission}, n_parents={args.n_parents})"
    if args.delta > 0:
        mode_info += f", delta={args.delta}"

    print(f"{'=' * 60}")
    print(f"  ARCHITECTURE COMPARISON SWEEP")
    print(f"  Device: {device}")
    print(f"  Mode: {mode_info}")
    print(f"  Steps: {train_config.total_steps:,}")
    print(f"  Seeds: {sweep_config.seeds}")
    print(f"  Rho: {sweep_config.rho_values}")
    print(f"  K: {sweep_config.k_values}")
    print(f"  T: {sweep_config.T_values}")
    print(f"  Models: {[m.name for m in models]}")
    print(f"  Total jobs: {len(jobs)}")
    print(f"{'=' * 60}")

    if args.dry_run:
        for i, (seed, rho, k, entry) in enumerate(jobs, 1):
            seed_str = f"  seed={seed}" if len(sweep_config.seeds) > 1 else ""
            print(f"  [{i:3d}/{len(jobs)}]  {entry.name:20s}  rho={rho}  k={k}{seed_str}")
        print("\n  (dry run — no training)")
        return

    run_sweep(
        models=models,
        data_config=data_config,
        train_config=train_config,
        sweep_config=sweep_config,
        device=device,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
