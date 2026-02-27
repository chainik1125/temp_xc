"""Main experiment runner: sweep rho x arch x sparsity x seeds."""

import json
import os

import torch

from src.utils.device import DEFAULT_DEVICE
from src.utils.logging import log, log_sweep
from src.utils.seed import set_seed
from src.v2_crosscoder_comparison.architectures import create_architecture
from src.v2_crosscoder_comparison.configs import (
    ArchitectureConfig,
    DataConfig,
    ExperimentConfig,
    ToyModelConfig,
)
from src.v2_crosscoder_comparison.data_generation import (
    build_cross_position_correlation_matrix,
    generate_two_position_batch,
)
from src.v2_crosscoder_comparison.eval import EvalResult, evaluate
from src.v2_crosscoder_comparison.plotting import (
    plot_crosscoder_advantage,
    plot_pareto_frontiers,
    plot_rho_sweep_summary,
)
from src.v2_crosscoder_comparison.toy_model import TwoPositionToyModel
from src.v2_crosscoder_comparison.train import train_architecture


# Sweep parameters
RHO_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ARCH_TYPES = ["naive_sae", "stacked_sae", "crosscoder"]
TOP_K_VALUES = [5, 8, 11, 14, 17, 20]
SEEDS = [42, 43, 44]


def run_single(
    rho: float,
    arch_type: str,
    top_k: int,
    seed: int,
    toy_model: TwoPositionToyModel,
    device: torch.device = DEFAULT_DEVICE,
) -> EvalResult:
    """Run a single experiment configuration.

    Args:
        rho: Cross-position correlation.
        arch_type: Architecture type.
        top_k: TopK sparsity level.
        seed: Random seed.
        toy_model: Shared toy model instance.
        device: Torch device.

    Returns:
        EvalResult for this run.
    """
    set_seed(seed)

    data_cfg = DataConfig(
        num_features=toy_model.cfg.num_features,
        rho=rho,
    )
    arch_cfg = ArchitectureConfig(
        arch_type=arch_type,
        d_sae=toy_model.cfg.hidden_dim,  # 2x expansion since hidden_dim = 2 * num_features
        top_k=top_k,
    )
    exp_cfg = ExperimentConfig(
        toy_model=toy_model.cfg,
        data=data_cfg,
        architecture=arch_cfg,
        seed=seed,
    )

    corr_matrix = build_cross_position_correlation_matrix(
        data_cfg.num_features, rho
    ).to(device)

    def generate_hidden_fn(batch_size: int) -> torch.Tensor:
        feature_acts = generate_two_position_batch(
            batch_size, data_cfg, corr_matrix, device
        )
        return toy_model(feature_acts.to(device))

    arch = create_architecture(arch_cfg, toy_model.cfg.hidden_dim)

    train_architecture(arch_type, arch, generate_hidden_fn, exp_cfg, device)

    result = evaluate(
        arch=arch,
        generate_hidden_fn=generate_hidden_fn,
        feature_directions=toy_model.embedding.data,
        arch_type=arch_type,
        rho=rho,
        top_k=top_k,
        seed=seed,
    )

    return result


def run_sweep(
    rho_values: list[float] | None = None,
    arch_types: list[str] | None = None,
    top_k_values: list[int] | None = None,
    seeds: list[int] | None = None,
    results_dir: str = "results",
    plots_dir: str = "plots",
) -> dict[float, dict[str, list[EvalResult]]]:
    """Run the full experiment sweep.

    Args:
        rho_values: Rho values to sweep over.
        arch_types: Architecture types to compare.
        top_k_values: TopK values to sweep.
        seeds: Random seeds for replication.
        results_dir: Directory for JSON results.
        plots_dir: Directory for plots.

    Returns:
        Nested dict: rho -> arch_type -> list of EvalResults.
    """
    rho_values = rho_values or RHO_VALUES
    arch_types = arch_types or ARCH_TYPES
    top_k_values = top_k_values or TOP_K_VALUES
    seeds = seeds or SEEDS

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    device = DEFAULT_DEVICE
    total_runs = len(rho_values) * len(arch_types) * len(top_k_values) * len(seeds)
    log("info", f"starting sweep | total_runs={total_runs} | device={device}")

    # Create toy model once (shared embedding)
    toy_cfg = ToyModelConfig()
    set_seed(0)  # Fixed seed for toy model
    toy_model = TwoPositionToyModel(toy_cfg).to(device)
    log("info", "initialized toy_model", num_features=toy_cfg.num_features, hidden_dim=toy_cfg.hidden_dim)

    results_by_rho: dict[float, dict[str, list[EvalResult]]] = {}
    run_idx = 0

    for rho in rho_values:
        results_by_rho[rho] = {}

        for arch_type in arch_types:
            results_by_rho[rho][arch_type] = []

            for top_k in top_k_values:
                for seed in seeds:
                    run_idx += 1
                    log_sweep(
                        run_idx, total_runs,
                        arch=arch_type, rho=rho, top_k=top_k, seed=seed,
                    )

                    result = run_single(
                        rho=rho,
                        arch_type=arch_type,
                        top_k=top_k,
                        seed=seed,
                        toy_model=toy_model,
                        device=device,
                    )
                    results_by_rho[rho][arch_type].append(result)

        # Save results per rho
        rho_results = {}
        for arch_type, results in results_by_rho[rho].items():
            rho_results[arch_type] = [r.to_dict() for r in results]

        results_path = os.path.join(results_dir, f"results_rho_{rho:.1f}.json")
        with open(results_path, "w") as f:
            json.dump(rho_results, f, indent=2)
        log("done", f"saved {results_path}")

        # Pareto plot per rho
        pareto_path = os.path.join(plots_dir, f"pareto_frontier_rho_{rho:.1f}.png")
        plot_pareto_frontiers(results_by_rho[rho], rho, pareto_path)
        log("done", f"saved {pareto_path}")

    # Summary plots
    plot_rho_sweep_summary(
        results_by_rho,
        os.path.join(plots_dir, "rho_sweep_summary.png"),
    )
    log("done", f"saved {os.path.join(plots_dir, 'rho_sweep_summary.png')}")

    # Crosscoder advantage plot
    rho_vals_sorted = sorted(results_by_rho.keys())
    advantages = []
    for rho in rho_vals_sorted:
        cc_results = results_by_rho[rho].get("crosscoder", [])
        cc_cos_sim = max((r.mean_max_cos_sim for r in cc_results), default=0.0)

        baseline_cos_sim = 0.0
        for arch in ["naive_sae", "stacked_sae"]:
            arch_results = results_by_rho[rho].get(arch, [])
            if arch_results:
                baseline_cos_sim = max(
                    baseline_cos_sim,
                    max(r.mean_max_cos_sim for r in arch_results),
                )
        advantages.append(cc_cos_sim - baseline_cos_sim)

    plot_crosscoder_advantage(
        rho_vals_sorted,
        advantages,
        os.path.join(plots_dir, "crosscoder_advantage.png"),
    )
    log("done", f"saved {os.path.join(plots_dir, 'crosscoder_advantage.png')}")

    # Summary log
    for rho in rho_vals_sorted:
        for arch_type in arch_types:
            results = results_by_rho[rho].get(arch_type, [])
            if results:
                avg_cos_sim = sum(r.mean_max_cos_sim for r in results) / len(results)
                log("summary", f"rho={rho:.1f}", arch=arch_type, avg_cos_sim=avg_cos_sim)

    log("done", "sweep complete")
    return results_by_rho


if __name__ == "__main__":
    run_sweep()
