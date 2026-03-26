"""Experiment runner: sweep over model type x data configuration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from itertools import product
from typing import Any

import torch

from .config import DataConfig, SweepConfig, TrainConfig
from .data.pipeline import DataPipeline
from .metrics import EvalMetrics, evaluate
from .models import MODEL_REGISTRY
from .train import train
from .utils import get_device, set_seed


def _create_model(
    model_name: str,
    d_in: int,
    d_sae: int,
    T: int,
    k: int,
    device: torch.device,
) -> Any:
    """Instantiate a model by name from the registry."""
    cls = MODEL_REGISTRY[model_name]
    model_name_lower = model_name.lower()

    if model_name_lower == "sae":
        model = cls(d_in=d_in, d_sae=d_sae, k=k)
    elif model_name_lower == "txcdr":
        model = cls(d_in=d_in, d_sae=d_sae, T=T, k_per_pos=k)
    elif model_name_lower == "per_feature_temporal":
        model = cls(d_in=d_in, d_sae=d_sae, T=T, k=k)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def _should_skip(model_name: str, k: int, T: int, n_features: int) -> bool:
    """Skip invalid configurations."""
    if model_name == "txcdr" and k * T >= n_features:
        return True
    return False


def run_sweep(config: SweepConfig) -> list[dict]:
    """Run full model x data configuration sweep.

    Iterates over product(models, rho_values, k_values, T_values),
    trains each model, evaluates, and saves results.

    Returns:
        List of result dicts, one per (model, rho, k, T, seed) combination.
    """
    device = get_device()
    results = []

    # Create data pipeline (shared across all runs)
    pipeline = DataPipeline(config.data, device=device)

    combos = list(
        product(
            config.models,
            config.rho_values,
            config.k_values,
            config.T_values,
            range(config.n_seeds),
        )
    )

    print(f"Sweep: {len(combos)} combinations (before skips)")
    print(f"Device: {device}")
    print(f"Features: {config.data.n_features}, d: {config.data.d_model}, pi: {config.data.pi}")

    for model_name, rho, k, T, seed_idx in combos:
        if _should_skip(model_name, k, T, config.data.n_features):
            continue

        seed = config.train.seed + seed_idx
        set_seed(seed)

        print(f"\n--- {model_name} | rho={rho} | k={k} | T={T} | seed={seed} ---")

        # Create eval data for this (rho, T) combination
        eval_data = pipeline.eval_data(
            n_sequences=200, T=T, rho=rho, seed=9999 + seed_idx
        )

        # Data sampling function
        def data_fn(batch_size: int, _rho=rho, _T=T) -> torch.Tensor:
            return pipeline.sample_windows(batch_size, _T, _rho)

        # Create and train model
        model = _create_model(
            model_name,
            d_in=config.data.d_model,
            d_sae=config.data.n_features,
            T=T,
            k=k,
            device=device,
        )

        history = train(
            model=model,
            data_fn=data_fn,
            config=config.train,
            eval_data=eval_data,
            true_features=pipeline.true_features,
            silent=False,
        )

        # Final evaluation
        final_metrics = history[-1] if history else evaluate(
            model, eval_data, pipeline.true_features
        )

        result = {
            "model": model_name,
            "rho": rho,
            "k": k,
            "T": T,
            "seed": seed,
            "nmse": final_metrics.nmse,
            "l0": final_metrics.l0,
            "auc": final_metrics.auc,
            "r_at_90": final_metrics.r_at_90,
            "r_at_80": final_metrics.r_at_80,
            "mean_max_cos": final_metrics.mean_max_cos,
        }
        results.append(result)

        print(
            f"  NMSE={final_metrics.nmse:.4f}  AUC={final_metrics.auc:.3f}  "
            f"L0={final_metrics.l0:.1f}  R@0.9={final_metrics.r_at_90:.2f}"
        )

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    out_path = os.path.join(config.output_dir, "sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    sweep_config = SweepConfig()
    run_sweep(sweep_config)
