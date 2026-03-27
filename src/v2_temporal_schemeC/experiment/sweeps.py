"""Sweep runners — unified TopK and L1 experiment loops."""

import hashlib
import os
import time
from dataclasses import dataclass, field, asdict

import torch

from src.utils.seed import set_seed
from src.v2_temporal_schemeC.experiment.data_pipeline import (
    DataConfig, DataPipeline, build_data_pipeline,
)
from src.v2_temporal_schemeC.experiment.model_specs import ModelEntry
from src.v2_temporal_schemeC.experiment.eval_unified import EvalResult, evaluate_model


def _cache_key(model_name: str, k, data_config: DataConfig, params: dict) -> str:
    """Deterministic cache key from model name, k, data config, and training params."""
    key_parts = [
        model_name,
        f"k={k}",
        f"seed={data_config.seed}",
        f"n={data_config.num_features}",
        f"d={data_config.hidden_dim}",
        f"pi={'_'.join(f'{p:.2f}' for p in data_config.pi[:3])}",
        f"rho={'_'.join(f'{r:.2f}' for r in data_config.rho[:3])}",
        f"steps={params.get('total_steps', 0)}",
        f"lr={params.get('lr', 0)}",
        f"l1={params.get('l1_coeff', 0)}",
    ]
    raw = "|".join(key_parts)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _save_model(model, cache_dir: str, cache_id: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cache_dir, f"{cache_id}.pt"))


def _load_model(spec, cache_dir: str, cache_id: str, d_in, d_sae, k, device):
    path = os.path.join(cache_dir, f"{cache_id}.pt")
    if not os.path.exists(path):
        return None
    model = spec.create(d_in=d_in, d_sae=d_sae, k=k, device=device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def _get_generator(pipeline: DataPipeline, gen_key: str):
    """Look up the right data generator from the pipeline."""
    if gen_key == "flat":
        return pipeline.gen_flat
    elif gen_key == "seq":
        return pipeline.gen_seq
    elif gen_key == "seq_shuffled":
        return pipeline.gen_seq_shuffled
    elif gen_key.startswith("window_"):
        T = int(gen_key.split("_")[1])
        return pipeline.gen_windows[T]
    else:
        raise ValueError(f"Unknown gen_key: {gen_key}")


def _get_training_params(spec, overrides: dict) -> dict:
    """Get training params with defaults based on model type, then apply overrides."""
    from src.v2_temporal_schemeC.experiment.model_specs import (
        SAEModelSpec, TFAModelSpec, TXCDRModelSpec, TXCDRv2ModelSpec,
        StackedSAEModelSpec,
    )
    if isinstance(spec, SAEModelSpec):
        defaults = {"total_steps": 30_000, "batch_size": 4096, "lr": 3e-4}
    elif isinstance(spec, TFAModelSpec):
        defaults = {"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}
    elif isinstance(spec, (TXCDRModelSpec, TXCDRv2ModelSpec, StackedSAEModelSpec)):
        defaults = {"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}
    else:
        defaults = {"total_steps": 30_000, "batch_size": 64, "lr": 3e-4}
    defaults.update(overrides)
    return defaults


def run_topk_sweep(
    models: list[ModelEntry],
    k_values: list[int],
    data_config: DataConfig,
    device: torch.device,
    compute_auc: bool = True,
    verbose: bool = True,
    cache_dir: str | None = None,
) -> dict[str, list[EvalResult]]:
    """Run Experiment 1: TopK sweep for all models.

    Args:
        models: List of ModelEntry specs to train and evaluate.
        k_values: List of k values to sweep.
        data_config: Data configuration.
        device: Torch device.
        compute_auc: Whether to compute feature recovery AUC.
        verbose: Print progress.

    Returns:
        Dict mapping model_name -> list of EvalResult (one per k).
    """
    # Collect window sizes needed
    window_sizes = set()
    for entry in models:
        if entry.gen_key.startswith("window_"):
            window_sizes.add(int(entry.gen_key.split("_")[1]))

    pipeline = build_data_pipeline(data_config, device,
                                    window_sizes=list(window_sizes) or None)
    true_features = pipeline.true_features if compute_auc else None

    results: dict[str, list[EvalResult]] = {entry.name: [] for entry in models}

    for k in k_values:
        if verbose:
            print(f"\n  k={k}:", flush=True)

        for entry in models:
            params = _get_training_params(entry.spec, entry.training_overrides)
            params_with_l1 = {**params, "l1_coeff": 0.0}
            gen_fn = _get_generator(pipeline, entry.gen_key)

            set_seed(data_config.seed)
            t0 = time.time()

            # Try loading from cache
            cached = False
            cache_id = None
            if cache_dir:
                cache_id = _cache_key(entry.name, k, data_config, params_with_l1)
                model = _load_model(entry.spec, cache_dir, cache_id,
                                     data_config.hidden_dim, data_config.dict_width,
                                     k, device)
                if model is not None:
                    cached = True

            if not cached:
                model = entry.spec.create(
                    d_in=data_config.hidden_dim,
                    d_sae=data_config.dict_width,
                    k=k,
                    device=device,
                )
                config = entry.spec.make_train_config(
                    total_steps=params["total_steps"],
                    batch_size=params["batch_size"],
                    lr=params["lr"],
                    l1_coeff=0.0,
                    log_every=params["total_steps"],
                )
                model, _ = entry.spec.train(model, gen_fn, config, device)
                if cache_dir and cache_id:
                    _save_model(model, cache_dir, cache_id)

            result = evaluate_model(
                entry.spec, model, pipeline.eval_hidden, device,
                true_features=true_features,
                seq_len=data_config.seq_len,
            )
            results[entry.name].append(result)

            if verbose:
                auc_str = f" AUC={result.auc:.4f}" if result.auc is not None else ""
                src = "cached" if cached else f"{time.time()-t0:.0f}s"
                print(f"    {entry.name:>15}: NMSE={result.nmse:.6f} "
                      f"nL0={result.novel_l0:.1f} tL0={result.total_l0:.1f}"
                      f"{auc_str} ({src})", flush=True)

            del model
            torch.cuda.empty_cache()

    return results


def run_l1_sweep(
    models: list[ModelEntry],
    l1_coeffs: dict[str, list[float]],
    data_config: DataConfig,
    device: torch.device,
    compute_auc: bool = True,
    verbose: bool = True,
    cache_dir: str | None = None,
) -> dict[str, list[EvalResult]]:
    """Run Experiment 2: L1 Pareto sweep for all models.

    Args:
        models: List of ModelEntry specs.
        l1_coeffs: Dict mapping model_name -> list of L1 coefficients.
        data_config: Data configuration.
        device: Torch device.
        compute_auc: Whether to compute AUC.
        verbose: Print progress.

    Returns:
        Dict mapping model_name -> list of EvalResult (one per l1_coeff).
    """
    window_sizes = set()
    for entry in models:
        if entry.gen_key.startswith("window_"):
            window_sizes.add(int(entry.gen_key.split("_")[1]))

    pipeline = build_data_pipeline(data_config, device,
                                    window_sizes=list(window_sizes) or None)
    true_features = pipeline.true_features if compute_auc else None

    results: dict[str, list[EvalResult]] = {entry.name: [] for entry in models}

    for entry in models:
        coeffs = l1_coeffs.get(entry.name, [])
        if not coeffs:
            continue

        if verbose:
            print(f"\n  {entry.name} L1 sweep ({len(coeffs)} values):", flush=True)

        params = _get_training_params(entry.spec, entry.training_overrides)
        gen_fn = _get_generator(pipeline, entry.gen_key)

        for l1c in coeffs:
            set_seed(data_config.seed)
            t0 = time.time()
            params_with_l1 = {**params, "l1_coeff": l1c}

            cached = False
            cache_id = None
            if cache_dir:
                cache_id = _cache_key(entry.name, f"l1_{l1c:.6f}", data_config, params_with_l1)
                model = _load_model(entry.spec, cache_dir, cache_id,
                                     data_config.hidden_dim, data_config.dict_width,
                                     None, device)
                if model is not None:
                    cached = True

            if not cached:
                model = entry.spec.create(
                    d_in=data_config.hidden_dim,
                    d_sae=data_config.dict_width,
                    k=None,  # L1 mode
                    device=device,
                )
                config = entry.spec.make_train_config(
                    total_steps=params["total_steps"],
                    batch_size=params["batch_size"],
                    lr=params["lr"],
                    l1_coeff=l1c,
                    log_every=params["total_steps"],
                )
                model, _ = entry.spec.train(model, gen_fn, config, device)
                if cache_dir and cache_id:
                    _save_model(model, cache_dir, cache_id)

            result = evaluate_model(
                entry.spec, model, pipeline.eval_hidden, device,
                true_features=true_features,
                seq_len=data_config.seq_len,
            )
            results[entry.name].append(result)

            if verbose:
                auc_str = f" AUC={result.auc:.4f}" if result.auc is not None else ""
                src = "cached" if cached else f"{time.time()-t0:.0f}s"
                print(f"    l1={l1c:.4f}: NMSE={result.nmse:.6f} "
                      f"nL0={result.novel_l0:.2f} tL0={result.total_l0:.2f}"
                      f"{auc_str} ({src})", flush=True)

            del model
            torch.cuda.empty_cache()

    return results
