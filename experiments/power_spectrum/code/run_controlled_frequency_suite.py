"""Run the controlled Shamir and frequency-HMM benchmark.

This runner is intentionally experiment-local.  It compares a token SAE, a
matched-support TXC, Spectral v1, and a global-selection frequency-Matryoshka
variant on:

- the exact polynomial-clock/Shamir construction at several privacy
  thresholds and window lengths; and
- factorial two-state HMMs with low-, high-, and mixed-frequency modes.

Representation training, probe training, and final evaluation use independent
episode pools with a shared observation dictionary.  Completed cells are
append-only, final model states are retained, and a conservative persistent
ledger bounds incremental GPU spend.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import statistics
import time
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.power_spectrum.code.controlled_tasks import (
    FactorialHMMBatch,
    NarrowbandSourceBatch,
    ShamirClockBatch,
    dct_basis,
    expected_dct_energy,
    generate_factorial_hmm_splits,
    generate_narrowband_source_splits,
    generate_shamir_splits,
    recover_leading_coefficient,
)
from experiments.power_spectrum.code.run_synthetic_benchmark import (
    BudgetGuard,
    StopRequested,
    _append_jsonl,
    _atomic_json,
    _canonical_json,
    _stable_id,
    _utc_now,
    latest_results,
)


POWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = POWER_ROOT / "configs" / "controlled_frequency_suite.json"
DEFAULT_RESULTS = POWER_ROOT / "results" / "controlled_frequency_suite"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("controlled suite requires schema_version=1")
    names = [task["name"] for task in config["tasks"]]
    if len(names) != len(set(names)):
        raise ValueError("task names must be unique")
    model_names = [model["name"] for model in config["models"]]
    if len(model_names) != len(set(model_names)):
        raise ValueError("model names must be unique")
    for task in config["tasks"]:
        window = int(task["window"])
        if window < 1 or window > int(task["sequence_length"]):
            raise ValueError(f"{task['name']}: invalid window={window}")
        if task["family"] == "shamir":
            if window >= int(task["q"]):
                raise ValueError(
                    f"{task['name']}: window must be smaller than q so times are distinct"
                )
        elif task["family"] not in {"factorial_hmm", "narrowband_sources"}:
            raise ValueError(f"{task['name']}: unknown family {task['family']!r}")
    prior = float(config["overall_spend"]["estimated_prior_usd"])
    incremental = float(config["budget"]["max_total_usd"])
    cap = float(config["overall_spend"]["cap_usd"])
    if prior + incremental > cap:
        raise ValueError("prior estimate plus incremental hard cap exceeds overall cap")
    return config


def _model_k(model: dict[str, Any], window: int) -> int:
    rule = model["sparsity_rule"]
    if rule == "one_per_token":
        return 1
    if rule == "window_length":
        return window
    raise ValueError(f"unknown sparsity rule {rule!r}")


def _model_d_sae(task: dict[str, Any], model: dict[str, Any]) -> int:
    overrides = task.get("d_sae_by_model", {})
    return int(overrides.get(model["name"], task["d_sae"]))


def _training_identity(
    config: dict[str, Any],
    task: dict[str, Any],
    model: dict[str, Any],
    seed: int,
    *,
    n_steps: int,
) -> dict[str, Any]:
    consumes = str(model["consumes"])
    # A token SAE is independent of evaluation-window length and is trained
    # once per underlying dataset group/seed, then reused for every W.
    training_window = 1 if consumes == "token" else int(task["window"])
    training_task = str(task["group"]) if consumes == "token" else str(task["name"])
    return {
        "run_name": config["run_name"],
        "training_task": training_task,
        "family": task["family"],
        "model": model["name"],
        "class_path": model["class_path"],
        "implementation_version": model["implementation_version"],
        "hparams": model.get("hparams", {}),
        "seed": int(seed),
        "d_in": int(task["d_in"]),
        "d_sae": _model_d_sae(task, model),
        "training_window": training_window,
        "k_pos": _model_k(model, int(task["window"])),
        "n_steps": int(n_steps),
        "learning_rate": float(config["training"]["learning_rate"]),
        "batch_tokens": int(config["training"]["batch_tokens"]),
    }


def enumerate_cells(
    config: dict[str, Any],
    *,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    smoke_config = config["smoke"]
    cells: list[dict[str, Any]] = []
    for task in config["tasks"]:
        if smoke and task["name"] not in smoke_config["tasks"]:
            continue
        task_models = {
            str(name) for name in task.get("models", [m["name"] for m in config["models"]])
        }
        for model in config["models"]:
            if model["name"] not in task_models:
                continue
            if smoke and model["name"] not in smoke_config["models"]:
                continue
            if int(task["window"]) in {
                int(value) for value in model.get("skip_window_lengths", [])
            }:
                continue
            seeds = [int(smoke_config["seed"])] if smoke else config["seeds"]
            for seed in seeds:
                n_steps = int(smoke_config["n_steps"]) if smoke else int(task["n_steps"])
                identity = _training_identity(
                    config,
                    task,
                    model,
                    int(seed),
                    n_steps=n_steps,
                )
                core = {
                    "run_name": config["run_name"],
                    "task": task["name"],
                    "group": task["group"],
                    "family": task["family"],
                    "model": model["name"],
                    "seed": int(seed),
                    "window": int(task["window"]),
                    "d_in": int(task["d_in"]),
                    "d_sae": _model_d_sae(task, model),
                    "k_pos": _model_k(model, int(task["window"])),
                    "n_steps": n_steps,
                    "primary_metric": task["primary_metric"],
                    "smoke": smoke,
                }
                cells.append(
                    {
                        **core,
                        "training_id": _stable_id(identity),
                        "cell_id": _stable_id({**core, "training_id": _stable_id(identity)}),
                        "training_identity": identity,
                        "task_spec": task,
                        "model_spec": model,
                    }
                )
    return cells


def build_plan(
    config: dict[str, Any],
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    cells = enumerate_cells(config, smoke=smoke)
    unique_training: dict[str, int] = {}
    for cell in cells:
        unique_training[cell["training_id"]] = max(
            unique_training.get(cell["training_id"], 0),
            int(cell["n_steps"]),
        )
    planning = config["planning"]
    seconds = (
        float(planning["setup_seconds"])
        + sum(unique_training.values()) / float(planning["estimated_steps_per_second"])
        + len(cells) * float(planning["estimated_eval_seconds_per_cell"])
    )
    budget = config["budget"]
    effective_rate = float(budget["assumed_usd_per_gpu_hour"]) * float(
        budget["cost_overhead_multiplier"]
    )
    estimated_cost = seconds / 3600.0 * effective_rate
    usable = float(budget["max_total_usd"]) - float(budget["reserve_usd"])
    return {
        "run_name": config["run_name"],
        "smoke": smoke,
        "evaluation_cells": len(cells),
        "unique_training_runs": len(unique_training),
        "total_optimizer_steps": sum(unique_training.values()),
        "task_cell_counts": {
            task["name"]: sum(cell["task"] == task["name"] for cell in cells)
            for task in config["tasks"]
            if any(cell["task"] == task["name"] for cell in cells)
        },
        "estimated_gpu_hours": round(seconds / 3600.0, 3),
        "effective_assumed_usd_per_hour": round(effective_rate, 3),
        "estimated_cost_usd": round(estimated_cost, 2),
        "usable_budget_usd": round(usable, 2),
        "max_session_hours": float(budget["max_session_hours"]),
        "within_cost_plan": estimated_cost <= usable,
        "within_time_plan": seconds / 3600.0 <= float(budget["max_session_hours"]),
        "overall_cap_usd": float(config["overall_spend"]["cap_usd"]),
        "estimated_prior_usd": float(config["overall_spend"]["estimated_prior_usd"]),
        "worst_case_overall_usd": float(config["overall_spend"]["estimated_prior_usd"])
        + float(budget["max_total_usd"]),
    }


def _import_model(spec: dict[str, Any]):
    module_name, class_name = str(spec["class_path"]).split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not inspect.isclass(cls):
        raise TypeError(f"{spec['class_path']} is not a class")
    return cls


def _model_kwargs(cell: dict[str, Any]) -> dict[str, Any]:
    spec = cell["model_spec"]
    return {
        **spec.get("hparams", {}),
        "d_in": int(cell["d_in"]),
        "d_sae": int(cell["d_sae"]),
        "T": 1 if spec["consumes"] == "token" else int(cell["window"]),
        "k_pos": int(cell["k_pos"]),
    }


def _checkpoint_path(results_dir: Path, training_id: str) -> Path:
    return results_dir / "checkpoints" / training_id / "model.pt"


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _materialize_group(
    config: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, ShamirClockBatch | FactorialHMMBatch | NarrowbandSourceBatch]:
    sizes = {key: int(value) for key, value in config["split_sizes"].items()}
    seeds = {key: int(value) for key, value in config["split_seeds"].items()}
    if task["family"] == "shamir":
        return generate_shamir_splits(
            h=int(task["h"]),
            q=int(task["q"]),
            d=int(task["d_in"]),
            sigma=float(task["sigma"]),
            seq_len=int(task["sequence_length"]),
            split_sizes=sizes,
            split_seeds=seeds,
            alphabet_seed=0,
        )
    if task["family"] == "factorial_hmm":
        return generate_factorial_hmm_splits(
            lambdas=task["lambdas"],
            d=int(task["d_in"]),
            sigma=float(task["sigma"]),
            seq_len=int(task["sequence_length"]),
            split_sizes=sizes,
            split_seeds=seeds,
            emission_seed=0,
        )
    return generate_narrowband_source_splits(
        frequencies=task["frequencies"],
        d=int(task["d_in"]),
        sigma=float(task["sigma"]),
        seq_len=int(task["sequence_length"]),
        split_sizes=sizes,
        split_seeds=seeds,
        emission_seed=0,
        amplitude_range=tuple(task.get("amplitude_range", (0.75, 1.25))),
        min_frequency_separation=float(task.get("min_frequency_separation", 1.0 / 16)),
    )


def _sample_training_batch(
    data: ShamirClockBatch | FactorialHMMBatch | NarrowbandSourceBatch,
    *,
    consumes: str,
    window: int,
    batch_tokens: int,
    generator: torch.Generator,
    device: str,
) -> torch.Tensor:
    x = data.x
    n_sequences, sequence_length, d_in = x.shape
    if consumes == "token":
        flat = x.reshape(-1, d_in)
        indices = torch.randint(
            0,
            flat.shape[0],
            (batch_tokens,),
            generator=generator,
        )
        return flat[indices].to(device=device, dtype=torch.float32)
    batch_size = max(1, batch_tokens // window)
    sequence_indices = torch.randint(
        0,
        n_sequences,
        (batch_size,),
        generator=generator,
    )
    starts = torch.randint(
        0,
        sequence_length - window + 1,
        (batch_size,),
        generator=generator,
    )
    positions = starts[:, None] + torch.arange(window)[None, :]
    rows = sequence_indices[:, None]
    return x[rows, positions].to(device=device, dtype=torch.float32)


def train_or_load(
    config: dict[str, Any],
    cell: dict[str, Any],
    train_data: ShamirClockBatch | FactorialHMMBatch | NarrowbandSourceBatch,
    results_dir: Path,
    budget: BudgetGuard,
):
    seed = int(cell["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cls = _import_model(cell["model_spec"])
    model = cls(**_model_kwargs(cell))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    checkpoint = _checkpoint_path(results_dir, cell["training_id"])
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("training_id") != cell["training_id"]:
            raise RuntimeError(f"checkpoint identity mismatch at {checkpoint}")
        model.load_state_dict(payload["model"])
        model.eval()
        return model, {
            "loaded_checkpoint": True,
            "checkpoint": str(checkpoint),
            "end_step": int(payload["step"]),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "device": device,
        }

    training = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
    )
    warmup_steps = int(training["warmup_steps"])
    n_steps = int(cell["n_steps"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    precision = str(training["precision"])
    use_autocast = device == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    last_metrics: dict[str, float] = {}
    started = time.monotonic()
    model.train()
    for step in range(n_steps):
        if step % int(training["budget_check_every_steps"]) == 0:
            budget.check()
        learning_rate = float(training["learning_rate"])
        if warmup_steps and step < warmup_steps:
            learning_rate *= (step + 1) / warmup_steps
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        batch = _sample_training_batch(
            train_data,
            consumes=str(cell["model_spec"]["consumes"]),
            window=int(cell["window"]),
            batch_tokens=int(training["batch_tokens"]),
            generator=generator,
            device=device,
        )
        model.pre_step()
        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if use_autocast
            else contextlib.nullcontext()
        )
        with autocast_context:
            metrics = model.train_step(batch)
            loss = metrics["loss"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        model.post_step()
        last_metrics = {
            key: float(value.detach().item())
            for key, value in metrics.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        }
        last_metrics["gradient_norm"] = float(gradient_norm.detach().item())
        log_every = int(training["log_every_steps"])
        if (step + 1) % log_every == 0 or step + 1 == n_steps:
            print(
                f"  train {cell['model']}/{cell['task']}/s{seed} "
                f"{step + 1}/{n_steps} loss={last_metrics.get('loss', math.nan):.5f}",
                flush=True,
            )
    model.eval()
    _atomic_torch_save(
        checkpoint,
        {
            "schema_version": 1,
            "training_id": cell["training_id"],
            "step": n_steps,
            "model_kwargs": _model_kwargs(cell),
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "completed_at": _utc_now(),
        },
    )
    return model, {
        "loaded_checkpoint": False,
        "checkpoint": str(checkpoint),
        "end_step": n_steps,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": device,
        "precision": precision,
        "last_metrics": last_metrics,
    }


@torch.no_grad()
def encode_windows(
    model,
    x: torch.Tensor,
    *,
    window: int,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return probe features and native codes.

    Token models expose concatenated ordered token codes to the Shamir probe;
    native codes retain ``(N,W,H)``.  Window models expose their shared code
    and native codes are ``(N,H)``.
    """

    device = next(model.parameters()).device
    native_batches = []
    windows = x[:, :window]
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size].to(
            device=device,
            dtype=torch.float32,
        )
        code = model.encode(batch).detach().float().cpu()
        if model.consumes == "window":
            if code.ndim == 3:
                if code.shape[1] != 1:
                    raise ValueError(f"unexpected shared code shape {tuple(code.shape)}")
                code = code[:, 0]
        native_batches.append(code)
    native = torch.cat(native_batches).numpy()
    features = native.reshape(native.shape[0], -1)
    return features, native


@torch.no_grad()
def reconstruct_windows(
    model,
    x: torch.Tensor,
    *,
    window: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Reconstruct the first analysis window of each independent episode."""

    device = next(model.parameters()).device
    outputs = []
    windows = x[:, :window]
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size].to(
            device=device,
            dtype=torch.float32,
        )
        outputs.append(model.decode(model.encode(batch)).detach().float().cpu())
    return torch.cat(outputs).numpy()


def _row_shuffle(x: torch.Tensor, window: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.rand(x.shape[0], window, generator=generator).argsort(dim=1)
    rows = torch.arange(x.shape[0])[:, None]
    shuffled = x[:, :window].clone()
    shuffled = shuffled[rows, order]
    return shuffled


def _fit_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    eval_y: np.ndarray,
    *,
    seed: int,
):
    from scipy import sparse
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-4,
            average=True,
            random_state=seed,
        ),
    )
    classifier.fit(sparse.csr_matrix(train_x), train_y)
    prediction = classifier.predict(sparse.csr_matrix(eval_x))
    accuracy = float(balanced_accuracy_score(eval_y, prediction))
    return classifier, accuracy


def _predict_classifier(classifier, values: np.ndarray) -> np.ndarray:
    from scipy import sparse

    return classifier.predict(sparse.csr_matrix(values))


def _secret_selectivity(
    probe_code: np.ndarray,
    probe_secret: np.ndarray,
    eval_code: np.ndarray,
    eval_secret: np.ndarray,
    q: int,
) -> dict[str, float]:
    """Cross-split single-feature secret selectivity.

    Preferred classes and the top-q feature set are chosen only on the probe
    split.  Their activation-mass precision is then measured on the independent
    eval split.  A minimum fire count prevents one-off activations from looking
    perfectly selective in a wide sparse dictionary.
    """

    probe = probe_code.reshape(probe_code.shape[0], -1)
    evaluation = eval_code.reshape(eval_code.shape[0], -1)
    probe_mass = np.stack(
        [probe[probe_secret == value].sum(axis=0) for value in range(q)],
        axis=0,
    )
    eval_mass = np.stack(
        [evaluation[eval_secret == value].sum(axis=0) for value in range(q)],
        axis=0,
    )
    minimum_fires = max(10, int(math.ceil(0.005 * probe.shape[0])))
    probe_fires = (probe > 0).sum(axis=0)
    eval_fires = (evaluation > 0).sum(axis=0)
    eligible = (
        (probe_fires >= minimum_fires)
        & (eval_fires >= minimum_fires)
        & (probe_mass.sum(axis=0) > 1e-12)
        & (eval_mass.sum(axis=0) > 1e-12)
    )
    if not eligible.any():
        return {
            "selective_feature_count": 0,
            "top_q_selectivity": 0.0,
            "top_q_secret_coverage": 0.0,
            "minimum_fires_per_split": minimum_fires,
        }
    probe_class_counts = np.bincount(probe_secret, minlength=q).astype(np.float64)
    probe_means = probe_mass[:, eligible] / probe_class_counts[:, None]
    preferred = probe_means.argmax(axis=0)
    probe_precision = probe_means.max(axis=0) / probe_means.sum(axis=0).clip(1e-12)
    probe_score = (probe_precision - 1.0 / q) / (1.0 - 1.0 / q)
    eval_eligible_mass = eval_mass[:, eligible]
    columns = np.arange(eval_eligible_mass.shape[1])
    eval_precision = eval_eligible_mass[preferred, columns] / eval_eligible_mass.sum(axis=0).clip(
        1e-12
    )
    eval_score = (eval_precision - 1.0 / q) / (1.0 - 1.0 / q)
    count = min(q, probe_score.shape[0])
    order = np.argsort(probe_score)[-count:]
    return {
        "selective_feature_count": int(eligible.sum()),
        "top_q_selectivity": float(eval_score[order].mean()),
        "top_q_secret_coverage": float(len(set(preferred[order].tolist())) / q),
        "minimum_fires_per_split": minimum_fires,
    }


def _spectral_usage(model, native_code: np.ndarray) -> dict[str, Any]:
    if not hasattr(model, "band_of_features"):
        return {}
    slices = list(model.band_of_features())
    code = native_code.reshape(native_code.shape[0], -1)
    activation_energy = np.array(
        [np.square(code[:, start:stop]).sum() for start, stop in slices],
        dtype=np.float64,
    )
    activation_mass = np.array(
        [code[:, start:stop].sum() for start, stop in slices],
        dtype=np.float64,
    )
    l0 = np.array(
        [(code[:, start:stop] > 0).sum() for start, stop in slices],
        dtype=np.float64,
    )
    device = next(model.parameters()).device
    z = torch.from_numpy(code).to(device=device, dtype=torch.float32)
    decoder = model._dec_full().detach().float()
    reconstructed = torch.einsum("bh,htd->btd", z, decoder)
    basis = dct_basis(reconstructed.shape[1]).to(
        device=device,
        dtype=torch.float32,
    )
    coefficients = torch.einsum("wt,btd->bwd", basis, reconstructed)
    decoded_frequency_energy = (
        coefficients.square().sum(dim=(0, 2)).cpu().numpy().astype(np.float64)
    )
    bias = model.b_dec.detach().float()
    bias_coefficients = torch.einsum("wt,td->wd", basis, bias)
    bias_frequency_energy = bias_coefficients.square().sum(dim=1).cpu().numpy().astype(np.float64)

    def shares(values: np.ndarray) -> list[float]:
        total = float(values.sum())
        return (values / total).tolist() if total > 0 else [0.0 for _ in values]

    usage = {
        "bands": [list(map(int, band)) for band in model.bands],
        "band_slices": [[int(start), int(stop)] for start, stop in slices],
        "allocated_atoms": [int(value) for value in model.h_per_band],
        "nominal_k_per_band": [
            int(value) for value in getattr(model, "selection_k_per_band", model.k_per_band)
        ],
        "activation_energy_share": shares(activation_energy),
        "activation_mass_share": shares(activation_mass),
        "selection_event_share": shares(l0),
        "decoded_frequency_energy_share": shares(decoded_frequency_energy),
        "bias_frequency_energy_share": shares(bias_frequency_energy),
    }
    if hasattr(model, "learned_frequency_weights"):
        learned = model.learned_frequency_weights().detach().float().cpu().numpy()
        prior = model.frequency_weight_prior.detach().float().cpu().numpy()
        full_prior = np.zeros(len(slices), dtype=np.float64)
        full_prior[list(model.active_bands)] = prior
        usage["learned_frequency_weight"] = learned.tolist()
        usage["frequency_weight_prior"] = full_prior.tolist()
        usage["frequency_weight_lift"] = np.divide(
            learned,
            full_prior,
            out=np.zeros_like(learned, dtype=np.float64),
            where=full_prior > 0,
        ).tolist()
    return usage


@torch.no_grad()
def _reconstruction_metrics(
    model,
    x: torch.Tensor,
    *,
    window: int,
) -> dict[str, float]:
    device = next(model.parameters()).device
    batch = x[: min(1024, x.shape[0]), :window].to(
        device=device,
        dtype=torch.float32,
    )
    code = model.encode(batch)
    reconstruction = model.decode(code)
    error = (batch - reconstruction).square().sum()
    centered = batch - batch.mean(dim=(0, 1), keepdim=True)
    denominator = centered.square().sum().clamp_min(1e-12)
    l0 = (code != 0).float().reshape(code.shape[0], -1).sum(dim=1).mean()
    return {
        "nmse": float((error / denominator).item()),
        "l0_per_window": float(l0.item()),
        "l0_per_token": float((l0 / window).item()),
    }


def evaluate_shamir(
    model,
    task: dict[str, Any],
    probe_data: ShamirClockBatch,
    eval_data: ShamirClockBatch,
    *,
    seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score

    window = int(task["window"])
    q = int(task["q"])
    h = int(task["h"])
    probe_code, probe_native = encode_windows(model, probe_data.x, window=window)
    eval_code, eval_native = encode_windows(model, eval_data.x, window=window)
    classifier, balanced_accuracy = _fit_classifier(
        probe_code,
        probe_data.secret.numpy(),
        eval_code,
        eval_data.secret.numpy(),
        seed=seed,
    )
    chance = 1.0 / q
    recovery = (balanced_accuracy - chance) / (1.0 - chance)
    permutation = np.random.default_rng(seed + 1000).permutation(probe_data.secret.numpy())
    _, shuffled_label_accuracy = _fit_classifier(
        probe_code,
        permutation,
        eval_code,
        eval_data.secret.numpy(),
        seed=seed + 1,
    )
    shuffled_x = _row_shuffle(eval_data.x, window, seed + 2000)
    shuffled_code, _ = encode_windows(model, shuffled_x, window=window)
    time_shuffled_accuracy = float(
        balanced_accuracy_score(
            eval_data.secret.numpy(),
            _predict_classifier(classifier, shuffled_code),
        )
    )
    raw_probe = probe_data.x[:, :window].reshape(probe_data.x.shape[0], -1).numpy()
    raw_eval = eval_data.x[:, :window].reshape(eval_data.x.shape[0], -1).numpy()
    _, raw_accuracy = _fit_classifier(
        raw_probe,
        probe_data.secret.numpy(),
        raw_eval,
        eval_data.secret.numpy(),
        seed=seed + 2,
    )
    if window >= h + 1:
        symbol_scores = torch.einsum(
            "ntd,qd->ntq",
            eval_data.x[:, : h + 1],
            eval_data.alphabet,
        )
        observed_symbols = symbol_scores.argmax(dim=-1)
        oracle_prediction = recover_leading_coefficient(observed_symbols, q)
        oracle_accuracy = float((oracle_prediction == eval_data.secret).double().mean().item())
    else:
        oracle_accuracy = chance
    metrics: dict[str, Any] = {
        "secret_balanced_accuracy": balanced_accuracy,
        "secret_recovery": float(recovery),
        "secret_chance": chance,
        "shuffled_label_balanced_accuracy": shuffled_label_accuracy,
        "time_shuffled_balanced_accuracy": time_shuffled_accuracy,
        "raw_window_balanced_accuracy": raw_accuracy,
        "symbolic_interpolation_oracle": oracle_accuracy,
        "identifiable": bool(window >= h + 1),
        **_secret_selectivity(
            probe_native,
            probe_data.secret.numpy(),
            eval_native,
            eval_data.secret.numpy(),
            q,
        ),
        **_reconstruction_metrics(model, eval_data.x, window=window),
    }
    usage = _spectral_usage(model, eval_native)
    if usage:
        metrics["spectral_usage"] = usage
        band_recovery = []
        for band_index, (start, stop) in enumerate(model.band_of_features()):
            _, accuracy = _fit_classifier(
                probe_native[:, start:stop],
                probe_data.secret.numpy(),
                eval_native[:, start:stop],
                eval_data.secret.numpy(),
                seed=seed + 10 + band_index,
            )
            band_recovery.append(float((accuracy - chance) / (1.0 - chance)))
        metrics["band_secret_recovery"] = band_recovery
    return metrics


def _fit_hmm_probe(
    probe_native: np.ndarray,
    probe_states: np.ndarray,
    eval_native: np.ndarray,
    eval_states: np.ndarray,
    *,
    consumes: str,
):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    if consumes == "token":
        train_x = probe_native.reshape(-1, probe_native.shape[-1])
        train_y = probe_states.reshape(-1, probe_states.shape[-1])
        test_x = eval_native.reshape(-1, eval_native.shape[-1])
        output_shape = eval_states.shape
    else:
        train_x = probe_native.reshape(probe_native.shape[0], -1)
        train_y = probe_states.reshape(probe_states.shape[0], -1)
        test_x = eval_native.reshape(eval_native.shape[0], -1)
        output_shape = eval_states.shape
    probe = Ridge(alpha=1.0).fit(train_x, train_y)
    prediction = probe.predict(test_x).reshape(output_shape)
    per_source = [
        float(
            r2_score(
                eval_states[:, :, source].reshape(-1),
                prediction[:, :, source].reshape(-1),
            )
        )
        for source in range(eval_states.shape[-1])
    ]
    return probe, prediction, per_source


def _predict_hmm_probe(
    probe,
    native: np.ndarray,
    *,
    consumes: str,
    output_shape: tuple[int, ...],
) -> np.ndarray:
    if consumes == "token":
        features = native.reshape(-1, native.shape[-1])
    else:
        features = native.reshape(native.shape[0], -1)
    return probe.predict(features).reshape(output_shape)


def evaluate_hmm(
    model,
    task: dict[str, Any],
    probe_data: FactorialHMMBatch,
    eval_data: FactorialHMMBatch,
    *,
    seed: int,
) -> dict[str, Any]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    window = int(task["window"])
    consumes = str(model.consumes)
    _, probe_native = encode_windows(model, probe_data.x, window=window)
    _, eval_native = encode_windows(model, eval_data.x, window=window)
    probe_states = probe_data.states[:, :window].numpy().astype(np.float64)
    eval_states = eval_data.states[:, :window].numpy().astype(np.float64)
    ridge, _, per_source = _fit_hmm_probe(
        probe_native,
        probe_states,
        eval_native,
        eval_states,
        consumes=consumes,
    )
    reconstructed = torch.from_numpy(reconstruct_windows(model, eval_data.x, window=window)).to(
        dtype=eval_data.emissions.dtype
    )
    direct_prediction = torch.einsum(
        "ntd,jd->ntj",
        reconstructed,
        eval_data.emissions,
    ).numpy()
    direct_per_source = [
        float(
            r2_score(
                eval_states[:, :, source].reshape(-1),
                direct_prediction[:, :, source].reshape(-1),
            )
        )
        for source in range(eval_states.shape[-1])
    ]
    raw_projection = torch.einsum(
        "ntd,jd->ntj",
        eval_data.x[:, :window],
        eval_data.emissions,
    ).numpy()
    raw_projection_per_source = [
        float(
            r2_score(
                eval_states[:, :, source].reshape(-1),
                raw_projection[:, :, source].reshape(-1),
            )
        )
        for source in range(eval_states.shape[-1])
    ]
    shuffled_x = _row_shuffle(eval_data.x, window, seed + 3000)
    _, shuffled_native = encode_windows(model, shuffled_x, window=window)
    shuffled_prediction = _predict_hmm_probe(
        ridge,
        shuffled_native,
        consumes=consumes,
        output_shape=eval_states.shape,
    )
    shuffled_r2 = float(
        r2_score(
            eval_states.reshape(-1),
            shuffled_prediction.reshape(-1),
        )
    )
    raw_probe_x = probe_data.x[:, :window].reshape(-1, int(task["d_in"])).numpy()
    raw_probe_y = probe_states.reshape(-1, probe_states.shape[-1])
    raw_eval_x = eval_data.x[:, :window].reshape(-1, int(task["d_in"])).numpy()
    raw_eval_y = eval_states.reshape(-1, eval_states.shape[-1])
    raw_ridge = Ridge(alpha=1.0).fit(raw_probe_x, raw_probe_y)
    raw_prediction = raw_ridge.predict(raw_eval_x)
    raw_per_source = [
        float(r2_score(raw_eval_y[:, source], raw_prediction[:, source]))
        for source in range(raw_eval_y.shape[-1])
    ]
    metrics: dict[str, Any] = {
        "direct_latent_r2": statistics.fmean(direct_per_source),
        "direct_latent_r2_per_source": direct_per_source,
        "latent_r2": statistics.fmean(per_source),
        "latent_r2_per_source": per_source,
        "raw_token_r2": statistics.fmean(raw_per_source),
        "raw_token_r2_per_source": raw_per_source,
        "raw_projection_r2": statistics.fmean(raw_projection_per_source),
        "raw_projection_r2_per_source": raw_projection_per_source,
        "time_shuffled_latent_r2": shuffled_r2,
        "lambdas": [float(value) for value in eval_data.lambdas],
        **_reconstruction_metrics(model, eval_data.x, window=window),
    }
    usage = _spectral_usage(model, eval_native)
    if usage:
        metrics["spectral_usage"] = usage
        band_source_r2 = []
        for start, stop in model.band_of_features():
            _, _, source_values = _fit_hmm_probe(
                probe_native[:, start:stop],
                probe_states,
                eval_native[:, start:stop],
                eval_states,
                consumes="window",
            )
            band_source_r2.append(source_values)
        recovered = np.asarray(band_source_r2, dtype=np.float64).T
        expected_frequency = expected_dct_energy(
            eval_data.lambdas,
            window,
        ).numpy()
        expected_band = np.stack(
            [expected_frequency[:, list(map(int, band))].sum(axis=1) for band in model.bands],
            axis=1,
        )
        expected_band /= expected_band.sum(axis=1, keepdims=True)
        expected_winner = expected_band.argmax(axis=1)
        recovered_winner = recovered.argmax(axis=1)
        metrics["band_latent_r2_per_source"] = recovered.tolist()
        metrics["expected_dct_band_energy_per_source"] = expected_band.tolist()
        metrics["expected_band_per_source"] = expected_winner.astype(int).tolist()
        metrics["recovered_band_per_source"] = recovered_winner.astype(int).tolist()
        metrics["band_localization_accuracy"] = float((expected_winner == recovered_winner).mean())
    return metrics


def evaluate_narrowband(
    model,
    task: dict[str, Any],
    probe_data: NarrowbandSourceBatch,
    eval_data: NarrowbandSourceBatch,
    *,
    seed: int,
) -> dict[str, Any]:
    """Evaluate phase-varying independent narrowband causes."""

    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    window = int(task["window"])
    consumes = str(model.consumes)
    _, probe_native = encode_windows(model, probe_data.x, window=window)
    _, eval_native = encode_windows(model, eval_data.x, window=window)
    n_sources = int(eval_data.states.shape[2])
    probe_states = (
        probe_data.states[:, :window]
        .reshape(probe_data.states.shape[0], window, 2 * n_sources)
        .numpy()
    )
    eval_states = (
        eval_data.states[:, :window]
        .reshape(eval_data.states.shape[0], window, 2 * n_sources)
        .numpy()
    )
    ridge, _, per_quadrature = _fit_hmm_probe(
        probe_native,
        probe_states,
        eval_native,
        eval_states,
        consumes=consumes,
    )
    per_source = np.asarray(per_quadrature, dtype=np.float64).reshape(n_sources, 2).mean(axis=1)

    reconstructed = torch.from_numpy(reconstruct_windows(model, eval_data.x, window=window)).to(
        dtype=eval_data.emissions.dtype
    )
    direct_prediction = torch.einsum(
        "ntd,jcd->ntjc",
        reconstructed,
        eval_data.emissions,
    ).numpy()
    direct_target = eval_data.states[:, :window].numpy()
    direct_per_quadrature = np.asarray(
        [
            r2_score(
                direct_target[:, :, source, component].reshape(-1),
                direct_prediction[:, :, source, component].reshape(-1),
            )
            for source in range(n_sources)
            for component in range(2)
        ],
        dtype=np.float64,
    )
    direct_per_source = direct_per_quadrature.reshape(n_sources, 2).mean(axis=1)
    raw_projection = torch.einsum(
        "ntd,jcd->ntjc",
        eval_data.x[:, :window],
        eval_data.emissions,
    ).numpy()
    raw_projection_per_quadrature = np.asarray(
        [
            r2_score(
                direct_target[:, :, source, component].reshape(-1),
                raw_projection[:, :, source, component].reshape(-1),
            )
            for source in range(n_sources)
            for component in range(2)
        ],
        dtype=np.float64,
    )
    raw_projection_per_source = raw_projection_per_quadrature.reshape(n_sources, 2).mean(axis=1)

    shuffled_x = _row_shuffle(eval_data.x, window, seed + 4000)
    _, shuffled_native = encode_windows(model, shuffled_x, window=window)
    shuffled_prediction = _predict_hmm_probe(
        ridge,
        shuffled_native,
        consumes=consumes,
        output_shape=eval_states.shape,
    )
    shuffled_r2 = float(r2_score(eval_states.reshape(-1), shuffled_prediction.reshape(-1)))

    raw_probe_x = probe_data.x[:, :window].reshape(-1, int(task["d_in"])).numpy()
    raw_probe_y = probe_states.reshape(-1, 2 * n_sources)
    raw_eval_x = eval_data.x[:, :window].reshape(-1, int(task["d_in"])).numpy()
    raw_eval_y = eval_states.reshape(-1, 2 * n_sources)
    raw_ridge = Ridge(alpha=1.0).fit(raw_probe_x, raw_probe_y)
    raw_prediction = raw_ridge.predict(raw_eval_x)
    raw_per_quadrature = np.asarray(
        [
            r2_score(raw_eval_y[:, component], raw_prediction[:, component])
            for component in range(2 * n_sources)
        ],
        dtype=np.float64,
    )
    raw_per_source = raw_per_quadrature.reshape(n_sources, 2).mean(axis=1)

    metrics: dict[str, Any] = {
        "direct_latent_r2": float(direct_per_source.mean()),
        "direct_latent_r2_per_source": direct_per_source.tolist(),
        "direct_latent_r2_per_quadrature": direct_per_quadrature.tolist(),
        "latent_r2": float(per_source.mean()),
        "latent_r2_per_source": per_source.tolist(),
        "latent_r2_per_quadrature": list(map(float, per_quadrature)),
        "raw_token_r2": float(raw_per_source.mean()),
        "raw_token_r2_per_source": raw_per_source.tolist(),
        "raw_projection_r2": float(raw_projection_per_source.mean()),
        "raw_projection_r2_per_source": raw_projection_per_source.tolist(),
        "time_shuffled_latent_r2": shuffled_r2,
        "frequencies": [float(value) for value in eval_data.frequencies],
        **_reconstruction_metrics(model, eval_data.x, window=window),
    }
    usage = _spectral_usage(model, eval_native)
    if usage:
        metrics["spectral_usage"] = usage
        band_source_r2 = []
        for start, stop in model.band_of_features():
            _, _, values = _fit_hmm_probe(
                probe_native[:, start:stop],
                probe_states,
                eval_native[:, start:stop],
                eval_states,
                consumes="window",
            )
            band_source_r2.append(
                np.asarray(values, dtype=np.float64).reshape(n_sources, 2).mean(axis=1)
            )
        recovered = np.stack(band_source_r2, axis=1)

        states = eval_data.states[:, :window]
        basis = dct_basis(window).to(dtype=states.dtype)
        coefficients = torch.einsum("wt,ntjc->nwjc", basis, states)
        expected_frequency = coefficients.square().sum(dim=(0, 3)).T.numpy()
        expected_frequency /= expected_frequency.sum(axis=1, keepdims=True)
        expected_band = np.stack(
            [expected_frequency[:, list(map(int, band))].sum(axis=1) for band in model.bands],
            axis=1,
        )
        expected_band /= expected_band.sum(axis=1, keepdims=True)
        expected_winner = expected_band.argmax(axis=1)
        recovered_winner = recovered.argmax(axis=1)
        metrics["band_latent_r2_per_source"] = recovered.tolist()
        metrics["expected_dct_band_energy_per_source"] = expected_band.tolist()
        metrics["expected_band_per_source"] = expected_winner.astype(int).tolist()
        metrics["recovered_band_per_source"] = recovered_winner.astype(int).tolist()
        metrics["band_localization_accuracy"] = float((expected_winner == recovered_winner).mean())
    return metrics


def evaluate_cell(
    model,
    cell: dict[str, Any],
    splits: dict[
        str,
        ShamirClockBatch | FactorialHMMBatch | NarrowbandSourceBatch,
    ],
) -> dict[str, Any]:
    if cell["family"] == "shamir":
        return evaluate_shamir(
            model,
            cell["task_spec"],
            splits["probe"],
            splits["eval"],
            seed=int(cell["seed"]),
        )
    if cell["family"] == "factorial_hmm":
        return evaluate_hmm(
            model,
            cell["task_spec"],
            splits["probe"],
            splits["eval"],
            seed=int(cell["seed"]),
        )
    return evaluate_narrowband(
        model,
        cell["task_spec"],
        splits["probe"],
        splits["eval"],
        seed=int(cell["seed"]),
    )


def write_summary(
    config: dict[str, Any],
    results_dir: Path,
    *,
    plan: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    expected = enumerate_cells(config, smoke=smoke)
    expected_ids = {cell["cell_id"] for cell in expected}
    latest = latest_results(results_dir / "results.jsonl")
    observed = {cell_id: row for cell_id, row in latest.items() if bool(row.get("smoke")) == smoke}
    ok = [row for row in observed.values() if row.get("status") == "ok"]
    aggregates = []
    for task in config["tasks"]:
        for model in config["models"]:
            selected = [
                row for row in ok if row["task"] == task["name"] and row["model"] == model["name"]
            ]
            if not selected:
                continue
            values = [float(row["metrics"][task["primary_metric"]]) for row in selected]
            aggregates.append(
                {
                    "task": task["name"],
                    "family": task["family"],
                    "model": model["name"],
                    "metric": task["primary_metric"],
                    "n": len(values),
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "seed_values": {
                        str(row["seed"]): float(row["metrics"][task["primary_metric"]])
                        for row in selected
                    },
                    "mean_l0_per_window": statistics.fmean(
                        float(row["metrics"]["l0_per_window"]) for row in selected
                    ),
                    "mean_nmse": statistics.fmean(
                        float(row["metrics"]["nmse"]) for row in selected
                    ),
                }
            )
    summary = {
        "schema_version": 1,
        "run_name": config["run_name"],
        "generated_at": _utc_now(),
        "smoke": smoke,
        "complete": (
            expected_ids == set(observed)
            and len(ok) == len(expected_ids)
            and not any(row.get("status") == "failed" for row in observed.values())
        ),
        "expected_cells": len(expected_ids),
        "observed_cells": len(observed),
        "ok_cells": len(ok),
        "missing_cell_ids": sorted(expected_ids - set(observed)),
        "unexpected_cell_ids": sorted(set(observed) - expected_ids),
        "failed": [
            {
                "cell_id": row["cell_id"],
                "task": row["task"],
                "model": row["model"],
                "seed": row["seed"],
                "error": row.get("error"),
            }
            for row in observed.values()
            if row.get("status") == "failed"
        ],
        "aggregates": aggregates,
        "plan": plan,
        "fairness": config["fairness"],
    }
    _atomic_json(results_dir / "summary.json", summary)
    return summary


def seed_compatible_results(
    config: dict[str, Any],
    source_dir: Path,
    destination_dir: Path,
    *,
    smoke: bool,
) -> int:
    """Copy completed cells that are still present after a plan reduction."""

    source_path = source_dir / "results.jsonl"
    if not source_path.exists():
        return 0
    expected_ids = {cell["cell_id"] for cell in enumerate_cells(config, smoke=smoke)}
    existing = latest_results(destination_dir / "results.jsonl")
    copied = 0
    for cell_id, row in latest_results(source_path).items():
        if (
            cell_id in expected_ids
            and cell_id not in existing
            and row.get("status") == "ok"
            and bool(row.get("smoke")) == smoke
        ):
            _append_jsonl(destination_dir / "results.jsonl", row)
            copied += 1
    return copied


def run(
    config: dict[str, Any],
    results_dir: Path,
    *,
    smoke: bool,
    seed_results_dir: Path | None = None,
) -> int:
    plan = build_plan(config, smoke=smoke)
    if not plan["within_cost_plan"] or not plan["within_time_plan"]:
        raise RuntimeError(f"refusing out-of-plan run: {plan}")
    results_dir.mkdir(parents=True, exist_ok=True)
    if seed_results_dir is not None:
        copied = seed_compatible_results(
            config,
            seed_results_dir,
            results_dir,
            smoke=smoke,
        )
        print(f"[seed] copied {copied} compatible completed cells", flush=True)
    _atomic_json(results_dir / "frozen_config.json", config)
    _atomic_json(results_dir / "plan.json", plan)
    config_hash = hashlib.sha256(_canonical_json(config).encode()).hexdigest()
    budget = BudgetGuard(config, results_dir, config_hash)
    status = "complete"
    data_cache: dict[
        str,
        dict[
            str,
            ShamirClockBatch | FactorialHMMBatch | NarrowbandSourceBatch,
        ],
    ] = {}
    try:
        completed = latest_results(results_dir / "results.jsonl")
        for cell in enumerate_cells(config, smoke=smoke):
            prior = completed.get(cell["cell_id"])
            if prior and prior.get("status") == "ok":
                print(f"[resume] {cell['cell_id']}", flush=True)
                continue
            group = str(cell["group"])
            if group not in data_cache:
                data_cache[group] = _materialize_group(config, cell["task_spec"])
            model = None
            started = time.monotonic()
            print(
                f"[cell] {cell['model']}/{cell['task']}/s{cell['seed']} "
                f"steps={cell['n_steps']} id={cell['cell_id']}",
                flush=True,
            )
            try:
                model, training = train_or_load(
                    config,
                    cell,
                    data_cache[group]["train"],
                    results_dir,
                    budget,
                )
                metrics = evaluate_cell(model, cell, data_cache[group])
                primary = str(cell["primary_metric"])
                row = {
                    "schema_version": 1,
                    **{
                        key: cell[key]
                        for key in (
                            "cell_id",
                            "training_id",
                            "task",
                            "group",
                            "family",
                            "model",
                            "seed",
                            "window",
                            "d_in",
                            "d_sae",
                            "k_pos",
                            "n_steps",
                            "primary_metric",
                            "smoke",
                        )
                    },
                    "status": "ok",
                    "metrics": metrics,
                    "primary_value": float(metrics[primary]),
                    "training": training,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "estimated_incremental_spend_usd": budget.update(),
                    "completed_at": _utc_now(),
                }
            except StopRequested:
                raise
            except Exception as exc:
                row = {
                    "schema_version": 1,
                    "cell_id": cell["cell_id"],
                    "training_id": cell["training_id"],
                    "task": cell["task"],
                    "group": cell["group"],
                    "family": cell["family"],
                    "model": cell["model"],
                    "seed": cell["seed"],
                    "window": cell["window"],
                    "smoke": smoke,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-5000:],
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "completed_at": _utc_now(),
                }
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _append_jsonl(results_dir / "results.jsonl", row)
            completed[cell["cell_id"]] = row
            print(
                f"  -> {row['status']} {row.get('primary_value', row.get('error', ''))}",
                flush=True,
            )
            if row["status"] != "ok":
                raise RuntimeError(f"cell {cell['cell_id']} failed: {row.get('error')}")
        write_summary(config, results_dir, plan=plan, smoke=smoke)
        return 0
    except StopRequested as exc:
        status = "deadline"
        print(f"[stop] {exc}", flush=True)
        write_summary(config, results_dir, plan=plan, smoke=smoke)
        return 0
    except Exception:
        status = "failed"
        write_summary(config, results_dir, plan=plan, smoke=smoke)
        raise
    finally:
        budget.finish(status)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed-results-dir", type=Path)
    parser.add_argument("--mode", choices=("plan", "smoke", "full"), default="plan")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    if args.mode == "plan":
        print(json.dumps(build_plan(config), indent=2, sort_keys=True))
        return 0
    return run(
        config,
        args.results_dir,
        smoke=args.mode == "smoke",
        seed_results_dir=args.seed_results_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
