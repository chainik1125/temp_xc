"""Train the Step-7 sparse representations for 10k steps and audit health."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from experiments.power_spectrum.persona_drift_txc.protocol import (
    EXPERIMENT_ROOT,
    config_digest,
    file_sha256,
    iter_jsonl,
    load_config,
    write_json,
)
from temp_bench.core.config import import_by_path


@dataclass(frozen=True)
class Normalization:
    mean: torch.Tensor
    scalar_rms: float

    def apply(self, activations: torch.Tensor) -> torch.Tensor:
        return (activations.float() - self.mean) / self.scalar_rms


def _load_data(
    activation_path: Path,
    metadata_path: Path,
) -> tuple[torch.Tensor, list[dict[str, Any]], Normalization]:
    payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    activations = payload["activations"]
    metadata = list(iter_jsonl(metadata_path))
    if list(payload["conversation_ids"]) != [record["conversation_id"] for record in metadata]:
        raise ValueError("activation and metadata conversation ordering differs")
    train_indices = [index for index, record in enumerate(metadata) if record["split"] == "train"]
    train = activations[train_indices].float()
    mean = train.mean(dim=(0, 1))
    centered = train - mean
    scalar_rms = float(centered.square().mean().sqrt())
    if not math.isfinite(scalar_rms) or scalar_rms <= 0:
        raise ValueError(f"invalid activation RMS: {scalar_rms}")
    return activations, metadata, Normalization(mean=mean, scalar_rms=scalar_rms)


def _architecture_spec(config: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        item for item in config["representation_training"]["architectures"] if item["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown/ambiguous architecture {name!r}")
    return matches[0]


def _instantiate(
    *,
    spec: dict[str, Any],
    d_in: int,
    d_sae: int,
    k_pos: int,
    dead_feature_threshold_tokens: int,
) -> torch.nn.Module:
    cls = import_by_path(spec["class_path"])
    window = int(spec["window"])
    kwargs: dict[str, Any] = {
        "d_in": d_in,
        "d_sae": d_sae,
        "k_pos": k_pos,
        "T": window,
    }
    if spec["name"] == "tsae":
        kwargs.update(
            h_frac=0.2,
            contrastive_alpha=float(spec["contrastive_alpha"]),
            auxk_alpha=1.0 / 32.0,
            threshold_start_step=1000,
            dead_feature_threshold_tokens=dead_feature_threshold_tokens,
        )
    elif spec["name"] == "sae":
        kwargs.update(
            auxk_alpha=1.0 / 32.0,
            threshold_start_step=1000,
            dead_feature_threshold_tokens=dead_feature_threshold_tokens,
        )
    else:
        kwargs.update(
            auxk_alpha=1.0 / 32.0,
            threshold_start_step=1000,
            dead_threshold_tokens=dead_feature_threshold_tokens,
        )
    return cls(**kwargs)


def _all_windows(sequences: torch.Tensor, window: int) -> torch.Tensor:
    if sequences.ndim != 3:
        raise ValueError("sequences must have shape (conversation, turn, d_in)")
    pieces = [
        sequences[:, start : start + window] for start in range(sequences.shape[1] - window + 1)
    ]
    return torch.stack(pieces, dim=1)


def _sample_batch(
    *,
    name: str,
    train_sequences: torch.Tensor,
    train_windows: torch.Tensor | None,
    positions_per_step: int,
    window: int,
    generator: torch.Generator,
) -> torch.Tensor:
    n_conversations, n_turns, _d_in = train_sequences.shape
    if name == "sae":
        conversation_index = torch.randint(
            n_conversations, (positions_per_step,), generator=generator
        )
        turn_index = torch.randint(n_turns, (positions_per_step,), generator=generator)
        return train_sequences[conversation_index, turn_index]
    if name == "tsae":
        conversation_index = torch.randint(
            n_conversations, (positions_per_step,), generator=generator
        )
        offset = torch.randint(n_turns - 1, (positions_per_step,), generator=generator)
        rows = torch.arange(positions_per_step)
        pairs = torch.stack(
            (
                train_sequences[conversation_index, offset],
                train_sequences[conversation_index, offset + 1],
            ),
            dim=1,
        )
        if pairs.shape[0] != len(rows):  # pragma: no cover - defensive
            raise AssertionError("pair sampling failed")
        return pairs
    if train_windows is None:
        raise ValueError("TXC requires precomputed windows")
    batch_size = max(1, positions_per_step // window)
    flattened = train_windows.reshape(-1, window, train_windows.shape[-1])
    indices = torch.randint(len(flattened), (batch_size,), generator=generator)
    return flattened[indices]


def _scalar_metrics(result: Any, d_in: int) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(result, tuple):
        loss, info = result
        metrics = {"loss": loss, **info}
    else:
        metrics = result
        loss = metrics["loss"]
    scalar: dict[str, float] = {}
    for key, value in metrics.items():
        if key == "z":
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            scalar[key] = float(value.detach())
        elif isinstance(value, (float, int)):
            scalar[key] = float(value)
    if "mse" in scalar:
        scalar["mse_per_dimension"] = scalar["mse"] / d_in
    return loss, scalar


def _plateau_summary(rows: list[dict[str, float]], steps: int) -> dict[str, Any]:
    mse = np.asarray(
        [row["fixed_validation_nmse"] for row in rows],
        dtype=np.float64,
    )
    step = np.asarray([row["step"] for row in rows], dtype=np.float64)
    tail_start = max(0, int(0.8 * len(rows)))
    tail_step = step[tail_start:]
    tail_mse = mse[tail_start:]
    slope = (
        float(np.polyfit(tail_step, np.log(np.maximum(tail_mse, 1e-12)), 1)[0])
        if len(tail_step) >= 2
        else float("nan")
    )
    quarter = max(1, len(tail_mse) // 4)
    beginning = float(tail_mse[:quarter].mean())
    ending = float(tail_mse[-quarter:].mean())
    relative_improvement = (beginning - ending) / max(beginning, 1e-12)
    return {
        "steps": steps,
        "final_fixed_validation_nmse": float(mse[-1]),
        "tail_log_mse_slope_per_step": slope,
        "tail_relative_improvement": relative_improvement,
        "plateaued_by_one_percent_rule": relative_improvement < 0.01,
    }


@torch.inference_mode()
def _fixed_validation_nmse(
    *,
    model: torch.nn.Module,
    name: str,
    validation_sequences: torch.Tensor,
    validation_windows: torch.Tensor | None,
    window: int,
    positions: int = 512,
) -> float:
    was_training = model.training
    model.eval()
    if name in {"sae", "tsae"}:
        batch = validation_sequences.reshape(-1, validation_sequences.shape[-1])[:positions]
    else:
        if validation_windows is None:
            raise ValueError("TXC validation requires windows")
        number_of_windows = max(1, positions // window)
        batch = validation_windows.reshape(-1, window, validation_windows.shape[-1])[
            :number_of_windows
        ]
    batch = batch.to(next(model.parameters()).device)
    reconstruction = model.decode(model.encode(batch))
    result = float((batch - reconstruction).square().sum() / batch.square().sum().clamp_min(1e-12))
    model.train(was_training)
    return result


@torch.inference_mode()
def _encode_decode_audit(
    *,
    model: torch.nn.Module,
    name: str,
    normalized: torch.Tensor,
    metadata: list[dict[str, Any]],
    window: int,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    split_names = ("train", "validation", "test")
    if name in {"sae", "tsae"}:
        shaped_input = normalized
        flat_input = normalized.reshape(-1, normalized.shape[-1])
        rows_per_conversation = normalized.shape[1]
        code_batches: list[torch.Tensor] = []
        squared_error = 0.0
        denominator = 0.0
        fire_counts = torch.zeros(model._d_sae, dtype=torch.long)  # noqa: SLF001
        for start in range(0, len(flat_input), batch_size):
            batch = flat_input[start : start + batch_size].to(next(model.parameters()).device)
            code = model.encode(batch)
            reconstruction = model.decode(code)
            squared_error += float((batch - reconstruction).square().sum())
            denominator += float(batch.square().sum())
            fire_counts += (code != 0).sum(dim=0).cpu()
            code_batches.append(code.to(torch.bfloat16).cpu())
        codes = torch.cat(code_batches).reshape(shaped_input.shape[0], shaped_input.shape[1], -1)
        n_rows = flat_input.shape[0]
    else:
        windows = _all_windows(normalized, window)
        flat_input = windows.reshape(-1, window, windows.shape[-1])
        rows_per_conversation = windows.shape[1]
        code_batches = []
        squared_error = 0.0
        denominator = 0.0
        fire_counts = torch.zeros(model._d_sae, dtype=torch.long)  # noqa: SLF001
        for start in range(0, len(flat_input), batch_size):
            batch = flat_input[start : start + batch_size].to(next(model.parameters()).device)
            code = model.encode(batch).squeeze(1)
            reconstruction = model.decode(code)
            squared_error += float((batch - reconstruction).square().sum())
            denominator += float(batch.square().sum())
            fire_counts += (code != 0).sum(dim=0).cpu()
            code_batches.append(code.to(torch.bfloat16).cpu())
        codes = torch.cat(code_batches).reshape(windows.shape[0], windows.shape[1], -1)
        n_rows = flat_input.shape[0]

    split_by_row = np.repeat(
        np.asarray([record["split"] for record in metadata]),
        rows_per_conversation,
    )
    squared_error_by_split = {split: 0.0 for split in split_names}
    denominator_by_split = {split: 0.0 for split in split_names}
    fire_counts_by_split: dict[str, torch.Tensor] = {}
    for split in split_names:
        conversation_mask = torch.tensor(
            [record["split"] == split for record in metadata],
            dtype=torch.bool,
        )
        split_input = flat_input[torch.from_numpy(split_by_row == split)]
        split_codes = codes[conversation_mask].reshape(-1, codes.shape[-1]).float()
        fire_counts_by_split[split] = (split_codes != 0).sum(dim=0)
        for start in range(0, len(split_input), batch_size):
            batch = split_input[start : start + batch_size].to(next(model.parameters()).device)
            code = model.encode(batch)
            reconstruction = model.decode(code)
            squared_error_by_split[split] += float((batch - reconstruction).square().sum())
            denominator_by_split[split] += float(batch.square().sum())

    l0_per_row = float(fire_counts.sum()) / max(n_rows, 1)
    nonzero_rates = fire_counts.float() / max(n_rows, 1)
    built_in = getattr(model, "num_tokens_since_fired", None)
    since_fired = built_in.detach().cpu() if built_in is not None else None
    health = {
        "reconstruction_nmse": squared_error / max(denominator, 1e-12),
        "realized_l0": l0_per_row,
        "n_eval_rows": int(n_rows),
        "dead_never_fired_on_corpus": int((fire_counts == 0).sum()),
        "dead_fraction_never_fired_on_corpus": float((fire_counts == 0).float().mean()),
        "fired_fewer_than_5_eval_rows": int((fire_counts < 5).sum()),
        "fired_fewer_than_10_eval_rows": int((fire_counts < 10).sum()),
        "median_firing_rate": float(nonzero_rates.median()),
        "maximum_firing_rate": float(nonzero_rates.max()),
        "reconstruction_nmse_by_split": {
            split: squared_error_by_split[split] / max(denominator_by_split[split], 1e-12)
            for split in split_names
        },
    }
    for split in split_names:
        split_counts = fire_counts_by_split[split]
        health[f"dead_never_fired_on_{split}"] = int((split_counts == 0).sum())
        health[f"dead_fraction_never_fired_on_{split}"] = float((split_counts == 0).float().mean())
    if since_fired is not None:
        health["training_tokens_since_fired_ge_100k"] = int((since_fired >= 100_000).sum())
        health["training_tokens_since_fired_ge_1m"] = int((since_fired >= 1_000_000).sum())
        health["training_tokens_since_fired_ge_10m"] = int((since_fired >= 10_000_000).sum())
    return health, {
        "codes": codes,
        "fire_counts": fire_counts,
    }


def train_one(
    *,
    architecture: str,
    activation_path: Path,
    metadata_path: Path,
    output_root: Path,
    force: bool,
    steps_override: int | None = None,
    d_sae_override: int | None = None,
    positions_per_step_override: int | None = None,
) -> None:
    config = load_config()
    training = config["representation_training"]
    spec = _architecture_spec(config, architecture)
    architecture_root = output_root / architecture
    final_checkpoint = architecture_root / "model.safetensors"
    final_health = architecture_root / "health.json"
    final_codes = architecture_root / "codes.pt"
    if not force and final_checkpoint.exists() and final_health.exists() and final_codes.exists():
        with final_health.open() as handle:
            existing_health = json.load(handle)
        expected_identity = {
            "config_sha256": config_digest(config),
            "activation_sha256": file_sha256(activation_path),
            "metadata_sha256": file_sha256(metadata_path),
        }
        actual_identity = {key: existing_health.get(key) for key in expected_identity}
        if actual_identity != expected_identity:
            raise RuntimeError(
                f"{architecture}: existing artifacts do not match inputs: "
                f"expected={expected_identity}, actual={actual_identity}"
            )
        print(f"[train] {architecture}: complete artifacts already exist; skipping")
        return

    activations, metadata, normalization = _load_data(activation_path, metadata_path)
    train_indices = [index for index, record in enumerate(metadata) if record["split"] == "train"]
    validation_indices = [
        index for index, record in enumerate(metadata) if record["split"] == "validation"
    ]
    normalized = normalization.apply(activations)
    train_sequences = normalized[train_indices].contiguous()
    validation_sequences = normalized[validation_indices].contiguous()
    window = int(spec["window"])
    train_windows = _all_windows(train_sequences, window) if window > 1 else None
    validation_windows = _all_windows(validation_sequences, window) if window > 1 else None

    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_sae = int(d_sae_override or training["d_sae"])
    model = _instantiate(
        spec=spec,
        d_in=activations.shape[-1],
        d_sae=d_sae,
        k_pos=int(training["k_pos"]),
        dead_feature_threshold_tokens=int(training["dead_feature_threshold_tokens"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    generator = torch.Generator(device="cpu").manual_seed(seed + 991)
    steps = int(steps_override or training["steps"])
    warmup = int(training["warmup_steps"])
    log_every = int(training["log_every"])
    checkpoint_every = int(training["checkpoint_every"])
    positions_per_step = int(positions_per_step_override or training["positions_per_step"])
    rows: list[dict[str, float]] = []
    resume_checkpoint = architecture_root / "resume_checkpoint.pt"
    start_step = 0
    if resume_checkpoint.exists() and not force:
        resume = torch.load(
            resume_checkpoint,
            map_location=device,
            weights_only=False,
        )
        expected = {
            "architecture": architecture,
            "config_sha256": config_digest(config),
            "activation_sha256": file_sha256(activation_path),
            "metadata_sha256": file_sha256(metadata_path),
            "d_sae": d_sae,
            "positions_per_step": positions_per_step,
            "steps": steps,
        }
        actual = {key: resume.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"{architecture}: incompatible resume checkpoint: "
                f"expected={expected}, actual={actual}"
            )
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["sample_generator_state"])
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        if torch.cuda.is_available() and resume.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(resume["cuda_rng_state"].cpu(), device=device)
        rows = resume["rows"]
        start_step = int(resume["completed_steps"])
        print(
            f"[train] {architecture}: resuming at step {start_step}/{steps}",
            flush=True,
        )
    model.train()

    architecture_root.mkdir(parents=True, exist_ok=True)
    for step in range(start_step, steps):
        batch = _sample_batch(
            name=architecture,
            train_sequences=train_sequences,
            train_windows=train_windows,
            positions_per_step=positions_per_step,
            window=window,
            generator=generator,
        ).to(device)
        learning_rate = float(training["learning_rate"]) * min(1.0, (step + 1) / max(warmup, 1))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        result = model.train_step(batch)
        loss, scalar = _scalar_metrics(result, activations.shape[-1])
        if architecture == "tsae":
            latent = result[1]["z"]
            reconstruction = model.decode(latent)
            reconstruction_mse = (batch[:, 0] - reconstruction).square().sum(dim=-1).mean()
            scalar["reconstruction_mse"] = float(reconstruction_mse.detach())
            scalar["reconstruction_mse_per_dimension"] = (
                scalar["reconstruction_mse"] / activations.shape[-1]
            )
        else:
            scalar["reconstruction_mse"] = scalar["mse"]
            scalar["reconstruction_mse_per_dimension"] = scalar["mse_per_dimension"]
        loss.backward()
        optimizer.step()
        model.post_step()

        if step % log_every == 0 or step == steps - 1:
            row = {
                "step": float(step + 1),
                "learning_rate": learning_rate,
                "fixed_validation_nmse": _fixed_validation_nmse(
                    model=model,
                    name=architecture,
                    validation_sequences=validation_sequences,
                    validation_windows=validation_windows,
                    window=window,
                ),
                **scalar,
            }
            rows.append(row)
            print(
                f"[train] {architecture} step={step + 1:>5}/{steps} "
                f"mse/d={row['reconstruction_mse_per_dimension']:.5f} "
                f"val_nmse={row['fixed_validation_nmse']:.5f} "
                f"l0={row.get('l0', float('nan')):.2f} "
                f"dead_builtin={row.get('dead', float('nan')):.0f}",
                flush=True,
            )

        if (step + 1) % checkpoint_every == 0 and step + 1 < steps:
            temporary_resume = resume_checkpoint.with_suffix(".pt.tmp")
            torch.save(
                {
                    "architecture": architecture,
                    "config_sha256": config_digest(config),
                    "activation_sha256": file_sha256(activation_path),
                    "metadata_sha256": file_sha256(metadata_path),
                    "d_sae": d_sae,
                    "positions_per_step": positions_per_step,
                    "steps": steps,
                    "completed_steps": step + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sample_generator_state": generator.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
                    ),
                    "rows": rows,
                },
                temporary_resume,
            )
            os.replace(temporary_resume, resume_checkpoint)
            print(
                f"[train] {architecture}: checkpointed step {step + 1}",
                flush=True,
            )

    temporary_checkpoint = final_checkpoint.with_suffix(".safetensors.tmp")
    save_file(
        {key: value.detach().contiguous().cpu() for key, value in model.state_dict().items()},
        str(temporary_checkpoint),
    )
    os.replace(temporary_checkpoint, final_checkpoint)
    temporary_normalization = architecture_root / "normalization.pt.tmp"
    torch.save(
        {"mean": normalization.mean, "scalar_rms": normalization.scalar_rms},
        temporary_normalization,
    )
    os.replace(temporary_normalization, architecture_root / "normalization.pt")
    fieldnames = sorted({key for row in rows for key in row})
    temporary_log = architecture_root / "training_log.csv.tmp"
    with temporary_log.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_log, architecture_root / "training_log.csv")

    audit, code_payload = _encode_decode_audit(
        model=model,
        name=architecture,
        normalized=normalized,
        metadata=metadata,
        window=window,
        batch_size=256 if window == 1 else max(8, 256 // window),
    )
    temporary_codes = final_codes.with_suffix(".pt.tmp")
    torch.save(
        {
            **code_payload,
            "conversation_ids": [record["conversation_id"] for record in metadata],
            "window": window,
        },
        temporary_codes,
    )
    os.replace(temporary_codes, final_codes)
    plateau = _plateau_summary(rows, steps)
    health = {
        "architecture": architecture,
        "class_path": spec["class_path"],
        "window": window,
        "d_in": int(activations.shape[-1]),
        "d_sae": d_sae,
        "k_pos": int(training["k_pos"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seed": seed,
        "dead_feature_threshold_tokens": int(training["dead_feature_threshold_tokens"]),
        "realized_l0_unit": "token" if window == 1 else "shared_window_code",
        "normalization_scalar_rms": normalization.scalar_rms,
        "config_sha256": config_digest(config),
        "activation_sha256": file_sha256(activation_path),
        "metadata_sha256": file_sha256(metadata_path),
        **plateau,
        **audit,
    }
    temporary_health = final_health.with_suffix(".json.tmp")
    write_json(temporary_health, health)
    os.replace(temporary_health, final_health)
    resume_checkpoint.unlink(missing_ok=True)
    print(f"[train] {architecture} health={json_compact(health)}", flush=True)


def json_compact(payload: dict[str, Any]) -> str:
    fields = (
        "final_fixed_validation_nmse",
        "tail_relative_improvement",
        "plateaued_by_one_percent_rule",
        "reconstruction_nmse",
        "realized_l0",
        "dead_never_fired_on_corpus",
    )
    return " ".join(f"{key}={payload[key]}" for key in fields)


def load_trained_model(
    *,
    architecture: str,
    activation_path: Path,
    output_root: Path,
) -> torch.nn.Module:
    config = load_config()
    training = config["representation_training"]
    spec = _architecture_spec(config, architecture)
    payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    model = _instantiate(
        spec=spec,
        d_in=payload["activations"].shape[-1],
        d_sae=int(training["d_sae"]),
        k_pos=int(training["k_pos"]),
        dead_feature_threshold_tokens=int(training["dead_feature_threshold_tokens"]),
    )
    state = load_file(str(output_root / architecture / "model.safetensors"))
    model.load_state_dict(state)
    return model


def plot_training_diagnostics(output_root: Path) -> None:
    config = load_config()
    architectures = [item["name"] for item in config["representation_training"]["architectures"]]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    colors = {
        "sae": "#0072B2",
        "tsae": "#009E73",
        "txc_w4": "#E69F00",
        "txc_w8": "#D55E00",
    }
    health_rows: list[dict[str, Any]] = []
    for architecture in architectures:
        log_path = output_root / architecture / "training_log.csv"
        health_path = output_root / architecture / "health.json"
        if not log_path.exists() or not health_path.exists():
            continue
        with log_path.open() as handle:
            rows = list(csv.DictReader(handle))
        steps = np.asarray([float(row["step"]) for row in rows])
        mse = np.asarray([float(row["fixed_validation_nmse"]) for row in rows])
        axes[0].plot(steps, mse, label=architecture, color=colors.get(architecture))
        with health_path.open() as handle:
            health = json.load(handle)
        health_rows.append(
            {
                "architecture": architecture,
                "final_fixed_validation_nmse": health["final_fixed_validation_nmse"],
                "tail_relative_improvement": health["tail_relative_improvement"],
                "plateaued_by_one_percent_rule": health["plateaued_by_one_percent_rule"],
                "heldout_test_nmse": health["reconstruction_nmse_by_split"]["test"],
                "realized_l0": health["realized_l0"],
                "realized_l0_unit": health["realized_l0_unit"],
                "dead_all": health["dead_never_fired_on_corpus"],
                "dead_train": health["dead_never_fired_on_train"],
                "dead_validation": health["dead_never_fired_on_validation"],
                "dead_test": health["dead_never_fired_on_test"],
                "d_sae": health["d_sae"],
            }
        )
        axes[1].bar(
            architecture,
            health["dead_never_fired_on_train"],
            color=colors.get(architecture),
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Fixed validation reconstruction NMSE")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].set_ylabel("Latents never firing on train split")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output_root.mkdir(parents=True, exist_ok=True)
    if health_rows:
        with (output_root / "health_summary.csv").open(
            "w",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(health_rows[0]))
            writer.writeheader()
            writer.writerows(health_rows)
    figure.savefig(output_root / "training_diagnostics.png", dpi=220)
    figure.savefig(output_root / "training_diagnostics.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activations",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts" / "activations" / "turn_activations.pt",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts" / "activations" / "metadata.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts" / "representations",
    )
    parser.add_argument(
        "--architecture",
        action="append",
        choices=("sae", "tsae", "txc_w4", "txc_w8"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--d-sae", type=int)
    parser.add_argument("--positions-per-step", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.plot_only:
        architectures = args.architecture or ["sae", "tsae", "txc_w4", "txc_w8"]
        for architecture in architectures:
            train_one(
                architecture=architecture,
                activation_path=args.activations,
                metadata_path=args.metadata,
                output_root=args.output_root,
                force=args.force,
                steps_override=args.steps,
                d_sae_override=args.d_sae,
                positions_per_step_override=args.positions_per_step,
            )
    plot_training_diagnostics(args.output_root)


if __name__ == "__main__":
    main()
