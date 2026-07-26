"""Matched, resumable TopK-SAE/TXC training on the Ward activation cache."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from temp_bench.archs.topk_sae import TopKSAE
from temp_bench.archs.txc_base import TXCBase


@dataclass(frozen=True)
class TrainCellConfig:
    arch: str
    window: int
    seed: int
    d_in: int
    d_sae: int
    k_pos: int
    batch_size: int
    steps: int
    learning_rate: float
    warmup_steps: int
    checkpoint_every: int
    schedule_seed: int
    amp: bool
    schedule_max_window: int | None = None
    record_effective_l0: bool = False


def build_model(config: TrainCellConfig) -> torch.nn.Module:
    if config.arch == "txc":
        return TXCBase(
            d_in=config.d_in,
            d_sae=config.d_sae,
            T=config.window,
            k_pos=config.k_pos,
        )
    if config.arch == "sae":
        return TopKSAE(
            d_in=config.d_in,
            d_sae=config.d_sae,
            k_pos=config.k_pos,
        )
    raise ValueError(f"arch must be txc or sae, got {config.arch!r}")


def scheduled_window_indices(
    *,
    step: int,
    n_sequences: int,
    sequence_length: int,
    window: int,
    batch_size: int,
    schedule_seed: int,
    max_window: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Nested counter schedule shared by architectures and every T.

    Sequence IDs and a length-``max_window`` start are sampled without using
    ``window`` in the RNG. A shorter T receives the trailing subset ending at
    the same token as the max-window sample.
    """

    if not 1 <= window <= max_window <= sequence_length:
        raise ValueError(f"invalid window {window} for sequence length {sequence_length}")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(schedule_seed), int(step)])
    )
    sequence_indices = rng.integers(
        0, n_sequences, size=batch_size, dtype=np.int64
    )
    starts = rng.integers(
        0, sequence_length - max_window + 1, size=batch_size, dtype=np.int64
    )
    return sequence_indices, starts + (max_window - window)


def materialize_windows(
    cache: np.ndarray,
    *,
    step: int,
    window: int,
    batch_size: int,
    schedule_seed: int,
    max_window: int = 6,
) -> np.ndarray:
    sequence_indices, starts = scheduled_window_indices(
        step=step,
        n_sequences=int(cache.shape[0]),
        sequence_length=int(cache.shape[1]),
        window=window,
        batch_size=batch_size,
        schedule_seed=schedule_seed,
        max_window=max_window,
    )
    positions = starts[:, None] + np.arange(window, dtype=np.int64)[None, :]
    return np.asarray(
        cache[sequence_indices[:, None], positions],
        dtype=np.float32,
        order="C",
    )


def _atomic_safetensors(model: torch.nn.Module, path: Path) -> None:
    from safetensors.torch import save_file

    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        name: tensor.detach().contiguous().cpu()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(temporary))
    os.replace(temporary, path)


def _atomic_torch(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _model_dtype(config: TrainCellConfig, device: str) -> torch.dtype:
    return (
        torch.bfloat16
        if config.amp and device.startswith("cuda")
        else torch.float32
    )


def _load_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_path: Path,
    state_path: Path,
    *,
    device: str,
) -> tuple[int, dict[str, float]]:
    from safetensors.torch import load_file

    if model_path.exists() != state_path.exists():
        raise ValueError(
            f"partial checkpoint: {model_path} and {state_path} must coexist"
        )
    if not model_path.exists():
        return 0, {}
    model.load_state_dict(load_file(str(model_path), device=device))
    payload = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["step"]), dict(payload.get("last_metrics", {}))


def train_dictionary(
    *,
    activation_cache: Path,
    checkpoint_dir: Path,
    config: TrainCellConfig,
    device: str,
) -> dict:
    """Train or resume one cell, persisting model and optimizer atomically."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.safetensors"
    state_path = checkpoint_dir / "training_state.pt"
    requested = asdict(config)
    # Keep the frozen T<=6 checkpoint contract byte-for-byte compatible.
    # The new fields are serialized only when the isolated T16 protocol uses
    # them, so existing v2026-07-23.2 checkpoints still resume unchanged.
    if requested["schedule_max_window"] is None:
        requested.pop("schedule_max_window")
    if not requested["record_effective_l0"]:
        requested.pop("record_effective_l0")
    requested["exposure_contract"] = (
        "SAE and TXC receive the identical BxT raw windows at every step; "
        "SAE flattens those values to B*T tokens"
    )
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != requested:
            raise ValueError(
                f"checkpoint config mismatch at {config_path}; refuse to mix cells"
            )
    else:
        _atomic_json(requested, config_path)

    cache = np.load(activation_cache, mmap_mode="r")
    if cache.ndim != 3 or cache.shape[-1] != config.d_in:
        raise ValueError(
            f"activation cache shape {cache.shape} is incompatible with d_in={config.d_in}"
        )
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    model = build_model(config).to(
        device=device, dtype=_model_dtype(config, device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_step, last_metrics = _load_state(
        model, optimizer, model_path, state_path, device=device
    )
    if start_step > config.steps:
        raise ValueError(
            f"checkpoint step {start_step} exceeds requested target {config.steps}"
        )
    if start_step == config.steps:
        result = {
            "arch": config.arch,
            "window": config.window,
            "seed": config.seed,
            "completed_steps": start_step,
            "checkpoint": str(model_path),
            "cached": True,
            "last_metrics": last_metrics,
        }
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    model.train()
    for step in range(start_step, config.steps):
        raw = materialize_windows(
            cache,
            step=step,
            window=config.window,
            batch_size=config.batch_size,
            schedule_seed=config.schedule_seed,
            max_window=config.schedule_max_window or 6,
        )
        batch = torch.from_numpy(raw).to(
            device=device, dtype=_model_dtype(config, device)
        )
        if config.warmup_steps > 0:
            scale = min(1.0, float(step + 1) / config.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * scale
        model.pre_step()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
            enabled=config.amp and device.startswith("cuda"),
        ):
            result = (
                model.train_step(batch)
                if config.arch == "txc"
                else model.train_step(batch.reshape(-1, config.d_in))
            )
            loss = result["loss"]
        loss.backward()
        optimizer.step()
        model.post_step()
        last_metrics = {
            name: float(value.detach().float().cpu())
            for name, value in result.items()
            if value.numel() == 1
        }
        if config.record_effective_l0:
            nominal_l0 = (
                int(model.k_win)
                if config.arch == "txc"
                else int(model.k)
            )
            effective_l0 = float(last_metrics["l0"])
            last_metrics.update(
                {
                    "nominal_l0": float(nominal_l0),
                    "effective_l0": effective_l0,
                    "effective_l0_fill_fraction": (
                        effective_l0 / nominal_l0
                        if nominal_l0 > 0
                        else 0.0
                    ),
                    "effective_l0_underfill": float(
                        nominal_l0 - effective_l0
                    ),
                }
            )
        completed = step + 1
        should_save = (
            completed == config.steps
            or completed % config.checkpoint_every == 0
        )
        if should_save:
            _atomic_safetensors(model, model_path)
            _atomic_torch(
                {
                    "step": completed,
                    "optimizer": optimizer.state_dict(),
                    "last_metrics": last_metrics,
                },
                state_path,
            )
            print(
                f"[train] arch={config.arch} T={config.window} "
                f"seed={config.seed} step={completed}/{config.steps} "
                f"loss={last_metrics.get('loss', float('nan')):.6g}",
                flush=True,
            )
    result = {
        "arch": config.arch,
        "window": config.window,
        "seed": config.seed,
        "completed_steps": config.steps,
        "checkpoint": str(model_path),
        "cached": False,
        "last_metrics": last_metrics,
    }
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def load_dictionary(
    checkpoint_dir: Path, *, device: str
) -> tuple[torch.nn.Module, TrainCellConfig]:
    from safetensors.torch import load_file

    raw = json.loads((checkpoint_dir / "config.json").read_text())
    raw.pop("exposure_contract", None)
    config = TrainCellConfig(**raw)
    model = build_model(config).to(
        device=device, dtype=_model_dtype(config, device)
    )
    model.load_state_dict(
        load_file(str(checkpoint_dir / "model.safetensors"), device=device)
    )
    model.eval()
    return model, config


def run_memory_smoke(
    *,
    activation_cache: Path,
    config: TrainCellConfig,
    device: str,
) -> dict:
    """Run one real-width train step without writing a checkpoint.

    This is intentionally separate from :func:`train_dictionary`: it measures
    the full T16 model, gradients, and Adam state while leaving no giant
    one-step checkpoint that a later full run could mistake for progress.
    """

    if config.steps != 1:
        raise ValueError("memory smoke requires exactly one optimization step")
    cache = np.load(activation_cache, mmap_mode="r")
    if cache.ndim != 3 or cache.shape[-1] != config.d_in:
        raise ValueError(
            f"activation cache shape {cache.shape} is incompatible with "
            f"d_in={config.d_in}"
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA memory smoke requested but CUDA is unavailable")

    torch.manual_seed(config.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
        torch.set_float32_matmul_precision("high")
    model = build_model(config).to(
        device=device, dtype=_model_dtype(config, device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    raw = materialize_windows(
        cache,
        step=0,
        window=config.window,
        batch_size=config.batch_size,
        schedule_seed=config.schedule_seed,
        max_window=config.schedule_max_window or 6,
    )
    batch = torch.from_numpy(raw).to(
        device=device, dtype=_model_dtype(config, device)
    )
    model.pre_step()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda" if device.startswith("cuda") else "cpu",
        dtype=torch.bfloat16,
        enabled=config.amp and device.startswith("cuda"),
    ):
        step_result = (
            model.train_step(batch)
            if config.arch == "txc"
            else model.train_step(batch.reshape(-1, config.d_in))
        )
    step_result["loss"].backward()
    optimizer.step()
    model.post_step()

    nominal_l0 = int(model.k_win) if config.arch == "txc" else int(model.k)
    effective_l0 = float(step_result["l0"].detach().float().cpu())
    payload = {
        "status": "complete",
        "arch": config.arch,
        "window": config.window,
        "batch_size": config.batch_size,
        "d_in": config.d_in,
        "d_sae": config.d_sae,
        "dtype": str(_model_dtype(config, device)),
        "loss": float(step_result["loss"].detach().float().cpu()),
        "nominal_l0": nominal_l0,
        "effective_l0": effective_l0,
        "effective_l0_fill_fraction": effective_l0 / nominal_l0,
        "checkpoint_written": False,
    }
    if device.startswith("cuda"):
        payload.update(
            {
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            }
        )
    del batch, model, optimizer, step_result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload
