"""Single canonical SAE trainer — used by every component.

Components do not write training loops. They call :func:`train_sae`
and pass a ``model`` (built via :func:`temp_bench.config.instantiate_arch`),
a ``batch_iter`` (callable: ``int -> Tensor``), and a ``TrainingConfig``.
The trainer handles:

- optimizer + LR warmup + linear decay
- gradient clipping + grad-norm tracking
- decoder unit-norm projection via :meth:`TempBenchArch.post_step`
- per-step diagnostics from :meth:`TempBenchArch.train_step`
- optional Bricken dead-feature resample (if ``training_cfg.bricken_enabled``)
- intermediate snapshots every N steps (so a crashed run is recoverable)
- bf16 / fp16 mixed precision via the ``training_cfg.precision`` field

Per-arch behaviour (auxK, contrastive, matryoshka) lives inside the
arch's ``train_step``. The trainer does NOT branch on arch type; it
just calls the abstract methods. This is what enforces "one trainer
for all SAE-family archs" — see PROTOCOL.md § 11 *Code reuse contract*.

Returns:
    {
        "state_dict": OrderedDict,         # final model weights
        "log":        dict[str, list],     # per-step metrics ('loss', 'mse', 'l0', 'lr', 'grad_norm', ...)
        "n_steps":    int,                 # actually run (early-stopped or full)
        "bricken":    list[ResampleStats]  # if Bricken was enabled, else []
    }
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn

from temp_bench.architectures.base import TempBenchArch
from temp_bench.schemas import TrainingConfig

BatchIter = Callable[[int], torch.Tensor]


def _make_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    raise ValueError(f"Unknown optimizer {cfg.optimizer!r}; expected 'adam' or 'adamw'.")


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _lr_at(step: int, cfg: TrainingConfig) -> float:
    """Linear warmup over ``warmup_steps``, then constant. Decay (if any)
    is the arch's responsibility — most don't decay; the few that do
    encode it in ``train_step``."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    return cfg.learning_rate


def _autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32" or precision is None:
        return None
    raise ValueError(f"Unknown precision {precision!r}; expected bf16/fp16/fp32.")


def train_sae(
    model: TempBenchArch,
    batch_iter: BatchIter,
    training_cfg: TrainingConfig,
    *,
    device: str | torch.device = "cuda",
    snapshot_every: int = 0,
    snapshot_fn: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the canonical SAE training loop.

    Args:
        model: TempBenchArch instance — must implement ``encode``,
            ``decode``, optionally ``train_step`` (default = MSE) and
            ``post_step`` (default = no-op).
        batch_iter: Callable that takes a batch size and returns a
            tensor on CPU (the trainer moves it to device). Same iter
            is used for the training batch and Bricken's check batch.
        training_cfg: see :class:`temp_bench.schemas.TrainingConfig`.
        device: target device. Use ``"cpu"`` for unit tests.
        snapshot_every: if > 0, call ``snapshot_fn`` every N steps with
            an intermediate state. ``0`` disables (only the final state
            is saved by the runner).
        snapshot_fn: callable invoked with ``(step, payload_dict)``.

    Returns:
        See module docstring.
    """
    device = torch.device(device)
    model.to(device).train()

    optimizer = _make_optimizer(model, training_cfg)
    autocast_dtype = _autocast_dtype(training_cfg.precision)
    use_amp = autocast_dtype is not None and device.type == "cuda"

    log: dict[str, list[float]] = defaultdict(list)
    bricken_stats: list[Any] = []

    # ── Optional Bricken resampler (opt-in per training_cfg.bricken_enabled)
    bricken = None
    if training_cfg.bricken_enabled:
        from temp_bench.training.bricken import BrickenConfig, BrickenResampler
        bricken = BrickenResampler(
            arch=model,
            cfg=BrickenConfig(
                resample_every=training_cfg.bricken_resample_every,
                min_fires=training_cfg.bricken_min_fires,
                n_check=training_cfg.bricken_n_check,
                max_resample_fraction=training_cfg.bricken_max_resample_fraction,
            ),
        )

    plateau_window: list[float] = []
    last_step = 0

    for step in range(training_cfg.n_steps):
        last_step = step

        _set_lr(optimizer, _lr_at(step, training_cfg))

        x = batch_iter(training_cfg.batch_size).to(device)

        if use_amp:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                loss, info = model.train_step(x)
        else:
            loss, info = model.train_step(x)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.post_step()

        # ── Logging (detached scalars only)
        log["loss"].append(float(loss.detach()))
        log["mse"].append(float(info.get("mse", loss.detach())))
        log["l0"].append(float(info.get("l0", 0.0)))
        log["lr"].append(_lr_at(step, training_cfg))
        log["grad_norm"].append(float(grad_norm))

        # ── Plateau early stopping (optional)
        if training_cfg.plateau_early_stop and step >= training_cfg.plateau_patience:
            plateau_window.append(log["loss"][-1])
            if len(plateau_window) > training_cfg.plateau_patience:
                plateau_window.pop(0)
            recent = plateau_window
            if len(recent) == training_cfg.plateau_patience:
                if max(recent) - min(recent) < training_cfg.plateau_min_delta:
                    break

        # ── Bricken resample
        if bricken is not None:
            check_fn = lambda: batch_iter(bricken.cfg.n_check)
            fired = bricken.maybe_resample(step=step, check_batch_fn=check_fn)
            if fired:
                bricken_stats.append(bricken.last_stats)

        # ── Intermediate snapshot
        if snapshot_every and step > 0 and step % snapshot_every == 0 and snapshot_fn:
            snapshot_fn(step, {"state_dict": model.state_dict(), "log": dict(log)})

    return {
        "state_dict": model.state_dict(),
        "log": dict(log),
        "n_steps": last_step + 1,
        "bricken": bricken_stats,
    }
