"""The single canonical pathway from (arch × seed × eval_cfg) → leaderboard row.

``run_cell`` is the **only** function that may append to ``leaderboard.jsonl``.
Components import it; they do not write their own caching logic.

Three things ``run_cell`` does:

1. Resolve config — load arch, datasource, training cfg, eval cfg.
2. Train (or load cached) — based on ``train_key``.
3. Evaluate (or skip cached) — based on ``eval_key``.

Adding an architecture, switching a datasource, fixing a metric bug —
all flow through here. See ``docs/paper/framework.md`` for the
ten principles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from temp_bench.cache import (
    append_leaderboard,
    checkpoint_exists,
    eval_in_leaderboard,
    load_checkpoint_state_dict,
    metrics_exist,
    save_checkpoint,
    save_metrics,
)
from temp_bench.config import (
    compute_act_cache_key,
    compute_eval_key,
    compute_train_key,
    list_archs as _list_archs,
    load_arch,
    load_datasource,
)
from temp_bench.schemas import LeaderboardRow, TrainingConfig


# ── Public types ─────────────────────────────────────────────────────────


class TrainFn(Protocol):
    """A training function. Components either use the default or supply their own."""
    def __call__(
        self,
        *,
        arch_name: str,
        arch_hparams: dict[str, Any],
        seed: int,
        training_cfg: TrainingConfig,
        act_cache_key: str,
        component: str,
    ) -> dict[str, Any]:
        """Return a state_dict ready for ``save_checkpoint``."""
        ...


class EvalFn(Protocol):
    """A per-component evaluation function."""
    def __call__(
        self,
        *,
        model: Any,
        eval_cfg: dict[str, Any],
        component: str,
    ) -> tuple[dict[str, float], str]:
        """Return ``(metrics_dict, primary_metric_key)``."""
        ...


@dataclass
class CellResult:
    eval_key: str
    train_key: str
    cached: bool
    metrics: dict[str, float] | None = None


# ── The core function ────────────────────────────────────────────────────


def run_cell(
    *,
    component: str,
    arch_name: str,
    seed: int,
    datasource_name: str,
    training_cfg: TrainingConfig | dict[str, Any] | None,
    eval_cfg: dict[str, Any],
    eval_protocol_version: str,
    train_fn: TrainFn,
    eval_fn: EvalFn,
    primary_metric: str | None = None,
    agent: str | None = None,
    force_train: bool = False,
    force_eval: bool = False,
) -> CellResult:
    """Run one (arch, seed, eval_cfg) cell, with caching.

    ``training_cfg=None`` uses :class:`TrainingConfig` defaults.
    ``primary_metric=None`` will be set by ``eval_fn``'s second return.
    """
    if training_cfg is None:
        training_cfg = TrainingConfig()
    elif isinstance(training_cfg, dict):
        training_cfg = TrainingConfig(**training_cfg)

    arch_spec = load_arch(arch_name, component=component)
    datasource = load_datasource(datasource_name)
    agent = agent or os.environ.get("AGENT_NAME", "unknown")

    # 1. Activation / data cache key
    act_cache_key = compute_act_cache_key(datasource)

    # 2. Train (or load cached)
    train_key = compute_train_key(
        arch=arch_spec,
        seed=seed,
        training_cfg=training_cfg,
        act_cache_key=act_cache_key,
    )

    if not force_train and checkpoint_exists(train_key):
        # We don't actually load the model state here — the eval_fn may
        # not need a Module instance (e.g. it might re-instantiate from
        # state_dict + arch class). To keep run_cell agnostic, we pass
        # a "checkpoint resolver" via the eval_cfg.
        loaded_state = load_checkpoint_state_dict(train_key)
    else:
        loaded_state = train_fn(
            arch_name=arch_name,
            arch_hparams=arch_spec.hparams,
            seed=seed,
            training_cfg=training_cfg,
            act_cache_key=act_cache_key,
            component=component,
        )
        save_checkpoint(
            train_key=train_key,
            arch=arch_name,
            arch_version=arch_spec.arch_version,
            seed=seed,
            datasource=datasource_name,
            act_cache_key=act_cache_key,
            training_cfg=training_cfg.model_dump(),
            state_dict=loaded_state,
            agent=agent,
        )

    # 3. Evaluate (or skip cached)
    eval_key = compute_eval_key(
        train_key=train_key,
        eval_protocol_version=eval_protocol_version,
        eval_cfg=eval_cfg,
    )

    if not force_eval and eval_in_leaderboard(eval_key) and metrics_exist(eval_key):
        return CellResult(eval_key=eval_key, train_key=train_key, cached=True)

    # eval_fn instantiates the model from arch_spec + loaded_state if
    # it wants a Module. We pass both via eval_cfg for clarity.
    enriched_cfg = {
        **eval_cfg,
        "_arch_name": arch_name,
        "_arch_hparams": arch_spec.hparams,
        "_state_dict": loaded_state,
        "_train_key": train_key,
    }
    metrics, primary = eval_fn(model=None, eval_cfg=enriched_cfg, component=component)
    primary = primary_metric or primary

    save_metrics(eval_key=eval_key, metrics=metrics, extras={"eval_cfg": eval_cfg})
    append_leaderboard(LeaderboardRow(
        eval_key=eval_key,
        train_key=train_key,
        act_cache_key=act_cache_key,
        component=component,
        arch=arch_name,
        arch_version=arch_spec.arch_version,
        seed=seed,
        datasource=datasource_name,
        eval_protocol_version=eval_protocol_version,
        eval_cfg=eval_cfg,
        metrics=metrics,
        primary_metric=primary,
        agent=agent,
        ts=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    ))
    return CellResult(eval_key=eval_key, train_key=train_key, cached=False, metrics=metrics)


# ── Helpers for component scripts ────────────────────────────────────────


def list_archs(category: str | None = None) -> list[str]:
    """All archs in the locked registry, optionally filtered by category."""
    names = _list_archs()
    if category is None:
        return names
    from temp_bench.config import load_arch as _la
    return [n for n in names if _la(n).category == category]


def default_training_cfg(arch_name: str) -> TrainingConfig:
    """Return a default :class:`TrainingConfig` for the named arch.

    Currently identical for every arch. Components override fields they
    need (e.g. C6 sets ``bricken_enabled=True`` and ``ema_auxk_alpha=1/8``).
    """
    return TrainingConfig()


# ── Pre-flight ───────────────────────────────────────────────────────────


def preflight() -> list[str]:
    """Check that the framework is in a runnable state.

    Returns a list of warnings (empty = clean). Run by
    ``scripts/agent_smoke_test.sh`` at the start of every agent session.

    Also checks GPU pinning on shared pods: when ``CUDA_VISIBLE_DEVICES``
    is set, exactly one GPU should be visible. When it's unset on a
    multi-GPU host, prints a critical warning (agents on shared pods
    must pin themselves to avoid collisions).
    """
    import os as _os

    warns: list[str] = []

    # ── GPU pinning check ────────────────────────────────────────────
    cuda_visible = _os.environ.get("CUDA_VISIBLE_DEVICES")
    agent = _os.environ.get("AGENT_NAME", "").strip()

    try:
        import torch
        n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n_visible = 0

    if n_visible > 1 and not cuda_visible:
        warns.append(
            f"CRITICAL: CUDA_VISIBLE_DEVICES is unset and {n_visible} GPUs "
            "are visible. On a shared pod this means two agents may collide "
            "on the same GPU. Run `source scripts/set_agent_env.sh <agent>` "
            "before any CUDA work."
        )
    elif n_visible > 1 and cuda_visible:
        warns.append(
            f"CUDA_VISIBLE_DEVICES={cuda_visible} but torch sees {n_visible} "
            "GPUs — pinning didn't take. Likely cause: an existing Python "
            "process initialised CUDA before the env var was set. Restart "
            "the shell."
        )

    if not agent:
        warns.append(
            "AGENT_NAME is unset — leaderboard rows will be tagged "
            "'unknown'. Source scripts/set_agent_env.sh <name>."
        )

    # ── Architecture imports ─────────────────────────────────────────
    for name in _list_archs():
        try:
            spec = load_arch(name)
        except Exception as e:
            warns.append(f"arch {name!r} failed to load: {e}")
            continue
        module_path, class_name = spec.class_path.split(":")
        try:
            mod = __import__(module_path, fromlist=[class_name])
            if not hasattr(mod, class_name):
                warns.append(f"arch {name!r}: class {class_name} not found in {module_path}")
        except ImportError as e:
            warns.append(f"arch {name!r}: cannot import {module_path}: {e}")

    return warns
