"""YAML config loaders + cache-key computation.

The two yaml files under ``purified/configs/`` are the source of truth
for what the paper claims. This module reads them, validates against
the schemas in :mod:`temp_bench.schemas`, and provides deterministic
cache keys.

Cache-key inputs are JSON-canonicalised (sorted keys, no whitespace) so
the SHA-256 is reproducible across machines.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from temp_bench.schemas import ArchSpec, DataSourceSpec, TrainingConfig

KEY_LEN = 16  # bytes of hex from sha256


def purified_root() -> Path:
    """Locate purified/ from $TEMP_BENCH_ROOT or by walking up from this file."""
    env = os.environ.get("TEMP_BENCH_ROOT")
    if env:
        return Path(env).resolve()
    # src/temp_bench/config.py → ../../..
    return Path(__file__).resolve().parents[2]


# ── YAML loading ─────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_archs_yaml() -> dict[str, dict[str, Any]]:
    path = purified_root() / "configs" / "locked_archs.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["archs"]


@lru_cache(maxsize=1)
def _load_datasources_yaml() -> dict[str, dict[str, Any]]:
    path = purified_root() / "configs" / "datasources.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["datasources"]


def list_archs() -> list[str]:
    """Names of all architectures in the locked registry."""
    return sorted(_load_archs_yaml().keys())


def list_datasources() -> list[str]:
    return sorted(_load_datasources_yaml().keys())


def load_arch(name: str, *, component: str | None = None) -> ArchSpec:
    """Resolve an architecture by name.

    ``component`` is optional — if provided, ``per_component_hparams[component]``
    are merged on top of the base ``hparams``. This is how a component
    supplies setting-specific overrides (e.g., a component's d_sae) without
    forking the architecture.
    """
    raw = _load_archs_yaml().get(name)
    if raw is None:
        raise KeyError(
            f"Unknown architecture {name!r}. Locked set: {list_archs()}. "
            "Adding a new arch requires (1) an entry in "
            "configs/locked_archs.yaml and (2) a class implementing "
            "TempBenchArch. See docs/paper/framework.md."
        )
    spec = ArchSpec(**raw)

    if component is not None and component in spec.per_component_hparams:
        merged = {**spec.hparams, **spec.per_component_hparams[component]}
        # Re-create spec with merged hparams; rest unchanged.
        return spec.model_copy(update={"hparams": merged})

    return spec


def load_datasource(name: str) -> DataSourceSpec:
    raw = _load_datasources_yaml().get(name)
    if raw is None:
        raise KeyError(
            f"Unknown datasource {name!r}. Available: {list_datasources()}. "
            "Adding a datasource requires an entry in configs/datasources.yaml."
        )
    return DataSourceSpec(**raw)


def _resolve_class(class_path: str):
    """Resolve a ``module.path:ClassName`` string to the actual class."""
    import importlib
    if ":" not in class_path:
        raise ValueError(
            f"class_path must be of the form 'module:Class', got {class_path!r}"
        )
    module_path, class_name = class_path.split(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(
            f"Module {module_path!r} has no attribute {class_name!r}. "
            f"Check that {class_path!r} matches the class definition."
        )
    return cls


def instantiate_arch(spec: ArchSpec, *, d_in: int):
    """Build the architecture from a yaml spec + datasource ``d_in``.

    This is the canonical pathway from ``configs/locked_archs.yaml`` to
    a concrete :class:`temp_bench.architectures.base.TempBenchArch`
    instance. Components do NOT call class constructors directly; they
    use this helper so that:

    1. ``per_component_hparams`` overrides are merged exactly once
       (via :func:`load_arch`).
    2. The class file is the single source of truth — adding a component
       does not require touching the class.
    3. ``d_in`` flows in from the datasource, not the YAML — so switching
       Gemma → Llama doesn't need YAML edits beyond the datasource.

    Example::

        spec = load_arch("txc_base", component="c7")     # merges c7 d_sae=32768
        ds = load_datasource("llama_3_1_8b_base_l10_ward")
        d_in = 4096   # from the datasource (real-LM) or generator (toy)
        model = instantiate_arch(spec, d_in=d_in)
    """
    cls = _resolve_class(spec.class_path)
    return cls(d_in=d_in, **spec.hparams)


# ── Cache-key computation ────────────────────────────────────────────────


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _hash(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()[:KEY_LEN]


def compute_act_cache_key(datasource: DataSourceSpec | str) -> str:
    """Deterministic key for an activation (or synthetic data spec) cache."""
    if isinstance(datasource, str):
        datasource = load_datasource(datasource)
    inputs = datasource.model_dump(exclude={"notes"})
    return _hash(inputs)


def compute_train_key(
    *,
    arch: ArchSpec | str,
    seed: int,
    training_cfg: TrainingConfig | dict[str, Any],
    act_cache_key: str,
    component: str | None = None,
) -> str:
    """Deterministic key for a trained-model cache entry.

    Two cells with identical inputs share the cached checkpoint —
    important so C3, C4, C5 don't each retrain TXC-base from scratch.
    Bumping ``arch_version`` is the canonical way to invalidate.
    """
    if isinstance(arch, str):
        arch = load_arch(arch, component=component)
    if isinstance(training_cfg, TrainingConfig):
        training_cfg_dict = training_cfg.model_dump()
    else:
        training_cfg_dict = dict(training_cfg)

    inputs = {
        "arch_class": arch.class_path,
        "arch_version": arch.arch_version,
        "hparams": arch.hparams,
        "seed": int(seed),
        "training_cfg": training_cfg_dict,
        "act_cache_key": act_cache_key,
    }
    return _hash(inputs)


def compute_eval_key(
    *,
    train_key: str,
    eval_protocol_version: str,
    eval_cfg: dict[str, Any],
) -> str:
    """Deterministic key for one cell of (train_key × eval protocol × eval cfg)."""
    inputs = {
        "train_key": train_key,
        "eval_protocol_version": eval_protocol_version,
        "eval_cfg": dict(eval_cfg),
    }
    return _hash(inputs)


# ── Path helpers (single source of truth for cache layout) ────────────────


def act_cache_dir(act_cache_key: str) -> Path:
    return purified_root() / "results" / "act_cache" / act_cache_key


def checkpoint_dir(train_key: str) -> Path:
    return purified_root() / "checkpoints" / train_key


def run_dir(eval_key: str) -> Path:
    return purified_root() / "results" / "runs" / eval_key
