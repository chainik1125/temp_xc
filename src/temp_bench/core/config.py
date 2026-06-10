"""YAML registry loaders + deterministic cache-key computation.

Three YAML files under ``configs/``:

- ``archs.yaml``        — every architecture in the framework.
- ``data.yaml``         — every datasource (synthetic + real-LM).
- ``experiments.yaml``  — canonical per-paper-section sweep configs.

This module reads them, validates via :mod:`temp_bench.core.schemas`,
and computes the two cache keys:

- ``data_key``    — over a DataSourceSpec (synthetic or real).
- ``train_key``   — over (arch_class, arch_version, hparams, seed,
                     training_cfg, data_key).
- ``eval_key``    — over (train_key, evaluator_name,
                     evaluator_protocol_version, eval_cfg).

Keys are 16-hex-char prefixes of SHA-256 over canonicalised JSON.
Reproducible across machines.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from temp_bench.core.schemas import ArchSpec, DataSourceSpec, TrainingConfig

KEY_LEN = 16   # bytes of hex from sha256


# ── Paths ─────────────────────────────────────────────────────────────


def repo_root() -> Path:
    """Locate the repo root from $TEMP_BENCH_ROOT or by walking up."""
    env = os.environ.get("TEMP_BENCH_ROOT")
    if env:
        return Path(env).resolve()
    # src/temp_bench/core/config.py → ../../../..
    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
    return repo_root() / "configs"


# ── YAML loading ──────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_archs_yaml() -> dict[str, dict[str, Any]]:
    path = configs_dir() / "archs.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["archs"]


@lru_cache(maxsize=1)
def _load_data_yaml() -> dict[str, dict[str, Any]]:
    path = configs_dir() / "data.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["datasources"]


@lru_cache(maxsize=1)
def _load_experiments_yaml() -> dict[str, dict[str, Any]]:
    path = configs_dir() / "experiments.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("experiments", {})


def list_archs() -> list[str]:
    return sorted(_load_archs_yaml().keys())


def list_datasources() -> list[str]:
    return sorted(_load_data_yaml().keys())


def list_experiments() -> list[str]:
    return sorted(_load_experiments_yaml().keys())


def load_arch(name: str, *, section: str | None = None) -> ArchSpec:
    """Resolve an architecture by registry name.

    ``section`` is optional — if provided, ``per_section_hparams[section]``
    is merged onto the base ``hparams``. Lets a paper section override
    (e.g.) ``d_sae`` without forking the arch.
    """
    raw = _load_archs_yaml().get(name)
    if raw is None:
        raise KeyError(
            f"Unknown architecture {name!r}. "
            f"Registered: {list_archs()}. "
            "Add to configs/archs.yaml + drop a class in temp_bench/archs/."
        )
    # Ensure 'name' is on the spec for downstream.
    raw = {**raw, "name": name}
    spec = ArchSpec(**raw)

    if section is not None and spec.per_section_hparams:
        if section in spec.per_section_hparams:
            merged = {**spec.hparams, **spec.per_section_hparams[section]}
            return spec.model_copy(update={"hparams": merged})

    return spec


def load_datasource(name: str) -> DataSourceSpec:
    raw = _load_data_yaml().get(name)
    if raw is None:
        raise KeyError(
            f"Unknown datasource {name!r}. "
            f"Registered: {list_datasources()}. "
            "Add to configs/data.yaml."
        )
    raw = {**raw, "name": name}
    return DataSourceSpec(**raw)


def load_experiment(name: str) -> dict[str, Any]:
    """Load a canonical paper-section sweep config."""
    raw = _load_experiments_yaml().get(name)
    if raw is None:
        raise KeyError(
            f"Unknown experiment {name!r}. "
            f"Registered: {list_experiments()}. "
            "Add to configs/experiments.yaml."
        )
    return raw


# ── Cache-key computation ─────────────────────────────────────────────


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _hash(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()[:KEY_LEN]


def compute_data_key(datasource: DataSourceSpec | str) -> str:
    """Deterministic key for one datasource spec (synthetic or real).

    For synthetic data: hashes the generator + params. Different seeds
    of the SAME generator produce the same data_key (seed is in
    training_cfg, not the data spec).

    For real-LM data: hashes the subject_model + layer + hookpoint +
    dataset + n_seqs + seq_len. Identical specs across runs share the
    same on-disk activation cache.
    """
    if isinstance(datasource, str):
        datasource = load_datasource(datasource)
    inputs = datasource.model_dump(exclude={"notes"})
    return _hash(inputs)


def compute_train_key(
    *,
    arch: ArchSpec | str,
    seed: int,
    training_cfg: TrainingConfig | dict[str, Any],
    data_key: str,
    section: str | None = None,
) -> str:
    """Deterministic key for a trained-model cache entry."""
    if isinstance(arch, str):
        arch = load_arch(arch, section=section)
    if isinstance(training_cfg, TrainingConfig):
        # exclude_none=True keeps defaults out of the hash so we don't
        # invalidate existing cached models when adding optional fields.
        training_cfg_dict = training_cfg.model_dump(exclude_none=True)
    else:
        training_cfg_dict = dict(training_cfg)

    inputs = {
        "arch_class": arch.class_path,
        "arch_version": arch.arch_version,
        "hparams": arch.hparams,
        "seed": int(seed),
        "training_cfg": training_cfg_dict,
        "data_key": data_key,
    }
    return _hash(inputs)


def compute_eval_key(
    *,
    train_key: str,
    evaluator_name: str,
    evaluator_protocol_version: str,
    eval_cfg: dict[str, Any],
) -> str:
    """Deterministic key for one eval cell."""
    inputs = {
        "train_key": train_key,
        "evaluator_name": evaluator_name,
        "evaluator_protocol_version": evaluator_protocol_version,
        "eval_cfg": dict(eval_cfg),
    }
    return _hash(inputs)


# ── Path helpers ──────────────────────────────────────────────────────


def data_cache_dir(data_key: str) -> Path:
    return repo_root() / "results" / "data_cache" / data_key


def checkpoint_dir(train_key: str) -> Path:
    return repo_root() / "checkpoints" / train_key


def run_dir(eval_key: str) -> Path:
    return repo_root() / "results" / "runs" / eval_key


# ── Plugin loading by class_path ──────────────────────────────────────


def import_by_path(class_path: str) -> Any:
    """Resolve ``module.path:ClassName`` to the actual class object."""
    if ":" not in class_path:
        raise ValueError(
            f"class_path must be 'module:ClassName'; got {class_path!r}."
        )
    module_path, name = class_path.split(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, name)
