"""Frozen protocol and small pure helpers for the C7 ``T=1..6`` sweep."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROTOCOL_VERSION = "2026-07-23.2"
EXPECTED_ARTIFACT_SHA256 = (
    "1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810"
)
EXPECTED_ARTIFACT_SHAPE = (25_204, 6, 4_096)
EXPECTED_CACHE_SHAPE = (4_044, 128, 4_096)
ARTIFACT_OFFSETS = (-13, -12, -11, -10, -9, -8)
FULL_WINDOWS = (1, 2, 3, 4, 5, 6)
FULL_SEEDS = (1, 2, 42)
ORDER_CONTROLS = ("shuffle", "reverse", "circular")


@dataclass(frozen=True)
class SweepProfile:
    mode: str
    windows: tuple[int, ...]
    seeds: tuple[int, ...]
    d_sae: int
    k_pos: int
    steps: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    checkpoint_every: int
    folds: int
    s_grid: tuple[int, ...]
    max_rows: int | None
    bootstrap_repeats: int
    amp: bool


def profile(mode: str) -> SweepProfile:
    if mode == "smoke":
        return SweepProfile(
            mode=mode,
            windows=(1, 2),
            seeds=(42,),
            d_sae=128,
            k_pos=2,
            steps=2,
            batch_size=16,
            learning_rate=3e-4,
            warmup_steps=0,
            checkpoint_every=1,
            folds=2,
            s_grid=(4, 8),
            max_rows=800,
            bootstrap_repeats=50,
            amp=False,
        )
    if mode == "full":
        return SweepProfile(
            mode=mode,
            windows=FULL_WINDOWS,
            seeds=FULL_SEEDS,
            d_sae=32_768,
            k_pos=20,
            steps=20_000,
            batch_size=1_024,
            learning_rate=3e-4,
            warmup_steps=1_000,
            checkpoint_every=1_000,
            folds=5,
            s_grid=(8, 16, 32),
            max_rows=None,
            bootstrap_repeats=2_000,
            amp=True,
        )
    raise ValueError(f"mode must be smoke or full, got {mode!r}")


def csv_ints(value: str | None, default: Iterable[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(int(item) for item in default)
    parsed = tuple(int(part) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("integer list must not be empty")
    return parsed


def validate_axes(windows: tuple[int, ...], seeds: tuple[int, ...]) -> None:
    if len(set(windows)) != len(windows):
        raise ValueError(f"duplicate windows: {windows}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")
    invalid = [window for window in windows if window not in FULL_WINDOWS]
    if invalid:
        raise ValueError(
            f"the current event artifact supports only T=1..6; got {invalid}"
        )


def physical_offsets(window: int) -> tuple[int, ...]:
    if window not in FULL_WINDOWS:
        raise ValueError(f"window must be in {FULL_WINDOWS}, got {window}")
    return ARTIFACT_OFFSETS[-window:]


def cell_name(window: int, seed: int) -> str:
    return f"T{window}_seed{seed}"


def seed_queue(seeds: tuple[int, ...]) -> tuple[int, ...]:
    """Put the submitted seed first, then preserve the requested order."""

    return tuple(sorted(seeds, key=lambda seed: (seed != 42, seeds.index(seed))))


def window_queue(windows: tuple[int, ...]) -> tuple[int, ...]:
    """Run the endpoint gate before filling the complete curve."""

    priority = {1: 0, 6: 1}
    return tuple(sorted(windows, key=lambda window: (priority.get(window, 2), window)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def profile_dict(value: SweepProfile) -> dict:
    payload = asdict(value)
    for key in ("windows", "seeds", "s_grid"):
        payload[key] = list(payload[key])
    return payload


def artifact_inventory(
    artifact: Path,
    activation_cache: Path,
    *,
    strict_full: bool,
) -> dict:
    """Inspect the two required arrays without materializing them."""

    result: dict = {
        "artifact": str(artifact),
        "activation_cache": str(activation_cache),
        "missing": [],
    }
    if not artifact.exists():
        result["missing"].append(str(artifact))
    else:
        with np.load(artifact, allow_pickle=True, mmap_mode="r") as payload:
            keys = sorted(payload.files)
            x_shape = tuple(int(value) for value in payload["X"].shape)
            y_shape = tuple(int(value) for value in payload["is_bt"].shape)
            key_shape = tuple(int(value) for value in payload["keys"].shape)
        result["artifact_keys"] = keys
        result["artifact_x_shape"] = list(x_shape)
        result["artifact_label_shape"] = list(y_shape)
        result["artifact_key_shape"] = list(key_shape)
        result["artifact_shape_ok"] = (
            x_shape == EXPECTED_ARTIFACT_SHAPE
            if strict_full
            else len(x_shape) == 3 and x_shape[1:] == EXPECTED_ARTIFACT_SHAPE[1:]
        )
        if strict_full:
            result["artifact_sha256"] = sha256(artifact)
            result["artifact_sha256_ok"] = (
                result["artifact_sha256"] == EXPECTED_ARTIFACT_SHA256
            )
    if not activation_cache.exists():
        result["missing"].append(str(activation_cache))
    else:
        cache = np.load(activation_cache, mmap_mode="r")
        result["activation_cache_shape"] = [int(value) for value in cache.shape]
        result["activation_cache_dtype"] = str(cache.dtype)
        result["activation_cache_shape_ok"] = (
            tuple(cache.shape) == EXPECTED_CACHE_SHAPE
            if strict_full
            else cache.ndim == 3 and cache.shape[-1] == EXPECTED_CACHE_SHAPE[-1]
        )
    return result


def assert_inventory(inventory: dict, *, strict_full: bool) -> None:
    if inventory["missing"]:
        raise FileNotFoundError(
            "missing required backtracking artifact(s): "
            + ", ".join(inventory["missing"])
        )
    checks = ["artifact_shape_ok", "activation_cache_shape_ok"]
    if strict_full:
        checks.append("artifact_sha256_ok")
    failures = {key: inventory.get(key) for key in checks if not inventory.get(key)}
    if failures:
        raise ValueError(f"backtracking artifact provenance mismatch: {failures}")


def whole_group_subsample(
    groups: np.ndarray, max_rows: int | None, seed: int
) -> np.ndarray:
    if max_rows is None or max_rows >= len(groups):
        return np.arange(len(groups), dtype=np.int64)
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for group in rng.permutation(np.unique(groups)):
        keep.extend(np.flatnonzero(groups == group).tolist())
        if len(keep) >= max_rows:
            break
    return np.asarray(sorted(keep), dtype=np.int64)
