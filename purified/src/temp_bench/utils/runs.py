"""Run-id allocation and leaderboard append.

Concurrency model: ``leaderboard.jsonl`` and ``manifest.jsonl`` are
**append-only**. Every concurrent agent appends without reading. We use
``fcntl.flock`` to make the append atomic on POSIX filesystems.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
import secrets
from pathlib import Path
from typing import Any


def make_run_id(component: str, arch: str, seed: int) -> str:
    """Deterministic-prefix, random-suffix id. Compute *before* training.

    Format: ``<component>_<arch>_<seed>_<8 hex chars>``. The hex suffix
    avoids collisions across agents that picked the same (component, arch, seed)
    independently.
    """
    if not component.startswith("c") or not component[1:].isdigit():
        raise ValueError(f"component must look like 'c3', got {component!r}")
    short = secrets.token_hex(4)
    return f"{component}_{arch}_{seed}_{short}"


def _purified_root() -> Path:
    """Locate purified/ from $TEMP_BENCH_ROOT or by walking up from this file."""
    env = os.environ.get("TEMP_BENCH_ROOT")
    if env:
        return Path(env).resolve()
    # src/temp_bench/utils/runs.py → ../../..
    return Path(__file__).resolve().parents[3]


def _append_jsonl(rel_path: str, row: dict[str, Any]) -> Path:
    path = _purified_root() / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return path


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_leaderboard(
    *,
    run_id: str,
    component: str,
    arch: str,
    seed: int,
    metric: str,
    value: float,
    agent: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Append one row to ``results/leaderboard.jsonl``."""
    row = {
        "run_id": run_id,
        "component": component,
        "arch": arch,
        "seed": seed,
        "metric": metric,
        "value": float(value),
        "agent": agent,
        "ts": _now_iso(),
    }
    if extra:
        row.update(extra)
    return _append_jsonl("results/leaderboard.jsonl", row)


def append_checkpoint(
    *,
    run_id: str,
    hf_url: str | None = None,
    local_path: str | None = None,
    size_mb: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Append one row to ``checkpoints/manifest.jsonl``."""
    if not (hf_url or local_path):
        raise ValueError("either hf_url or local_path must be set")
    row: dict[str, Any] = {"run_id": run_id, "ts": _now_iso()}
    if hf_url:
        row["hf_url"] = hf_url
    if local_path:
        row["local_path"] = local_path
    if size_mb is not None:
        row["size_mb"] = size_mb
    if extra:
        row.update(extra)
    return _append_jsonl("checkpoints/manifest.jsonl", row)
