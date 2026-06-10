"""Leaderboard + manifest I/O — the framework's only writers.

Two append-only JSONL files are the single source of truth for results:

- ``results/leaderboard.jsonl`` — one row per evaluated cell.
- ``checkpoints/manifest.jsonl`` — one row per trained checkpoint.

Writes are flock-protected so concurrent processes don't interleave.
Reads validate against the Pydantic schemas in
:mod:`temp_bench.core.schemas`. Bad lines abort the read; the runner
refuses to write a row that fails schema validation.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from temp_bench.core.config import repo_root
from temp_bench.core.schemas import (
    SCHEMA_VERSION,
    CheckpointManifest,
    LeaderboardRow,
)


# ── Time helpers ───────────────────────────────────────────────────────


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Low-level JSONL with file-locking ──────────────────────────────────


@contextmanager
def _flocked(path: Path, mode: str = "a") -> Iterator:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open in append mode; flock at the file descriptor level.
    with open(path, mode) as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield f
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # Defensive: skip merge-conflict markers if any (we've had two
            # such incidents on `final`; not letting them poison reads).
            if line.startswith("<<<<<<<") or line.startswith(">>>>>>>") or line == "=======":
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"{path}: line {i} is not valid JSON ({e}). "
                    "Either truncate the file to last valid line, "
                    "or regenerate from runs/ via run.py rebuild-leaderboard."
                )


# ── Leaderboard ────────────────────────────────────────────────────────


def leaderboard_path() -> Path:
    return repo_root() / "results" / "leaderboard.jsonl"


def append_leaderboard(row: LeaderboardRow | dict) -> None:
    """Validate and append one row to ``leaderboard.jsonl``.

    Accepts either a :class:`LeaderboardRow` or a dict; if a dict it is
    validated against the Pydantic schema first. Schema rejection raises
    and the runner aborts — no partial writes.
    """
    if isinstance(row, dict):
        row = LeaderboardRow(**row)   # validate
    if row.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: row has {row.schema_version!r}, "
            f"current is {SCHEMA_VERSION!r}. Bump SCHEMA_VERSION carefully "
            "(breaking change to LeaderboardRow / CheckpointManifest)."
        )
    line = row.model_dump_json() + "\n"
    with _flocked(leaderboard_path(), "a") as f:
        f.write(line)


def iter_leaderboard() -> Iterator[LeaderboardRow]:
    """Yield validated rows. Bad lines raise."""
    for raw in _read_jsonl(leaderboard_path()):
        yield LeaderboardRow(**raw)


def eval_in_leaderboard(eval_key: str) -> bool:
    """Cache-hit query: has ``eval_key`` already been written?"""
    for raw in _read_jsonl(leaderboard_path()):
        if raw.get("eval_key") == eval_key:
            return True
    return False


def find_row(eval_key: str) -> LeaderboardRow | None:
    """Return the latest row with the given ``eval_key`` (or None)."""
    latest = None
    for raw in _read_jsonl(leaderboard_path()):
        if raw.get("eval_key") == eval_key:
            latest = raw   # keep last (jsonl is append-only; last write wins)
    return LeaderboardRow(**latest) if latest else None


# ── Manifest (checkpoints) ─────────────────────────────────────────────


def manifest_path() -> Path:
    return repo_root() / "checkpoints" / "manifest.jsonl"


def append_manifest(entry: CheckpointManifest | dict) -> None:
    if isinstance(entry, dict):
        entry = CheckpointManifest(**entry)
    line = entry.model_dump_json() + "\n"
    with _flocked(manifest_path(), "a") as f:
        f.write(line)


def iter_manifest() -> Iterator[CheckpointManifest]:
    for raw in _read_jsonl(manifest_path()):
        yield CheckpointManifest(**raw)


def checkpoint_exists(train_key: str) -> bool:
    """Cache-hit query for trained models.

    True iff EITHER (a) ``checkpoints/<train_key>/model.safetensors``
    exists on disk OR (b) a manifest entry for ``train_key`` is recorded
    with an HF URL set (i.e., we can pull on demand).
    """
    from temp_bench.core.config import checkpoint_dir
    local = (checkpoint_dir(train_key) / "model.safetensors").exists()
    if local:
        return True
    for raw in _read_jsonl(manifest_path()):
        if raw.get("train_key") == train_key and raw.get("hf_url"):
            return True
    return False
