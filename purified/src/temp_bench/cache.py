"""Cache helpers: checkpoint save/load + leaderboard append.

All multi-agent shared state goes through this module. The leaderboard
and checkpoint manifest are append-only JSONL files protected by
``fcntl.flock`` for concurrent appends.

If you find yourself writing to ``leaderboard.jsonl`` or
``manifest.jsonl`` from anywhere except :func:`append_leaderboard` /
:func:`append_checkpoint_manifest`, **stop**. The framework guarantees
schema-checked rows; bypassing it produces corrupt state that future
agents can't trust.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Iterator

from temp_bench.config import (
    checkpoint_dir,
    purified_root,
    run_dir,
)
from temp_bench.schemas import CheckpointManifest, LeaderboardRow


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Append-only JSONL with flock ─────────────────────────────────────────


def _flocked_append(path: Path, row_json: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(row_json + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ── Leaderboard ──────────────────────────────────────────────────────────


def leaderboard_path() -> Path:
    return purified_root() / "results" / "leaderboard.jsonl"


def append_leaderboard(row: LeaderboardRow | dict) -> None:
    """Validate against schema and append. Raises pydantic ValidationError
    if the row is malformed. Do not catch — let the runner abort the
    cell so we don't write corrupt state.
    """
    if not isinstance(row, LeaderboardRow):
        row = LeaderboardRow(**row)
    _flocked_append(leaderboard_path(), row.model_dump_json())


def eval_in_leaderboard(eval_key: str) -> bool:
    """Has this eval_key already been recorded?"""
    for r in _read_jsonl(leaderboard_path()):
        if r.get("eval_key") == eval_key:
            return True
    return False


def iter_leaderboard() -> Iterator[LeaderboardRow]:
    """Yield validated rows. Skip rows that fail schema validation
    (with a stderr warning) — they are likely from a future schema
    version we don't understand.
    """
    import sys
    for raw in _read_jsonl(leaderboard_path()):
        try:
            yield LeaderboardRow(**raw)
        except Exception as e:
            print(
                f"[temp_bench.cache] skipping malformed leaderboard row: {e}",
                file=sys.stderr,
            )


# ── Checkpoint manifest ──────────────────────────────────────────────────


def manifest_path() -> Path:
    return purified_root() / "checkpoints" / "manifest.jsonl"


def append_checkpoint_manifest(entry: CheckpointManifest | dict) -> None:
    if not isinstance(entry, CheckpointManifest):
        entry = CheckpointManifest(**entry)
    _flocked_append(manifest_path(), entry.model_dump_json())


def checkpoint_exists(train_key: str) -> bool:
    """A checkpoint exists if its directory has the expected files."""
    d = checkpoint_dir(train_key)
    return d.exists() and (d / "model.safetensors").exists() and (d / "config.json").exists()


def save_checkpoint(
    *,
    train_key: str,
    arch: str,
    arch_version: str,
    seed: int,
    datasource: str,
    act_cache_key: str,
    training_cfg: dict[str, Any],
    state_dict: dict[str, "torch.Tensor"],   # noqa: F821 (torch only at runtime)
    extra_files: dict[str, bytes] | None = None,
    agent: str = "unknown",
) -> Path:
    """Persist a trained model under ``checkpoints/<train_key>/`` and
    append a manifest row. Idempotent: re-calling overwrites the
    checkpoint dir but the manifest row is append-only (kept).

    On ephemeral pods (``TEMP_BENCH_POD_MODE=ephemeral``), the freshly
    saved checkpoint is **also pushed to HuggingFace** so it survives
    pod stop. Failure to push is fatal — see hardware.md *Pod modes*.
    """
    from safetensors.torch import save_file  # local import — only needed at save time

    d = checkpoint_dir(train_key)
    d.mkdir(parents=True, exist_ok=True)

    save_file(state_dict, str(d / "model.safetensors"))
    config = {
        "train_key": train_key,
        "arch": arch,
        "arch_version": arch_version,
        "seed": seed,
        "datasource": datasource,
        "act_cache_key": act_cache_key,
        "training_cfg": training_cfg,
        "saved_ts": _now_iso(),
        "agent": agent,
    }
    (d / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    if extra_files:
        for fname, content in extra_files.items():
            (d / fname).write_bytes(content)

    size_mb = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e6
    hf_url = None

    # On ephemeral pods: auto-push to HF so the checkpoint survives pod stop.
    if os.environ.get("TEMP_BENCH_POD_MODE") == "ephemeral":
        hf_url = _push_checkpoint_to_hf(train_key, d, agent=agent)

    append_checkpoint_manifest(CheckpointManifest(
        train_key=train_key,
        act_cache_key=act_cache_key,
        arch=arch,
        arch_version=arch_version,
        seed=seed,
        datasource=datasource,
        training_cfg=training_cfg,
        local_path=str(d / "model.safetensors"),
        hf_url=hf_url,
        size_mb=size_mb,
        agent=agent,
        ts=_now_iso(),
    ))
    return d


def _push_checkpoint_to_hf(train_key: str, ckpt_dir: Path, *, agent: str) -> str:
    """Push a checkpoint dir to ``han1823123123/temp-bench-models``.

    Auto-called from :func:`save_checkpoint` on ephemeral pods.
    Failure is fatal — silently dropping a checkpoint that the pod
    might lose to a restart is a worse failure mode than aborting.
    """
    from huggingface_hub import HfApi

    repo_id = "han1823123123/temp-bench-models"
    api = HfApi()
    api.upload_folder(
        folder_path=str(ckpt_dir),
        path_in_repo=train_key,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"agent={agent} train_key={train_key}",
    )
    return f"https://huggingface.co/{repo_id}/tree/main/{train_key}"


def load_checkpoint_state_dict(train_key: str) -> dict[str, "torch.Tensor"]:  # noqa: F821
    """Load the state dict at ``checkpoints/<train_key>/model.safetensors``."""
    from safetensors.torch import load_file
    p = checkpoint_dir(train_key) / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"No checkpoint at {p} (train_key={train_key})")
    return load_file(str(p))


# ── Run-dir helpers (per-eval artifacts) ─────────────────────────────────


def save_metrics(*, eval_key: str, metrics: dict[str, float], extras: dict[str, Any] | None = None) -> Path:
    d = run_dir(eval_key)
    d.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "ts": _now_iso()}
    if extras:
        payload["extras"] = extras
    (d / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return d


def metrics_exist(eval_key: str) -> bool:
    return (run_dir(eval_key) / "metrics.json").exists()
