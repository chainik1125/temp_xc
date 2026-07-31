"""Pure protocol helpers for the Assistant-Axis persona-drift experiment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

Split = Literal["train", "validation", "test"]

EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_ROOT / "config.json"


@dataclass(frozen=True)
class ProbeIndex:
    """One causal prediction row."""

    conversation_index: int
    conversation_id: str
    split: Split
    domain: str
    turn: int
    window: int
    horizon: int


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for_persona(persona_index: int) -> Split:
    """Conversation-safe split: three train, one validation, one test persona."""
    remainder = int(persona_index) % 5
    if remainder <= 2:
        return "train"
    if remainder == 3:
        return "validation"
    return "test"


def validate_script_record(record: dict[str, Any], turns: int) -> None:
    required = {
        "conversation_id",
        "domain",
        "persona_id",
        "topic_id",
        "split",
        "user_messages",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"script record missing fields: {sorted(missing)}")
    if record["split"] != split_for_persona(int(record["persona_id"])):
        raise ValueError(f"{record['conversation_id']}: split is not determined by persona_id")
    messages = record["user_messages"]
    if not isinstance(messages, list) or len(messages) != turns:
        raise ValueError(
            f"{record['conversation_id']}: expected {turns} user messages, "
            f"got {len(messages) if isinstance(messages, list) else type(messages)}"
        )
    if not all(isinstance(message, str) and message.strip() for message in messages):
        raise ValueError(f"{record['conversation_id']}: empty/non-string user message")


def build_probe_indices(
    metadata: Sequence[dict[str, Any]],
    *,
    turns_per_conversation: Sequence[int],
    window: int,
    horizon: int,
) -> list[ProbeIndex]:
    """Build rows whose full past window and future horizon are in one conversation."""
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be positive")
    if len(metadata) != len(turns_per_conversation):
        raise ValueError("metadata and turn counts must align")
    rows: list[ProbeIndex] = []
    for conversation_index, (record, n_turns) in enumerate(
        zip(metadata, turns_per_conversation, strict=True)
    ):
        first_endpoint = window - 1
        last_endpoint = int(n_turns) - horizon - 1
        for turn in range(first_endpoint, last_endpoint + 1):
            rows.append(
                ProbeIndex(
                    conversation_index=conversation_index,
                    conversation_id=str(record["conversation_id"]),
                    split=record["split"],
                    domain=str(record["domain"]),
                    turn=turn,
                    window=window,
                    horizon=horizon,
                )
            )
    return rows


def future_targets(
    axis_scores: torch.Tensor,
    rows: Sequence[ProbeIndex],
    *,
    safe_threshold: float,
) -> dict[str, np.ndarray]:
    """Continuous and threshold targets, excluding the current turn."""
    if axis_scores.ndim != 2:
        raise ValueError("axis_scores must have shape (conversation, turn)")
    current: list[float] = []
    future_final: list[float] = []
    future_min: list[float] = []
    for row in rows:
        scores = axis_scores[row.conversation_index]
        current_value = float(scores[row.turn])
        future = scores[row.turn + 1 : row.turn + row.horizon + 1]
        if future.numel() != row.horizon:
            raise ValueError(f"{row.conversation_id}: incomplete future target")
        current.append(current_value)
        future_final.append(float(future[-1]))
        future_min.append(float(future.min()))
    current_array = np.asarray(current, dtype=np.float32)
    final_array = np.asarray(future_final, dtype=np.float32)
    minimum_array = np.asarray(future_min, dtype=np.float32)
    return {
        "current": current_array,
        "future_final": final_array,
        "future_min": minimum_array,
        "future_delta": final_array - current_array,
        "future_breach": (minimum_array < float(safe_threshold)).astype(np.int8),
    }


def stack_current(
    activations: torch.Tensor,
    rows: Sequence[ProbeIndex],
) -> torch.Tensor:
    return torch.stack([activations[row.conversation_index, row.turn] for row in rows])


def stack_windows(
    activations: torch.Tensor,
    rows: Sequence[ProbeIndex],
) -> torch.Tensor:
    windows = []
    for row in rows:
        start = row.turn - row.window + 1
        windows.append(activations[row.conversation_index, start : row.turn + 1])
    return torch.stack(windows)


def stack_user_embeddings(
    embeddings: torch.Tensor,
    rows: Sequence[ProbeIndex],
) -> torch.Tensor:
    return torch.stack([embeddings[row.conversation_index, row.turn] for row in rows])


def normalize_axis(axis: torch.Tensor, layer: int) -> torch.Tensor:
    vector = axis[layer].float()
    return vector / vector.norm().clamp_min(1e-8)


def project_axis(
    activations: torch.Tensor,
    axis: torch.Tensor,
    *,
    layer: int,
) -> torch.Tensor:
    """Vectorized equivalent of assistant_axis.project(..., normalize=True)."""
    return activations.float() @ normalize_axis(axis, layer)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
