"""Retain protocol-compatible T=16 candidates when bitwise replay drifts.

Aniket's frozen extractor intentionally discards a row unless the trailing
six activations are bit-for-bit identical to the official artifact.  That is
the right publication gate, but it also makes recovery impossible when a
different CUDA kernel produces small floating-point drift.  This wrapper
leaves the frozen shard contract and validation unchanged while atomically
saving every otherwise-valid 16-token candidate to a separate directory.

Run it with the same CLI as ``extract_wide_teacher_force`` and set
``BACKTRACKING_CANDIDATE_OUTPUT_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np


def candidate_arrays(
    hidden: np.ndarray,
    *,
    trace: Any,
    offsets: Sequence[tuple[int, int]],
    reference: ModuleType,
    expected_width: int,
) -> dict[str, np.ndarray]:
    """Collect every source event with all requested physical offsets."""

    hidden = np.asarray(hidden)
    if hidden.ndim != 2 or hidden.shape[1] != expected_width:
        raise ValueError(
            f"captured hidden state must have shape (tokens, {expected_width}), "
            f"got {hidden.shape}"
        )
    if len(hidden) != len(offsets):
        raise ValueError("captured hidden state and offsets disagree")

    rows: list[np.ndarray] = []
    keys: list[str] = []
    labels: list[int] = []
    boundaries: list[int] = []
    for event in trace.events:
        try:
            boundary = reference.token_containing_char(
                offsets,
                event.target_char,
            )
        except reference.TraceAlignmentError:
            continue
        positions = [
            boundary + value for value in reference.ARTIFACT_OFFSETS
        ]
        if min(positions) < 0:
            continue
        rows.append(
            np.asarray(hidden[positions], dtype=reference.OUTPUT_DTYPE)
        )
        keys.append(event.key)
        labels.append(event.label)
        boundaries.append(boundary)

    shape = (0, len(reference.ARTIFACT_OFFSETS), expected_width)
    return {
        "candidate_X": (
            np.stack(rows).astype(reference.OUTPUT_DTYPE, copy=False)
            if rows
            else np.empty(shape, dtype=reference.OUTPUT_DTYPE)
        ),
        "candidate_keys": np.asarray(keys),
        "candidate_is_bt": np.asarray(labels, dtype=np.uint8),
        "candidate_boundary_token": np.asarray(
            boundaries,
            dtype=np.int32,
        ),
    }


def main() -> None:
    """Patch only candidate persistence, then delegate to the frozen CLI."""

    from experiments.backtracking_window_sweep import (
        extract_wide_teacher_force as reference,
    )

    output_dir_raw = os.environ.get("BACKTRACKING_CANDIDATE_OUTPUT_DIR")
    if not output_dir_raw:
        raise RuntimeError("set BACKTRACKING_CANDIDATE_OUTPUT_DIR")
    output_dir = Path(output_dir_raw)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_partition = reference.partition_trace_activations

    def partition_and_retain(
        hidden: np.ndarray,
        *,
        trace: Any,
        offsets: Sequence[tuple[int, int]],
        official_x_by_key: dict[str, np.ndarray],
        expected_width: int = reference.EXPECTED_WIDTH,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        candidates = candidate_arrays(
            hidden,
            trace=trace,
            offsets=offsets,
            reference=reference,
            expected_width=expected_width,
        )
        candidate_path = output_dir / f"trace_{trace.trace_idx:05d}.npz"
        reference._atomic_npz(candidate_path, **candidates)
        return original_partition(
            hidden,
            trace=trace,
            offsets=offsets,
            official_x_by_key=official_x_by_key,
            expected_width=expected_width,
        )

    reference.partition_trace_activations = partition_and_retain
    reference.main()


if __name__ == "__main__":
    main()
