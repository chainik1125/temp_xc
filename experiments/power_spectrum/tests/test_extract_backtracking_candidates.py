from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from experiments.power_spectrum.code.extract_backtracking_candidates import (
    candidate_arrays,
)


class AlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    key: str
    target_char: int
    label: int


def test_candidate_arrays_keep_only_complete_windows() -> None:
    reference = SimpleNamespace(
        ARTIFACT_OFFSETS=(-3, -2, -1),
        OUTPUT_DTYPE=np.float32,
        TraceAlignmentError=AlignmentError,
        token_containing_char=lambda offsets, target: target,
    )
    trace = SimpleNamespace(
        events=(
            Event(key="too-early", target_char=2, label=0),
            Event(key="kept", target_char=4, label=1),
        )
    )
    hidden = np.arange(24, dtype=np.float32).reshape(6, 4)
    offsets = [(index, index + 1) for index in range(6)]

    arrays = candidate_arrays(
        hidden,
        trace=trace,
        offsets=offsets,
        reference=reference,
        expected_width=4,
    )

    assert arrays["candidate_keys"].tolist() == ["kept"]
    assert arrays["candidate_is_bt"].tolist() == [1]
    assert arrays["candidate_boundary_token"].tolist() == [4]
    np.testing.assert_array_equal(
        arrays["candidate_X"],
        hidden[[1, 2, 3]][None],
    )
