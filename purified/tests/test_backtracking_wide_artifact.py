"""Fail-closed contracts for reconstructing a wider C7 event artifact."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.backtracking_window_sweep.reconstruct_wide_artifact import (
    EVENT_MAP_SCHEMA,
    OFFICIAL_OFFSETS,
    STAGE_B_REPO,
    STAGE_B_REVISION,
    WIDE_OFFSETS,
    AmbiguousTraceError,
    MappingError,
    TailValidationError,
    audit_sentence_spans,
    parse_event_map,
    reconstruct_from_event_map,
    reconstruct_labeled_region,
)


def _sentence(text: str, start: int, *, label: bool = False) -> dict:
    return {
        "sentence": text,
        "char_start": start,
        "char_end": start + len(text),
        "is_backtracking": label,
    }


def _record(*sentences: dict) -> dict:
    return {
        "question_id": "q0",
        "trace_idx": 0,
        "sentences": list(sentences),
    }


def test_span_reconstruction_accepts_one_complete_consistent_region():
    record = _record(_sentence("abc", 0), _sentence("def", 3, label=True))
    audit = audit_sentence_spans(record)
    assert audit.exact_labeled_region
    assert audit.gap_chars == 0
    assert audit.conflicting_chars == 0
    assert reconstruct_labeled_region(record) == "abcdef"


def test_span_reconstruction_rejects_uncovered_characters():
    record = _record(_sentence("abc", 0), _sentence("def", 4))
    audit = audit_sentence_spans(record)
    assert audit.gap_chars == 1
    with pytest.raises(AmbiguousTraceError, match="1 uncovered"):
        reconstruct_labeled_region(record)


def test_span_reconstruction_rejects_conflicting_overlaps():
    record = _record(_sentence("abcd", 0), _sentence("XY", 2))
    audit = audit_sentence_spans(record)
    assert audit.conflicting_chars == 2
    with pytest.raises(AmbiguousTraceError, match="2 conflicting"):
        reconstruct_labeled_region(record)


def _event_payload(entries: list[dict]) -> dict:
    return {
        "schema_version": EVENT_MAP_SCHEMA,
        "source_repo": STAGE_B_REPO,
        "source_revision": STAGE_B_REVISION,
        "official_sha256": (
            "1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810"
        ),
        "entries": entries,
    }


def test_event_map_rejects_duplicate_keys_and_missing_tail():
    complete = {str(offset): [0, offset + 23] for offset in WIDE_OFFSETS}
    duplicate = [
        {"key": "q|0|0", "locations": complete},
        {"key": "q|0|0", "locations": complete},
    ]
    with pytest.raises(MappingError, match="duplicate"):
        parse_event_map(_event_payload(duplicate))

    missing = dict(complete)
    missing.pop(str(OFFICIAL_OFFSETS[0]))
    with pytest.raises(MappingError, match="official-tail coordinates missing"):
        parse_event_map(
            _event_payload([{"key": "q|0|0", "locations": missing}])
        )


def _locations(row: int, event_position: int) -> dict[int, tuple[int, int] | None]:
    return {
        offset: (
            (row, event_position + offset)
            if 0 <= event_position + offset < 32
            else None
        )
        for offset in WIDE_OFFSETS
    }


def test_explicit_map_reconstructs_every_tail_and_only_complete_wide_rows():
    source = np.arange(2 * 32 * 3, dtype=np.float16).reshape(2, 32, 3)
    keys = ["q0|0|0", "q1|1|0"]
    event_map = {
        keys[0]: _locations(0, 24),
        keys[1]: _locations(1, 20),
    }
    official_x = np.stack(
        [
            source[0, [24 + offset for offset in OFFICIAL_OFFSETS]],
            source[1, [20 + offset for offset in OFFICIAL_OFFSETS]],
        ]
    ).astype(np.float32)
    official_y = np.array([True, False])

    wide_x, wide_y, wide_keys, validation = reconstruct_from_event_map(
        source,
        official_x=official_x,
        official_y=official_y,
        official_keys=keys,
        event_map=event_map,
    )
    assert wide_x.shape == (1, 16, 3)
    np.testing.assert_array_equal(
        wide_x[0],
        source[0, [24 + offset for offset in WIDE_OFFSETS]].astype(np.float32),
    )
    np.testing.assert_array_equal(wide_y, [True])
    np.testing.assert_array_equal(wide_keys, [keys[0]])
    assert validation == {
        "official_rows_reconstructed": 2,
        "wide_rows": 1,
        "wide_rows_dropped_for_missing_early_offsets": 1,
        "trailing_six_exact_equal": True,
        "trailing_six_max_abs": 0.0,
    }


def test_explicit_map_fails_on_any_official_tail_mismatch():
    source = np.arange(32 * 3, dtype=np.float16).reshape(1, 32, 3)
    key = "q0|0|0"
    event_map = {key: _locations(0, 24)}
    official_x = source[
        0, [24 + offset for offset in OFFICIAL_OFFSETS]
    ].astype(np.float32)
    official_x[2, 1] += 1.0
    with pytest.raises(TailValidationError, match="do not exactly match"):
        reconstruct_from_event_map(
            source,
            official_x=official_x[None, ...],
            official_y=np.array([True]),
            official_keys=[key],
            event_map=event_map,
        )


def test_explicit_map_rejects_noncontiguous_cache_coordinates():
    source = np.arange(32 * 3, dtype=np.float16).reshape(1, 32, 3)
    key = "q0|0|0"
    locations = _locations(0, 24)
    locations[-10] = (0, 29)
    official_x = source[
        0, [24 + offset for offset in OFFICIAL_OFFSETS]
    ].astype(np.float32)
    with pytest.raises(MappingError, match="one contiguous cache row"):
        reconstruct_from_event_map(
            source,
            official_x=official_x[None, ...],
            official_y=np.array([True]),
            official_keys=[key],
            event_map={key: locations},
        )


def test_explicit_map_requires_exact_official_key_set():
    source = np.zeros((1, 32, 3), dtype=np.float16)
    key = "q0|0|0"
    with pytest.raises(MappingError, match="key set mismatch"):
        reconstruct_from_event_map(
            source,
            official_x=np.zeros((1, 6, 3), dtype=np.float32),
            official_y=np.array([False]),
            official_keys=[key],
            event_map={},
        )
