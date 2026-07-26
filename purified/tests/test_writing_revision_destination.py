from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from experiments.writing_revision_destination.evaluate_activations import (
    ActivationDataset,
    activation_view,
    evaluate_sweep as evaluate_activation_sweep,
    select_hidden_coordinates,
    stable_permutation,
)
from experiments.writing_revision_destination.extract_activations import (
    ExtractionConfig,
    _forward_batch,
    _ensure_shard_sidecar,
    build_padded_batch,
    extraction_invariance_diagnostics,
    padding_invariance_diagnostics,
    shard_tensors,
    singleton_repeatability_diagnostics,
    slice_final_windows,
    validate_shard_tensors,
    validate_token_cohort,
)
from experiments.writing_revision_destination.klicke import (
    LogRow,
    RevisionEvent,
    deduplicate_windows,
    extract_writer_events,
    normalize_logged_change,
)
from experiments.writing_revision_destination.report import render_publication
from experiments.writing_revision_destination.token_audit import (
    TokenRevisionEvent,
    align_revision_event,
    cohort_records,
    cohort_sha256,
    deduplicate_token_windows,
    evaluate_target_sweep,
    write_cohort,
)


class FakeDeletionTokenizer:
    """Deterministic whitespace tokenizer for exact-prefix unit tests."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def _id(self, token: str) -> int:
        if token not in self._ids:
            self._ids[token] = len(self._ids) + 1
        return self._ids[token]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list[int]]:
        assert not add_special_tokens
        assert not return_offsets_mapping
        return {"input_ids": [self._id(token) for token in text.split()]}


class BoundaryRetokenizingTokenizer(FakeDeletionTokenizer):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list[int]]:
        encoded = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )
        if text == "one two three":
            encoded["input_ids"][-1] = 10_000
        return encoded


class BosDeletionTokenizer(FakeDeletionTokenizer):
    bos_token_id = 128_000

    def build_inputs_with_special_tokens(
        self,
        token_ids: list[int],
    ) -> list[int]:
        return [self.bos_token_id, *token_ids]


class RecordingForwardModel:
    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        width_bias: float = 0.0,
        call_bias: float = 0.0,
    ) -> None:
        self.layer = layer
        self.width_bias = width_bias
        self.call_bias = call_bias
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool,
    ) -> object:
        assert use_cache is False
        self.calls.append(
            {
                "input_ids": input_ids.detach().cpu(),
                "attention_mask": attention_mask.detach().cpu(),
                "position_ids": position_ids.detach().cpu(),
            }
        )
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
        hidden = hidden + self.width_bias * input_ids.shape[1]
        hidden = hidden + self.call_bias * len(self.calls)
        self.layer(hidden)
        return object()


def _input(row: int, text: str, cursor: int) -> LogRow:
    return LogRow(
        row_index=row,
        activity="Paste",
        cursor_after=cursor,
        text_change=text,
        pause_ms=0.0,
        down_event="v",
    )


def _backspace(row: int, cursor: int, character: str) -> LogRow:
    return LogRow(
        row_index=row,
        activity="Remove/Cut",
        cursor_after=cursor,
        text_change=character,
        pause_ms=800.0 if row == 1 else 0.0,
        down_event="Backspace",
    )


def _revision_event(
    *,
    preburst: str,
    postburst: str,
    writer: str = "writer-a",
    row_index: int = 1,
    deleted_words: int = 3,
) -> RevisionEvent:
    lexical_label = min(deleted_words, 5)
    return RevisionEvent(
        event_hash=f"lexical-{writer}-{row_index}",
        window_hash="lexical-window",
        writer_id=writer,
        row_index=row_index,
        words=("one", "two", "three", "four", "five"),
        label=lexical_label,
        deleted_words=deleted_words,
        pause_ms=100.0,
        prefix_word_count=len(preburst.split()),
        remove_actions=deleted_words,
        single_character_backspaces=True,
        preburst_text=preburst,
        postburst_text=postburst,
    )


def _token_event(
    *,
    event_hash: str,
    writer_hash: str,
    window: tuple[int, ...],
    token_distance: int,
    lexical_label: int,
) -> TokenRevisionEvent:
    return TokenRevisionEvent(
        event_hash=event_hash,
        writer_hash=writer_hash,
        window_hash=f"window-{window}",
        input_ids=window,
        window_token_ids=window,
        token_distance=token_distance,
        capped_token_label=min(token_distance, 6),
        lexical_deleted_words=lexical_label,
        lexical_label=lexical_label,
        prefix_token_count=len(window),
        special_tokens_added=0,
        remove_actions=lexical_label,
        single_character_backspaces=True,
    )


def test_logger_escape_normalization_is_explicit() -> None:
    assert normalize_logged_change(r"\n") == "\n"
    assert normalize_logged_change(r"\"") == '"'
    assert normalize_logged_change(r"\\") == "\\"
    assert normalize_logged_change("word") == "word"


def test_trailing_deletion_is_labeled_from_strict_preburst_text() -> None:
    initial = "one two three four five"
    removed = "four five"
    rows = [_input(0, initial, len(initial))]
    current = initial
    for row_index, character in enumerate(reversed(removed), start=1):
        current = current[:-1]
        rows.append(
            _backspace(
                row_index,
                len(current),
                character,
            )
        )

    events, diagnostics = extract_writer_events(
        "writer-a",
        rows,
        window=5,
    )

    assert diagnostics["eligible_events"] == 1
    assert len(events) == 1
    assert events[0].words == ("one", "two", "three", "four", "five")
    assert events[0].label == 2
    assert events[0].deleted_words == 2
    assert events[0].single_character_backspaces is True
    assert events[0].pause_ms == 800.0


def test_partial_word_deletion_is_rejected() -> None:
    initial = "one two three four five"
    removed = "ur five"
    rows = [_input(0, initial, len(initial))]
    current = initial
    for row_index, character in enumerate(reversed(removed), start=1):
        current = current[:-1]
        rows.append(_backspace(row_index, len(current), character))

    events, diagnostics = extract_writer_events(
        "writer-a",
        rows,
        window=5,
    )

    assert events == []
    assert diagnostics["ineligible_trailing_bursts"] == 1


def test_complex_activity_truncates_after_preserving_prior_event() -> None:
    initial = "one two three four five"
    removed = "four five"
    rows = [_input(0, initial, len(initial))]
    current = initial
    for row_index, character in enumerate(reversed(removed), start=1):
        current = current[:-1]
        rows.append(_backspace(row_index, len(current), character))
    rows.append(
        LogRow(
            row_index=len(rows),
            activity="Replace",
            cursor_after=0,
            text_change="opaque",
            pause_ms=0.0,
            down_event="x",
        )
    )

    events, diagnostics = extract_writer_events(
        "writer-a",
        rows,
        window=5,
    )

    assert len(events) == 1
    assert diagnostics["truncated_complex_activity"] == 1


def test_exact_window_dedup_drops_conflicts_and_same_label_duplicates() -> None:
    base = RevisionEvent(
        event_hash="b",
        window_hash="same",
        writer_id="one",
        row_index=1,
        words=("a", "b", "c", "d", "e"),
        label=2,
        deleted_words=2,
        pause_ms=0.0,
        prefix_word_count=5,
        remove_actions=2,
        single_character_backspaces=True,
    )
    duplicate = replace(base, event_hash="a", writer_id="two")
    conflict_a = replace(
        base,
        event_hash="c",
        window_hash="conflict",
        label=3,
    )
    conflict_b = replace(
        conflict_a,
        event_hash="d",
        writer_id="three",
        label=4,
    )

    retained, diagnostics = deduplicate_windows(
        [base, duplicate, conflict_a, conflict_b]
    )

    assert [event.event_hash for event in retained] == ["a"]
    assert diagnostics["same_label_duplicate_rows_dropped"] == 1
    assert diagnostics["conflicting_exact_window_rows_dropped"] == 2


def test_model_token_alignment_uses_exact_pre_post_prefix() -> None:
    source = _revision_event(
        preburst="one two three four five six",
        postburst="one two three",
        deleted_words=3,
    )

    aligned, reason = align_revision_event(
        FakeDeletionTokenizer(),
        source,
        history_tokens=5,
        token_cap=2,
        max_model_tokens=32,
    )

    assert reason == "aligned"
    assert aligned is not None
    assert aligned.token_distance == 3
    assert aligned.capped_token_label == 2
    assert aligned.lexical_label == 3
    assert aligned.window_token_ids == aligned.input_ids[-5:]
    assert aligned.writer_hash != source.writer_id
    assert source.writer_id not in aligned.event_hash


def test_model_token_alignment_rejects_boundary_retokenization() -> None:
    source = _revision_event(
        preburst="one two three four five",
        postburst="one two three",
        deleted_words=2,
    )

    aligned, reason = align_revision_event(
        BoundaryRetokenizingTokenizer(),
        source,
        history_tokens=4,
        max_model_tokens=32,
    )

    assert aligned is None
    assert reason == "boundary_retokenized"


def test_model_token_cohort_includes_paper_style_bos_without_moving_anchor() -> None:
    source = _revision_event(
        preburst="one two three four five",
        postburst="one two three",
        deleted_words=2,
    )

    aligned, reason = align_revision_event(
        BosDeletionTokenizer(),
        source,
        history_tokens=4,
        max_model_tokens=8,
    )

    assert reason == "aligned"
    assert aligned is not None
    assert aligned.input_ids[0] == BosDeletionTokenizer.bos_token_id
    assert aligned.special_tokens_added == 1
    assert aligned.prefix_token_count == 6
    assert aligned.input_ids[-4:] == aligned.window_token_ids


def test_token_window_dedup_is_global_across_both_targets() -> None:
    base = _token_event(
        event_hash="b",
        writer_hash="writer-1",
        window=(1, 2, 3),
        token_distance=2,
        lexical_label=2,
    )
    duplicate = replace(base, event_hash="a", writer_hash="writer-2")
    token_conflict = replace(
        base,
        event_hash="c",
        window_token_ids=(4, 5, 6),
        token_distance=3,
        capped_token_label=3,
    )
    token_conflict_pair = replace(
        token_conflict,
        event_hash="d",
        writer_hash="writer-3",
        token_distance=4,
        capped_token_label=4,
    )
    lexical_conflict = replace(
        base,
        event_hash="e",
        window_token_ids=(7, 8, 9),
    )
    lexical_conflict_pair = replace(
        lexical_conflict,
        event_hash="f",
        writer_hash="writer-4",
        lexical_deleted_words=3,
        lexical_label=3,
    )

    retained, diagnostics = deduplicate_token_windows(
        [
            base,
            duplicate,
            token_conflict,
            token_conflict_pair,
            lexical_conflict,
            lexical_conflict_pair,
        ]
    )

    assert [event.event_hash for event in retained] == ["a"]
    assert diagnostics["same_target_duplicate_rows_dropped"] == 1
    assert diagnostics["conflicting_target_rows_dropped"] == 4


def test_token_cohort_records_never_include_text_or_writer_ids() -> None:
    event = _token_event(
        event_hash="event",
        writer_hash="writer-hash",
        window=(1, 2, 3),
        token_distance=2,
        lexical_label=2,
    )

    records = cohort_records([event])

    assert records[0]["writer_hash"] == "writer-hash"
    assert records[0]["input_ids"] == [1, 2, 3]
    assert records[0]["window_token_ids"] == [1, 2, 3]
    assert not {
        "writer_id",
        "preburst_text",
        "postburst_text",
        "words",
        "text",
    }.intersection(records[0])


def test_token_sweep_uses_observed_labels_and_writer_grouped_controls() -> None:
    events = []
    for index in range(12):
        low = index % 2 == 0
        events.append(
            _token_event(
                event_hash=f"event-{index:02d}",
                writer_hash=f"writer-{index:02d}",
                window=(11, 12, 99) if low else (12, 11, 99),
                token_distance=2 if low else 9,
                lexical_label=2 if low else 5,
            )
        )

    token_result = evaluate_target_sweep(
        events,
        target="capped_token_distance",
        max_history_tokens=3,
        c_value=1.0,
        folds=3,
        bootstrap_samples=0,
    )
    lexical_result = evaluate_target_sweep(
        events,
        target="lexical_destination",
        max_history_tokens=3,
        c_value=1.0,
        folds=3,
        bootstrap_samples=0,
    )

    assert token_result["status"] == "complete"
    assert token_result["labels_observed"] == [2, 6]
    assert lexical_result["labels_observed"] == [2, 5]
    assert set(token_result["fixed_cohort_window_sweep"]) == {"1", "2", "3"}
    full = token_result["fixed_cohort_window_sweep"]["3"]
    assert full["folds_effective"] == 3
    assert set(full["metrics"]) == {
        "prior",
        "endpoint",
        "bag",
        "canonical",
        "ordered",
        "reverse",
    }
    assert all(
        np.isfinite(metrics["log_loss"])
        for metrics in full["metrics"].values()
    )


def test_token_cohort_manifest_and_final_window_are_revalidated(
    tmp_path: Path,
) -> None:
    events = []
    for index, window in enumerate(((1, 2, 3), (4, 5, 6))):
        window_hash = hashlib.sha256(
            ",".join(str(value) for value in window).encode()
        ).hexdigest()
        events.append(
            TokenRevisionEvent(
                event_hash=hashlib.sha256(f"event-{index}".encode()).hexdigest(),
                writer_hash=hashlib.sha256(
                    f"writer-{index}".encode()
                ).hexdigest(),
                window_hash=window_hash,
                input_ids=(9, *window),
                window_token_ids=window,
                token_distance=2 + index,
                capped_token_label=2 + index,
                lexical_deleted_words=2 + index,
                lexical_label=2 + index,
                prefix_token_count=4,
                special_tokens_added=0,
                remove_actions=2 + index,
                single_character_backspaces=True,
            )
        )
    cohort_path = tmp_path / "cohort.parquet"
    manifest_path = tmp_path / "manifest.json"
    write_cohort(events, cohort_path)
    manifest = {
        "protocol": {
            "version": (
                "klicke-trailing-deletion-model-token-v1"
            ),
            "tokenizer": "NousResearch/Meta-Llama-3.1-8B",
            "history_tokens": 3,
            "token_cap": 6,
            "max_model_tokens": 8,
        },
        "counts": {"retained_events": 2},
        "cohort": {
            "sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest()
        },
        "cohort_sha256": cohort_sha256(events),
    }
    manifest_path.write_text(json.dumps(manifest))

    frame, loaded_manifest = validate_token_cohort(
        cohort_path,
        manifest_path,
        ExtractionConfig(window_tokens=3, max_model_tokens=8),
    )

    assert len(frame) == 2
    assert frame.iloc[0]["input_ids"][-3:] == frame.iloc[0][
        "window_token_ids"
    ]
    assert loaded_manifest["cohort_sha256"] == cohort_sha256(events)

    manifest["cohort_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="semantic fingerprint"):
        validate_token_cohort(
            cohort_path,
            manifest_path,
            ExtractionConfig(window_tokens=3, max_model_tokens=8),
        )


def test_final_activation_slicing_ignores_right_padding() -> None:
    hidden = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6, 1)

    result = slice_final_windows(hidden, [4, 6], window_tokens=3)

    assert result.dtype == torch.float16
    assert result[:, :, 0].tolist() == [[1.0, 2.0, 3.0], [9.0, 10.0, 11.0]]


def test_forward_batch_passes_explicit_real_token_positions() -> None:
    layer = torch.nn.Identity()
    model = RecordingForwardModel(layer)
    config = ExtractionConfig(
        window_tokens=2,
        max_model_tokens=8,
        device="cpu",
    )

    input_ids, mask, positions, lengths = build_padded_batch(
        [[5, 6, 7], [8, 9]],
        pad_token_id=0,
        device="cpu",
        forced_width=4,
    )
    output = _forward_batch(
        model,
        layer,
        [[5, 6, 7], [8, 9]],
        config,
        pad_token_id=0,
        forced_width=4,
        output_dtype=torch.float32,
    )

    assert lengths == [3, 2]
    assert input_ids.tolist() == [[5, 6, 7, 0], [8, 9, 0, 0]]
    assert mask.tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]
    assert positions.tolist() == [[0, 1, 2, 0], [0, 1, 0, 0]]
    assert model.calls[-1]["position_ids"].tolist() == positions.tolist()
    assert output[:, :, 0].tolist() == [[6.0, 7.0], [8.0, 9.0]]


def test_padding_diagnostic_localizes_padding_path_drift() -> None:
    layer = torch.nn.Identity()
    model = RecordingForwardModel(layer, width_bias=1.0)
    config = ExtractionConfig(
        window_tokens=2,
        max_model_tokens=8,
        device="cpu",
    )

    diagnostic = padding_invariance_diagnostics(
        model,
        layer,
        [[5, 6], [8, 9, 10, 11]],
        config,
        pad_token_id=0,
    )

    assert diagnostic["status"] == "failed"
    shortest = diagnostic["rows"][0]
    assert shortest["dominant_component"] == "padding_path"
    assert (
        shortest["padding_path_padded_single_vs_unpadded"]["relative_l2"]
        > 0.01
    )
    assert (
        shortest["total_batched_vs_unpadded"]["max_abs"] > 0
    )
    assert len(
        shortest["total_batched_vs_unpadded"]["per_offset"]
    ) == 2


def test_singleton_diagnostic_uses_unpadded_repeat_for_each_row() -> None:
    layer = torch.nn.Identity()
    model = RecordingForwardModel(layer)
    config = ExtractionConfig(
        window_tokens=2,
        max_model_tokens=8,
        batch_size=1,
        device="cpu",
    )

    diagnostic = extraction_invariance_diagnostics(
        model,
        layer,
        [[5, 6], [8, 9, 10, 11]],
        config,
        pad_token_id=0,
    )

    assert diagnostic["status"] == "passed"
    assert diagnostic["mode"] == "singleton_repeatability"
    assert len(model.calls) == 4
    assert [call["input_ids"].shape[0] for call in model.calls] == [1, 1, 1, 1]
    assert [call["input_ids"].shape[1] for call in model.calls] == [2, 2, 4, 4]
    assert all(
        record["padding_tokens"] == 0
        and record["exact_equal"]
        and record["repeat_second_vs_first"]["max_abs"] == 0
        for record in diagnostic["rows"]
    )


def test_singleton_diagnostic_fails_closed_on_repeat_drift() -> None:
    layer = torch.nn.Identity()
    model = RecordingForwardModel(layer, call_bias=1.0)
    config = ExtractionConfig(
        window_tokens=2,
        max_model_tokens=8,
        batch_size=1,
        device="cpu",
    )

    diagnostic = singleton_repeatability_diagnostics(
        model,
        layer,
        [[5, 6], [8, 9, 10, 11]],
        config,
        pad_token_id=0,
    )

    assert diagnostic["status"] == "failed"
    assert not diagnostic["rows"][0]["exact_equal"]
    assert diagnostic["rows"][0]["repeat_second_vs_first"]["max_abs"] > 0


def test_activation_shard_validation_binds_metadata_to_rows() -> None:
    frame = pd.DataFrame(
        {
            "event_hash": [hashlib.sha256(b"event").hexdigest()],
            "writer_hash": [hashlib.sha256(b"writer").hexdigest()],
            "window_hash": [hashlib.sha256(b"window").hexdigest()],
            "window_token_ids": [(1, 2, 3)],
            "token_distance": [2],
            "capped_token_label": [2],
            "lexical_deleted_words": [2],
            "lexical_label": [2],
            "prefix_token_count": [5],
            "special_tokens_added": [1],
            "remove_actions": [2],
            "single_character_backspaces": [True],
        }
    )
    activations = torch.arange(12, dtype=torch.float16).reshape(1, 3, 4)
    tensors = shard_tensors(
        frame,
        activations,
        start=7,
        window_tokens=3,
    )

    assert (
        validate_shard_tensors(
            tensors,
            frame,
            start=7,
            window_tokens=3,
            hidden_size=4,
        )
        == 4
    )

    tensors["capped_token_label"] = torch.tensor([3], dtype=torch.int16)
    with pytest.raises(ValueError, match="capped_token_label"):
        validate_shard_tensors(
            tensors,
            frame,
            start=7,
            window_tokens=3,
            hidden_size=4,
        )


def test_activation_shard_sidecar_rejects_tensor_drift(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "event_hash": [hashlib.sha256(b"event").hexdigest()],
        }
    )
    path = tmp_path / "rows_000000_000001.safetensors"
    save_file({"activations": torch.zeros(1, 3, 4)}, str(path))
    sidecar_path, _record = _ensure_shard_sidecar(
        path,
        frame,
        start=0,
        end=1,
        request_sha256="1" * 64,
        shape=(1, 3, 4),
    )
    assert sidecar_path.exists()

    save_file({"activations": torch.ones(1, 3, 4)}, str(path))
    with pytest.raises(ValueError, match="sidecar drifted"):
        _ensure_shard_sidecar(
            path,
            frame,
            start=0,
            end=1,
            request_sha256="1" * 64,
            shape=(1, 3, 4),
        )


def test_activation_views_and_shuffle_are_temporally_explicit() -> None:
    windows = np.arange(1 * 4 * 2, dtype=np.float32).reshape(1, 4, 2)

    np.testing.assert_array_equal(
        activation_view(windows, "endpoint"),
        windows[:, -1],
    )
    np.testing.assert_array_equal(
        activation_view(windows, "first_difference"),
        np.column_stack([windows[:, -1], windows[:, -1] - windows[:, -2]]),
    )
    residual = (
        windows[:, -1]
        + (2.0 / 3.0) * windows[:, -4]
        - (1.0 / 3.0) * windows[:, -3]
        - (4.0 / 3.0) * windows[:, -2]
    )
    np.testing.assert_array_equal(
        activation_view(windows, "trajectory_residual"),
        np.column_stack([windows[:, -1], residual]),
    )
    first = stable_permutation("event", 3, seed=7)
    second = stable_permutation("event", 3, seed=7)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, np.arange(3))
    np.testing.assert_array_equal(
        stable_permutation("event", 1, seed=7),
        [0],
    )


def test_hidden_coordinate_selection_uses_outer_training_signal() -> None:
    activations = np.zeros((8, 2, 4), dtype=np.float16)
    target = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    activations[target == 1, :, 2] = 10

    selected = select_hidden_coordinates(
        activations,
        target,
        np.arange(8),
        dimensions=1,
    )

    assert selected.tolist() == [2]


def test_raw_activation_sweep_has_fixed_and_retrained_order_controls() -> None:
    rows = 24
    activations = np.zeros((rows, 4, 6), dtype=np.float16)
    target = np.asarray([2 if index % 2 == 0 else 6 for index in range(rows)])
    for index, label in enumerate(target):
        activations[index, :, 0] = (
            [0, 3, 1, 2] if label == 2 else [0, 1, 3, 2]
        )
        activations[index, :, 1] = float(label)
    dataset = ActivationDataset(
        activations=activations,
        target=target,
        groups=np.asarray([f"writer-{index}" for index in range(rows)]),
        event_hashes=np.asarray(
            [
                hashlib.sha256(f"event-{index}".encode()).hexdigest()
                for index in range(rows)
            ]
        ),
        target_name="capped_token_label",
        provenance={"test": True},
    )

    result = evaluate_activation_sweep(
        dataset,
        window_sizes=[1, 2, 3, 4],
        projection_dimensions=2,
        outer_folds=3,
        inner_folds=2,
        c_value=1.0,
        max_iter=300,
        bootstrap_draws=0,
    )

    assert result["labels_observed"] == [2, 6]
    assert set(result["results"]) == {"1", "2", "3", "4"}
    full_metrics = result["results"]["4"]["metrics"]
    assert {
        "endpoint",
        "best_offset",
        "invariant_mean_std_max",
        "first_difference",
        "second_difference",
        "trajectory_residual",
        "ordered",
        "ordered_fixed_reverse",
        "ordered_fixed_shuffle",
        "ordered_retrained_shuffle",
    }.issubset(full_metrics)
    assert all(
        np.isfinite(values["log_loss"])
        for values in full_metrics.values()
    )


def _raw_activation_report_payload() -> dict:
    results = {}
    control_names = (
        "best_offset",
        "endpoint",
        "invariant_mean_std_max",
        "ordered_retrained_shuffle",
    )
    for window in range(1, 11):
        ordered_loss = 1.5 - 0.01 * window
        metrics = {
            "ordered": {
                "log_loss": ordered_loss,
                "balanced_accuracy": 0.3 + 0.01 * window,
            }
        }
        bootstrap = {}
        for index, name in enumerate(control_names, start=1):
            gap = 0.01 * index * max(window - 1, 0)
            metrics[name] = {
                "log_loss": ordered_loss + gap,
                "balanced_accuracy": 0.3,
            }
            bootstrap[f"{name}_minus_ordered"] = {
                "equal_writer_mean_log_loss_difference": gap,
                "ci95_lower": gap - 0.005,
                "ci95_upper": gap + 0.005,
                "writers_positive": 6,
                "writers_total": 10,
            }
        if window >= 3:
            gap = 0.02
            metrics["second_difference"] = {
                "log_loss": ordered_loss + gap,
                "balanced_accuracy": 0.31,
            }
            bootstrap["second_difference_minus_ordered"] = {
                "equal_writer_mean_log_loss_difference": gap,
                "ci95_lower": 0.01,
                "ci95_upper": 0.03,
                "writers_positive": 6,
                "writers_total": 10,
            }
        results[str(window)] = {
            "window_tokens": window,
            "metrics": metrics,
            "equal_writer_bootstrap": bootstrap,
        }
    return {
        "protocol_version": "klicke-deletion-raw-activation-gate-v1",
        "target": "capped_token_label",
        "rows": 20,
        "writers": 10,
        "configuration": {"window_sizes": list(range(1, 11))},
        "results": results,
    }


def test_raw_activation_publication_package_is_complete(tmp_path: Path) -> None:
    input_path = tmp_path / "result.json"
    input_path.write_text(json.dumps(_raw_activation_report_payload()))
    output_dir = tmp_path / "publication"

    summaries = render_publication([input_path], output_dir)

    assert summaries[0]["best_ordered"]["window_tokens"] == 10
    for suffix in (
        "png",
        "pdf",
        "csv",
        "md",
        "summary.json",
    ):
        assert (output_dir / f"token_distance.{suffix}").exists()
    markdown = (output_dir / "token_distance.md").read_text()
    assert "Positive paired gaps mean the control is worse" in markdown
    assert "equal-writer" in markdown.lower()


def test_raw_activation_publication_rejects_protocol_drift(
    tmp_path: Path,
) -> None:
    payload = _raw_activation_report_payload()
    payload["protocol_version"] = "unreviewed-protocol"
    input_path = tmp_path / "result.json"
    input_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="expected protocol"):
        render_publication([input_path], tmp_path / "publication")
