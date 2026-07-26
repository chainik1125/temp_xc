"""Smoke contracts for the pinned Ward T16 teacher-force builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.backtracking_window_sweep.extract_wide_teacher_force import (
    EXCLUSION_SCHEMA,
    MANIFEST_SCHEMA,
    SHARD_SCHEMA,
    TEACHER_FORCE_PROTOCOL,
    TAIL_COMPARATOR,
    ShardValidationError,
    SourceValidationError,
    TOKENIZATION_ADD_SPECIAL_TOKENS,
    TokenizerRoundTripError,
    TraceAlignmentError,
    _atomic_npz,
    _canonical_sha256,
    _cohort_summary,
    _float32_bit_equal,
    _require_coverage,
    _trace_input_sha256,
    _validate_existing_final_output,
    _validate_trace_shard,
    partition_trace_activations,
    sha256_file,
    tokenize_full_response,
    validate_source_pin,
    validate_trace_records,
)
from experiments.backtracking_window_sweep.protocol_t16 import ARTIFACT_OFFSETS
from experiments.backtracking_window_sweep.reconstruct_wide_artifact import (
    OFFICIAL_OFFSETS,
)


class CharacterTokenizer:
    """A deterministic fast-tokenizer fixture with one BOS plus one char/token."""

    is_fast = True

    def __init__(self) -> None:
        self.text = ""
        self.add_special_tokens = None

    def __call__(self, text, **kwargs):
        self.text = text
        self.add_special_tokens = kwargs["add_special_tokens"]
        prefix_ids = [128000] if self.add_special_tokens else []
        prefix_offsets = [(0, 0)] if self.add_special_tokens else []
        return {
            "input_ids": [*prefix_ids, *range(1, len(text) + 1)],
            "offset_mapping": [
                *prefix_offsets,
                *((index, index + 1) for index in range(len(text))),
            ],
        }

    def decode(self, _input_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return self.text


def _fixture_records():
    thinking = "0123456789abcdefghijklmnopqrstuvwxyz"
    response = f"{thinking}</think>\nanswer"
    prompts = [{"id": "q0"}]
    labels = [
        {
            "question_id": "q0",
            "trace_idx": 0,
            "sentences": [
                {
                    "sentence": thinking[13:17],
                    "char_start": 13,
                    "char_end": 17,
                    "is_backtracking": False,
                },
                {
                    "sentence": thinking[25:29],
                    "char_start": 25,
                    "char_end": 29,
                    "is_backtracking": True,
                },
            ],
        }
    ]
    traces = [
        {
            "question_id": "q0",
            "trace_idx": 0,
            "category": "arithmetic",
            "full_response": response,
            "thinking_process": thinking,
        }
    ]
    keys = np.asarray(["q0|0|0", "q0|0|1"])
    labels_array = np.asarray([0, 1], dtype=np.uint8)
    return prompts, labels, traces, keys, labels_array


def _validated_trace():
    prompts, labels, traces, keys, labels_array = _fixture_records()
    specs, report = validate_trace_records(
        prompts,
        labels,
        traces,
        official_keys=keys,
        official_labels=labels_array,
        strict_counts=False,
    )
    assert report["ordered_official_key_join"]
    assert report["label_text_mismatches"] == 0
    return specs[0]


def test_trace_join_locks_question_trace_and_official_order():
    prompts, labels, traces, keys, labels_array = _fixture_records()
    del traces[0]["trace_idx"]
    specs, report = validate_trace_records(
        prompts,
        labels,
        traces,
        official_keys=keys,
        official_labels=labels_array,
        strict_counts=False,
    )
    assert specs[0].thinking_char_start == 0
    assert [event.key for event in specs[0].events] == keys.tolist()
    assert report["official_rows"] == 2

    traces[0]["question_id"] = "wrong"
    with pytest.raises(TraceAlignmentError, match="raw question_id"):
        validate_trace_records(
            prompts,
            labels,
            traces,
            official_keys=keys,
            official_labels=labels_array,
            strict_counts=False,
        )


def test_trace_join_rejects_optional_trace_index_drift():
    prompts, labels, traces, keys, labels_array = _fixture_records()
    traces[0]["trace_idx"] = 7
    with pytest.raises(TraceAlignmentError, match="optional raw trace_idx"):
        validate_trace_records(
            prompts,
            labels,
            traces,
            official_keys=keys,
            official_labels=labels_array,
            strict_counts=False,
        )


def test_trace_join_requires_exact_thinking_process_prefix():
    prompts, labels, traces, keys, labels_array = _fixture_records()
    traces[0]["thinking_process"] = "different"
    with pytest.raises(TraceAlignmentError, match="exact full_response prefix"):
        validate_trace_records(
            prompts,
            labels,
            traces,
            official_keys=keys,
            official_labels=labels_array,
            strict_counts=False,
        )


def test_trace_join_excludes_source_text_drift_before_model_loading():
    prompts, labels, traces, keys, labels_array = _fixture_records()
    labels[0]["sentences"][0]["sentence"] = "WXYZ"
    specs, report = validate_trace_records(
        prompts,
        labels,
        traces,
        official_keys=keys,
        official_labels=labels_array,
        strict_counts=False,
    )
    assert [event.key for event in specs[0].events] == ["q0|0|1"]
    assert report["source_eligible_rows"] == 1
    assert report["source_excluded_rows"] == 1
    assert report["source_exclusion_counts"] == {
        "span_out_of_bounds": 0,
        "sentence_text_mismatch": 1,
    }
    assert len(report["source_eligible_sha256"]) == 64
    assert len(report["source_exclusions_sha256"]) == 64


def test_trace_join_never_admits_a_span_from_the_answer_suffix():
    prompts, labels, traces, keys, labels_array = _fixture_records()
    thinking_length = len(traces[0]["thinking_process"])
    labels[0]["sentences"][1].update(
        {
            "sentence": "</th",
            "char_start": thinking_length,
            "char_end": thinking_length + 4,
        }
    )
    specs, report = validate_trace_records(
        prompts,
        labels,
        traces,
        official_keys=keys,
        official_labels=labels_array,
        strict_counts=False,
    )
    assert [event.key for event in specs[0].events] == ["q0|0|0"]
    assert report["source_exclusion_counts"]["span_out_of_bounds"] == 1


def test_trace_join_rejects_reordered_official_keys():
    prompts, labels, traces, keys, labels_array = _fixture_records()
    with pytest.raises(TraceAlignmentError, match="order-preserving"):
        validate_trace_records(
            prompts,
            labels,
            traces,
            official_keys=keys[::-1],
            official_labels=labels_array[::-1],
            strict_counts=False,
        )


def test_source_pin_requires_exact_hash_commit_and_relative_path(tmp_path):
    traces = tmp_path / "traces.json"
    traces.write_text("[]\n")
    digest = hashlib.sha256(traces.read_bytes()).hexdigest()
    pin = validate_source_pin(
        traces,
        expected_sha256=digest,
        source_path="results/ward/traces.json",
        source_commit="a" * 40,
    )
    assert pin.sha256 == digest
    with pytest.raises(SourceValidationError, match="SHA-256 mismatch"):
        validate_source_pin(
            traces,
            expected_sha256="b" * 64,
            source_path="results/ward/traces.json",
            source_commit="a" * 40,
        )
    with pytest.raises(SourceValidationError, match="repository-relative"):
        validate_source_pin(
            traces,
            expected_sha256=digest,
            source_path="../other-branch/traces.json",
            source_commit="a" * 40,
        )


def test_teacher_force_partition_proves_tail_and_keeps_only_full_t16():
    trace = _validated_trace()
    tokenizer = CharacterTokenizer()
    tokenized = tokenize_full_response(tokenizer, trace.full_response)
    assert TOKENIZATION_ADD_SPECIAL_TOKENS is False
    assert tokenizer.add_special_tokens is False
    hidden = np.arange(
        len(tokenized["input_ids"]) * 3,
        dtype=np.float32,
    ).reshape(-1, 3)
    boundaries = [
        next(
            index
            for index, (start, end) in enumerate(tokenized["offsets"])
            if start <= event.target_char < end
        )
        for event in trace.events
    ]
    official = {
        event.key: hidden[
            [boundary + offset for offset in OFFICIAL_OFFSETS]
        ]
        for event, boundary in zip(trace.events, boundaries, strict=True)
    }
    arrays, comparison = partition_trace_activations(
        hidden,
        trace=trace,
        offsets=tokenized["offsets"],
        official_x_by_key=official,
        expected_width=3,
    )
    assert comparison["exact_tail_rows"] == 2
    assert comparison["nonexact_tail_rows"] == 0
    assert arrays["tail_only_keys"].tolist() == ["q0|0|0"]
    assert arrays["wide_keys"].tolist() == ["q0|0|1"]
    np.testing.assert_array_equal(
        arrays["wide_X"][0],
        hidden[
            [boundaries[1] + offset for offset in ARTIFACT_OFFSETS]
        ],
    )

    official["q0|0|1"] = official["q0|0|1"].copy()
    official["q0|0|1"][0, 0] += 1
    filtered, mismatch = partition_trace_activations(
        hidden,
        trace=trace,
        offsets=tokenized["offsets"],
        official_x_by_key=official,
        expected_width=3,
    )
    assert filtered["wide_keys"].tolist() == []
    assert filtered["tail_only_keys"].tolist() == ["q0|0|0"]
    assert mismatch["exact_tail_rows"] == 1
    assert mismatch["nonexact_tail_rows"] == 1
    assert mismatch["exclusions"][0]["key"] == "q0|0|1"
    assert mismatch["exclusions"][0]["reason"] == "tail_not_bit_exact"
    assert (
        mismatch["exclusions"][0]["metrics"]["bit_mismatched_values"]
        == 1
    )


def test_float32_comparator_is_bit_exact_for_signed_zero_and_nan_payloads():
    positive_zero = np.asarray([0.0], dtype=np.float32)
    negative_zero = np.asarray([-0.0], dtype=np.float32)
    assert not _float32_bit_equal(positive_zero, negative_zero)

    first_nan = np.asarray([0x7FC00001], dtype=np.uint32).view(np.float32)
    same_nan = np.asarray([0x7FC00001], dtype=np.uint32).view(np.float32)
    other_nan = np.asarray([0x7FC00002], dtype=np.uint32).view(np.float32)
    assert _float32_bit_equal(first_nan, same_nan)
    assert not _float32_bit_equal(first_nan, other_nan)


def test_decoded_text_mismatch_has_a_dedicated_trace_exclusion_error():
    class DriftTokenizer(CharacterTokenizer):
        def decode(self, _input_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return self.text + "x"

    with pytest.raises(TokenizerRoundTripError, match="round-trip"):
        tokenize_full_response(DriftTokenizer(), "abc")


def test_coverage_gate_requires_both_classes_and_all_category_cells():
    official = _cohort_summary(
        ["a", "b", "c", "d"],
        [0, 1, 0, 1],
        ["arithmetic", "arithmetic", "geometry", "geometry"],
    )
    valid = _cohort_summary(
        ["a", "b", "c", "d"],
        [0, 1, 0, 1],
        ["arithmetic", "arithmetic", "geometry", "geometry"],
    )
    report = _require_coverage(
        {"source": valid, "exact": valid, "wide": valid},
        official_summary=official,
    )
    assert all(gate["passed"] for gate in report.values())

    missing_cell = _cohort_summary(
        ["a", "b", "c"],
        [0, 1, 0],
        ["arithmetic", "arithmetic", "geometry"],
    )
    with pytest.raises(ShardValidationError, match="coverage gate"):
        _require_coverage(
            {"source": valid, "exact": valid, "wide": missing_cell},
            official_summary=official,
        )


def test_resume_sidecar_is_fail_closed(tmp_path):
    trace = _validated_trace()
    tensor = tmp_path / "trace_00000.npz"
    sidecar_path = tensor.with_suffix(".manifest.json")
    exclusion_path = tmp_path / "trace_00000.exclusions.json"
    arrays = {
        "wide_X": np.zeros((1, 16, 3), dtype=np.float32),
        "wide_keys": np.asarray(["q0|0|1"]),
        "wide_is_bt": np.asarray([1], dtype=np.uint8),
        "wide_boundary_token": np.asarray([29], dtype=np.int32),
        "tail_only_X": np.zeros((1, 6, 3), dtype=np.float32),
        "tail_only_keys": np.asarray(["q0|0|0"]),
        "tail_only_is_bt": np.asarray([0], dtype=np.uint8),
        "tail_only_boundary_token": np.asarray([15], dtype=np.int32),
    }
    _atomic_npz(tensor, **arrays)
    request_sha = "d" * 64
    entries = []
    exclusion_document = {
        "schema_version": EXCLUSION_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "request_sha256": request_sha,
        "question_id": trace.question_id,
        "trace_idx": trace.trace_idx,
        "category": trace.category,
        "trace_input_sha256": _trace_input_sha256(trace),
        "trace_exclusion_reason": None,
        "entries": entries,
        "entries_sha256": _canonical_sha256({"entries": entries}),
        "reason_counts": {},
    }
    exclusion_path.write_text(json.dumps(exclusion_document))
    sidecar = {
        "schema_version": SHARD_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "request_sha256": request_sha,
        "question_id": trace.question_id,
        "trace_idx": trace.trace_idx,
        "category": trace.category,
        "full_response_sha256": trace.full_response_sha256,
        "trace_input_sha256": _trace_input_sha256(trace),
        "tail_comparator": TAIL_COMPARATOR,
        "tail_exact_for_retained_events": True,
        "source_eligible_rows": 2,
        "exact_tail_rows": 2,
        "excluded_rows": 0,
        "exact_tail_cohort": _cohort_summary(
            [event.key for event in trace.events],
            [event.label for event in trace.events],
            [trace.category] * len(trace.events),
        ),
        "exclusion_manifest_sha256": sha256_file(exclusion_path),
        "wide_rows": 1,
        "tail_only_rows": 1,
        "sha256": sha256_file(tensor),
    }
    sidecar_path.write_text(json.dumps(sidecar))
    assert _validate_trace_shard(
        tensor,
        sidecar_path,
        exclusion_path,
        trace=trace,
        request_sha256=request_sha,
        expected_width=3,
    )

    sidecar["full_response_sha256"] = "e" * 64
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ShardValidationError, match="sidecar drifted"):
        _validate_trace_shard(
            tensor,
            sidecar_path,
            exclusion_path,
            trace=trace,
            request_sha256=request_sha,
            expected_width=3,
        )


def test_final_resume_rejects_edited_early_offset_and_updated_manifest(
    tmp_path,
):
    trace = _validated_trace()
    output_dir = tmp_path / "extraction"
    tensor_path = output_dir / "shards" / "trace_00000.npz"
    original_x = np.arange(16 * 3, dtype=np.float32).reshape(1, 16, 3)
    _atomic_npz(
        tensor_path,
        wide_X=original_x,
        wide_keys=np.asarray(["q0|0|1"]),
    )
    artifact_path = tmp_path / "sentence_acts_L10_T16.npz"
    manifest_path = tmp_path / "sentence_acts_L10_T16.manifest.json"
    _atomic_npz(
        artifact_path,
        X=original_x,
        is_bt=np.asarray([1], dtype=np.uint8),
        keys=np.asarray(["q0|0|1"]),
        offsets=np.asarray(ARTIFACT_OFFSETS, dtype=np.int32),
    )
    request_sha = "d" * 64
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "status": "complete",
        "request_sha256": request_sha,
        "output": {"sha256": sha256_file(artifact_path)},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    observed, observed_sha = _validate_existing_final_output(
        artifact_path,
        manifest_path,
        output_dir=output_dir,
        traces=[trace],
        request_sha256=request_sha,
        expected_keys=["q0|0|1"],
        expected_labels=[1],
        expected_width=3,
    )
    assert observed == manifest
    assert observed_sha == manifest["output"]["sha256"]

    edited_x = original_x.copy()
    edited_x[0, 0, 0] += 1
    _atomic_npz(
        artifact_path,
        X=edited_x,
        is_bt=np.asarray([1], dtype=np.uint8),
        keys=np.asarray(["q0|0|1"]),
        offsets=np.asarray(ARTIFACT_OFFSETS, dtype=np.int32),
    )
    manifest["output"]["sha256"] = sha256_file(artifact_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert manifest["output"]["sha256"] == sha256_file(artifact_path)
    with pytest.raises(
        ShardValidationError,
        match="activations drifted from trace shards",
    ):
        _validate_existing_final_output(
            artifact_path,
            manifest_path,
            output_dir=output_dir,
            traces=[trace],
            request_sha256=request_sha,
            expected_keys=["q0|0|1"],
            expected_labels=[1],
            expected_width=3,
        )


def test_tmux_launcher_is_branch_gated():
    launcher = (
        Path(__file__).parents[1]
        / "experiments/backtracking_window_sweep/launch_teacher_force_tmux.sh"
    ).read_text()
    worker = (
        Path(__file__).parents[1]
        / "experiments/backtracking_window_sweep/run_teacher_force_runpod.sh"
    ).read_text()
    assert '!= "neurips-aniket"' in launcher
    assert '!= "neurips-aniket"' in worker
    assert "tmux new-session -d" in launcher
