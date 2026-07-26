"""Smoke contracts for the pinned Ward T16 teacher-force builder."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from experiments.backtracking_window_sweep.extract_wide_teacher_force import (
    SHARD_SCHEMA,
    TEACHER_FORCE_PROTOCOL,
    ExactTailError,
    ShardValidationError,
    SourceValidationError,
    TraceAlignmentError,
    _atomic_npz,
    _trace_input_sha256,
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

    def __call__(self, text, **_kwargs):
        self.text = text
        return {
            "input_ids": [128000, *range(1, len(text) + 1)],
            "offset_mapping": [(0, 0)]
            + [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, _input_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return self.text


def _fixture_records():
    thinking = "0123456789abcdefghijklmnopqrstuvwxyz"
    response = f"<think>\n{thinking}</think>\nanswer"
    prompts = [{"id": "q0"}]
    labels = [
        {
            "question_id": "q0",
            "trace_idx": 0,
            "sentences": [
                {
                    "sentence": thinking[6:10],
                    "char_start": 6,
                    "char_end": 10,
                    "is_backtracking": False,
                },
                {
                    "sentence": thinking[20:24],
                    "char_start": 20,
                    "char_end": 24,
                    "is_backtracking": True,
                },
            ],
        }
    ]
    traces = [
        {
            "question_id": "q0",
            "trace_idx": 0,
            "full_response": response,
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
    specs, report = validate_trace_records(
        prompts,
        labels,
        traces,
        official_keys=keys,
        official_labels=labels_array,
        strict_counts=False,
    )
    assert specs[0].thinking_char_start == len("<think>\n")
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
    tokenized = tokenize_full_response(CharacterTokenizer(), trace.full_response)
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
    arrays = partition_trace_activations(
        hidden,
        trace=trace,
        offsets=tokenized["offsets"],
        official_x_by_key=official,
        expected_width=3,
    )
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
    with pytest.raises(ExactTailError, match="trailing six differ"):
        partition_trace_activations(
            hidden,
            trace=trace,
            offsets=tokenized["offsets"],
            official_x_by_key=official,
            expected_width=3,
        )


def test_resume_sidecar_is_fail_closed(tmp_path):
    trace = _validated_trace()
    tensor = tmp_path / "trace_00000.npz"
    sidecar_path = tensor.with_suffix(".manifest.json")
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
    sidecar = {
        "schema_version": SHARD_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "request_sha256": request_sha,
        "question_id": trace.question_id,
        "trace_idx": trace.trace_idx,
        "full_response_sha256": trace.full_response_sha256,
        "trace_input_sha256": _trace_input_sha256(trace),
        "tail_exact_against_official": True,
        "wide_rows": 1,
        "tail_only_rows": 1,
        "official_rows_compared": 2,
        "sha256": sha256_file(tensor),
    }
    sidecar_path.write_text(json.dumps(sidecar))
    assert _validate_trace_shard(
        tensor,
        sidecar_path,
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
            trace=trace,
            request_sha256=request_sha,
            expected_width=3,
        )


def test_tmux_launcher_is_branch_gated():
    launcher = (
        __import__("pathlib").Path(__file__).parents[1]
        / "experiments/backtracking_window_sweep/launch_teacher_force_tmux.sh"
    ).read_text()
    worker = (
        __import__("pathlib").Path(__file__).parents[1]
        / "experiments/backtracking_window_sweep/run_teacher_force_runpod.sh"
    ).read_text()
    assert '!= "neurips-aniket"' in launcher
    assert '!= "neurips-aniket"' in worker
    assert "tmux new-session -d" in launcher
