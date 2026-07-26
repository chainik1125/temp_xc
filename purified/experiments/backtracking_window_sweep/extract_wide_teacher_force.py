"""Teacher-force the pinned Ward traces into an exact T16 event artifact.

This builder never discovers or fetches source traces.  The caller must supply
one ``traces.json`` file together with its expected SHA-256, repository path,
and source commit.  The file is joined to the pinned prompts, sentence labels,
and official six-offset artifact by ``question_id``, locked list
position/``trace_idx``, and event key before a model is loaded.  The approved
source has no explicit ``trace_idx`` field; an optional one is accepted only
when it equals the locked list position.

Extraction is resumable at one file per trace.  Every source-eligible event's
offsets ``-13..-8`` are compared bit-for-bit against the official artifact.
Only exact events are committed; deterministic public-key exclusion manifests
record nonexact events and tokenizer round-trip exclusions without retaining
raw text, token IDs, or activation values.  Assembly repeats the exact keyed
comparison, then emits only exact events for which all offsets ``-23..-8``
exist.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .protocol_t16 import (
    ARTIFACT_OFFSETS,
    EXPECTED_WIDTH,
    TEACHER_FORCE_PROTOCOL,
    cohort_sha256,
)
from .reconstruct_wide_artifact import (
    EXPECTED_LABEL_RECORDS,
    EXPECTED_OFFICIAL_SHAPE,
    EXPECTED_SENTENCES,
    LOCKED_LABELS,
    LOCKED_OFFICIAL,
    LOCKED_PROMPTS,
    OFFICIAL_OFFSETS,
    ProvenanceError,
    sha256_file,
    verify_locked_file,
)


MODEL_ID = "NousResearch/Meta-Llama-3.1-8B"
MODEL_REVISION = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
TOKENIZER_ID = MODEL_ID
TOKENIZER_REVISION = MODEL_REVISION
LAYER = 10
COMPONENT = "resid_post"
MODEL_DTYPE = "bfloat16"
OUTPUT_DTYPE = np.dtype(np.float32)
ATTENTION_IMPLEMENTATION = "sdpa"
TOKENIZATION_ADD_SPECIAL_TOKENS = False
REQUEST_SCHEMA = "ward-c7-wide-teacher-force-request.v4"
OUTPUT_SCHEMA = "ward-c7-wide-teacher-force-output.v4"
SHARD_SCHEMA = "ward-c7-wide-teacher-force-shard.v4"
EXCLUSION_SCHEMA = "ward-c7-wide-teacher-force-exclusions.v4"
MANIFEST_SCHEMA = "ward-c7-wide-teacher-force-manifest.v4"
TAIL_COMPARATOR = "float32-uint32-bit-exact.v1"
TRACE_ROUNDTRIP_POLICY = "exclude-decoded-text-mismatch-before-forward.v1"
EVENT_SELECTION_POLICY = "source-aligned-and-bit-exact-tail-and-t16.v1"
CATEGORY_FIELD = "category"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SourceValidationError(RuntimeError):
    """The supplied trace source does not match its explicit provenance."""


class TraceAlignmentError(RuntimeError):
    """A raw trace cannot be joined unambiguously to the locked labels."""


class TokenizerRoundTripError(TraceAlignmentError):
    """Decoded tokenizer output differs from the exact pinned source text."""


class ExactTailError(RuntimeError):
    """Teacher-forced activations differ from the official trailing six."""


class ShardValidationError(RuntimeError):
    """A resumable trace shard is incomplete or has drifted."""


@dataclass(frozen=True)
class EventSpec:
    key: str
    question_id: str
    trace_idx: int
    sentence_idx: int
    label: int
    target_char: int
    label_char_start: int
    label_char_end: int
    label_text_matches: bool


@dataclass(frozen=True)
class TraceSpec:
    question_id: str
    trace_idx: int
    category: str
    full_response: str
    full_response_sha256: str
    thinking_char_start: int
    thinking_char_end: int
    label_text_matches: int
    label_sentence_count: int
    events: tuple[EventSpec, ...]


@dataclass(frozen=True)
class SourcePin:
    path: str
    sha256: str
    commit: str


@dataclass(frozen=True)
class Request:
    schema_version: str
    protocol_version: str
    source: SourcePin
    prompts_sha256: str
    labels_sha256: str
    official_sha256: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    layer: int
    component: str
    model_dtype: str
    output_dtype: str
    attention_implementation: str
    add_special_tokens: bool
    output_schema: str
    exclusion_schema: str
    tail_comparator: str
    trace_roundtrip_policy: str
    event_selection_policy: str
    category_field: str
    offsets: tuple[int, ...]
    trailing_offsets: tuple[int, ...]
    trace_count: int
    sentence_count: int
    official_rows: int
    source_eligible_rows: int
    source_eligible_sha256: str
    source_exclusions_sha256: str


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _event_key_sha256(key: str) -> str:
    return _sha256_text(key)


def _cohort_category_sha256(
    keys: Sequence[str],
    labels: Sequence[int],
    categories: Sequence[str],
) -> str:
    if not (len(keys) == len(labels) == len(categories)):
        raise ValueError("cohort keys, labels, and categories must align")
    return _canonical_sha256(
        {
            "entries": [
                {
                    "key": str(key),
                    "label": int(label),
                    "category": str(category),
                }
                for key, label, category in zip(
                    keys,
                    labels,
                    categories,
                    strict=True,
                )
            ]
        }
    )


def _cohort_counts(
    labels: Sequence[int],
    categories: Sequence[str],
) -> dict[str, Any]:
    if len(labels) != len(categories):
        raise ValueError("cohort labels and categories must align")
    class_counts = {"0": 0, "1": 0}
    category_counts: dict[str, int] = {}
    category_class_counts: dict[str, dict[str, int]] = {}
    for raw_label, raw_category in zip(labels, categories, strict=True):
        label = str(int(raw_label))
        category = str(raw_category)
        if label not in class_counts:
            raise ValueError(f"cohort label must be binary, got {raw_label!r}")
        class_counts[label] += 1
        category_counts[category] = category_counts.get(category, 0) + 1
        cell = category_class_counts.setdefault(
            category,
            {"0": 0, "1": 0},
        )
        cell[label] += 1
    return {
        "rows": len(labels),
        "class_counts": class_counts,
        "category_counts": dict(sorted(category_counts.items())),
        "category_class_counts": {
            category: category_class_counts[category]
            for category in sorted(category_class_counts)
        },
    }


def _cohort_summary(
    keys: Sequence[str],
    labels: Sequence[int],
    categories: Sequence[str],
) -> dict[str, Any]:
    key_array = np.asarray(keys)
    label_array = np.asarray(labels, dtype=np.uint8)
    return {
        **_cohort_counts(labels, categories),
        "sha256": cohort_sha256(key_array, label_array),
        "category_sha256": _cohort_category_sha256(
            keys,
            labels,
            categories,
        ),
    }


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _load_json_array(path: Path, *, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise SourceValidationError(f"{name} must be a JSON array of objects")
    return payload


def validate_source_pin(
    traces_path: Path,
    *,
    expected_sha256: str,
    source_path: str,
    source_commit: str,
) -> SourcePin:
    """Verify the caller-supplied trace source without discovering any source."""

    if not traces_path.is_file():
        raise SourceValidationError(f"missing supplied traces file: {traces_path}")
    expected_sha256 = expected_sha256.lower()
    source_commit = source_commit.lower()
    if HEX64.fullmatch(expected_sha256) is None:
        raise SourceValidationError("--traces-sha256 must be 64 hex characters")
    if HEX40.fullmatch(source_commit) is None:
        raise SourceValidationError("--source-commit must be a 40-hex commit")
    pure_path = PurePosixPath(source_path)
    if (
        not source_path.strip()
        or pure_path.is_absolute()
        or ".." in pure_path.parts
        or pure_path.name in {"", ".", ".."}
    ):
        raise SourceValidationError(
            "--source-path must be a nonempty repository-relative POSIX path"
        )
    observed = sha256_file(traces_path)
    if observed != expected_sha256:
        raise SourceValidationError(
            f"traces SHA-256 mismatch: observed {observed}, "
            f"expected {expected_sha256}"
        )
    return SourcePin(
        path=source_path,
        sha256=observed,
        commit=source_commit,
    )


def _thinking_bounds(
    full_response: str,
    thinking_process: str,
    *,
    trace_idx: int,
) -> tuple[int, int]:
    """Validate the source-native thinking-prefix coordinate system.

    The approved source stores ``thinking_process`` as an exact prefix of
    ``full_response`` and the locked sentence offsets are relative to that
    prefix.  Public labels contain gaps and a minority of conflicting
    overlapping annotations, so every sentence string cannot agree.  The
    final keyed activation comparison remains the authoritative event proof.
    """

    if not thinking_process:
        raise TraceAlignmentError(
            f"trace {trace_idx}: thinking_process must be nonempty text"
        )
    if not full_response.startswith(thinking_process):
        raise TraceAlignmentError(
            f"trace {trace_idx}: thinking_process is not an exact "
            "full_response prefix"
        )

    return 0, len(thinking_process)


def _validate_sentence(
    sentence: Mapping[str, Any],
    *,
    trace_idx: int,
    sentence_idx: int,
) -> tuple[str, int, int, int]:
    text = sentence.get("sentence")
    if not isinstance(text, str) or not text:
        raise TraceAlignmentError(
            f"trace {trace_idx} sentence {sentence_idx}: sentence must be text"
        )
    try:
        start = int(sentence["char_start"])
        end = int(sentence["char_end"])
    except (KeyError, TypeError, ValueError) as error:
        raise TraceAlignmentError(
            f"trace {trace_idx} sentence {sentence_idx}: invalid character span"
        ) from error
    if start < 0 or end <= start or end - start != len(text):
        raise TraceAlignmentError(
            f"trace {trace_idx} sentence {sentence_idx}: span/text length mismatch"
        )
    label = sentence.get("is_backtracking")
    if not isinstance(label, bool):
        raise TraceAlignmentError(
            f"trace {trace_idx} sentence {sentence_idx}: label must be boolean"
        )
    return text, start, end, int(label)


def validate_trace_records(
    prompts: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    *,
    official_keys: Sequence[str],
    official_labels: np.ndarray,
    strict_counts: bool,
) -> tuple[list[TraceSpec], dict[str, Any]]:
    """Join the raw records, locked labels, and official keyed cohort."""

    if not (len(prompts) == len(labels) == len(traces)):
        raise TraceAlignmentError(
            "prompts, labels, and traces must have exactly the same record count"
        )
    if strict_counts and len(traces) != EXPECTED_LABEL_RECORDS:
        raise TraceAlignmentError(
            f"trace count {len(traces)} != locked count {EXPECTED_LABEL_RECORDS}"
        )
    if len(official_keys) != len(official_labels):
        raise TraceAlignmentError("official keys and labels have different lengths")
    official_keys = [str(key) for key in official_keys]
    if len(set(official_keys)) != len(official_keys):
        raise TraceAlignmentError("official event keys are not unique")
    official_label_by_key = {
        key: int(label)
        for key, label in zip(official_keys, official_labels, strict=True)
    }
    official_key_set = set(official_keys)

    all_label_keys: list[str] = []
    all_label_by_key: dict[str, int] = {}
    trace_specs: list[TraceSpec] = []
    total_sentences = 0
    question_ids: list[str] = []
    total_text_matches = 0
    eligible_keys: list[str] = []
    eligible_labels: list[int] = []
    eligible_categories: list[str] = []
    official_categories_by_key: dict[str, str] = {}
    exclusions: list[dict[str, Any]] = []

    for row_index, (prompt, label_row, trace) in enumerate(
        zip(prompts, labels, traces, strict=True)
    ):
        prompt_id = str(prompt.get("id"))
        question_id = str(label_row.get("question_id"))
        if prompt_id != question_id:
            raise TraceAlignmentError(
                f"record {row_index}: prompt ID {prompt_id!r} != "
                f"label ID {question_id!r}"
            )
        try:
            label_trace_idx = int(label_row["trace_idx"])
        except (KeyError, TypeError, ValueError) as error:
            raise TraceAlignmentError(
                f"record {row_index}: locked trace_idx must be an integer"
            ) from error
        if label_trace_idx != row_index:
            raise TraceAlignmentError(
                f"record {row_index}: locked trace_idx must equal row order"
            )
        if "trace_idx" in trace:
            try:
                raw_trace_idx = int(trace["trace_idx"])
            except (TypeError, ValueError) as error:
                raise TraceAlignmentError(
                    f"record {row_index}: optional raw trace_idx must be an integer"
                ) from error
            if raw_trace_idx != row_index:
                raise TraceAlignmentError(
                    f"record {row_index}: optional raw trace_idx "
                    f"{raw_trace_idx} disagrees with locked row order"
                )
        trace_idx = row_index
        category = trace.get(CATEGORY_FIELD)
        if not isinstance(category, str) or not category.strip():
            raise TraceAlignmentError(
                f"record {row_index}: source category must be nonempty text"
            )
        category = category.strip()
        for locked_record in (prompt, label_row):
            locked_category = locked_record.get(CATEGORY_FIELD)
            if (
                locked_category is not None
                and str(locked_category).strip() != category
            ):
                raise TraceAlignmentError(
                    f"record {row_index}: source category disagrees with "
                    "a locked category"
                )
        raw_question_id = trace.get("question_id")
        if raw_question_id is None or str(raw_question_id) != question_id:
            raise TraceAlignmentError(
                f"record {row_index}: raw question_id "
                f"{raw_question_id!r} != {question_id!r}"
            )
        full_response = trace.get("full_response")
        if not isinstance(full_response, str) or not full_response:
            raise TraceAlignmentError(
                f"record {row_index}: full_response must be nonempty text"
            )
        thinking_process = trace.get("thinking_process")
        if not isinstance(thinking_process, str) or not thinking_process:
            raise TraceAlignmentError(
                f"record {row_index}: thinking_process must be nonempty text"
            )
        sentences = label_row.get("sentences")
        if not isinstance(sentences, list):
            raise TraceAlignmentError(
                f"record {row_index}: sentences must be a list"
            )
        indexed_sentences = []
        for sentence_idx, sentence in enumerate(sentences):
            parsed = _validate_sentence(
                sentence,
                trace_idx=row_index,
                sentence_idx=sentence_idx,
            )
            key = f"{question_id}|{trace_idx}|{sentence_idx}"
            indexed_sentences.append((sentence_idx, key, sentence, parsed))
        thinking_start, thinking_end = _thinking_bounds(
            full_response,
            thinking_process,
            trace_idx=row_index,
        )
        total_sentences += len(sentences)
        question_ids.append(question_id)

        events: list[EventSpec] = []
        match_count = 0
        for sentence_idx, key, _sentence, parsed in indexed_sentences:
            text, start, end, label = parsed
            if key in all_label_by_key:
                raise TraceAlignmentError(f"duplicate locked event key {key!r}")
            all_label_keys.append(key)
            all_label_by_key[key] = label
            if key not in official_key_set:
                continue
            official_categories_by_key[key] = category
            if thinking_start + end > thinking_end:
                exclusions.append(
                    {
                        "key": key,
                        "key_sha256": _event_key_sha256(key),
                        "trace_idx": trace_idx,
                        "sentence_idx": sentence_idx,
                        "label": label,
                        "category": category,
                        "reason": "span_out_of_bounds",
                    }
                )
                continue
            matches = full_response[
                thinking_start + start : thinking_start + end
            ] == text
            if not matches:
                exclusions.append(
                    {
                        "key": key,
                        "key_sha256": _event_key_sha256(key),
                        "trace_idx": trace_idx,
                        "sentence_idx": sentence_idx,
                        "label": label,
                        "category": category,
                        "reason": "sentence_text_mismatch",
                    }
                )
                continue
            match_count += 1
            eligible_keys.append(key)
            eligible_labels.append(label)
            eligible_categories.append(category)
            events.append(
                EventSpec(
                    key=key,
                    question_id=question_id,
                    trace_idx=trace_idx,
                    sentence_idx=sentence_idx,
                    label=label,
                    target_char=thinking_start + start,
                    label_char_start=start,
                    label_char_end=end,
                    label_text_matches=True,
                )
            )
        total_text_matches += match_count
        trace_specs.append(
            TraceSpec(
                question_id=question_id,
                trace_idx=trace_idx,
                category=category,
                full_response=full_response,
                full_response_sha256=_sha256_text(full_response),
                thinking_char_start=thinking_start,
                thinking_char_end=thinking_end,
                label_text_matches=match_count,
                label_sentence_count=len(sentences),
                events=tuple(events),
            )
        )

    if len(set(question_ids)) != len(question_ids):
        raise TraceAlignmentError("prompt/question IDs must be unique")
    if strict_counts and total_sentences != EXPECTED_SENTENCES:
        raise TraceAlignmentError(
            f"sentence count {total_sentences} != locked count {EXPECTED_SENTENCES}"
        )
    missing = official_key_set.difference(all_label_by_key)
    if missing:
        raise TraceAlignmentError(
            f"{len(missing)} official event keys are absent from locked labels"
        )
    for key, label in official_label_by_key.items():
        if all_label_by_key[key] != label:
            raise TraceAlignmentError(
                f"official label differs from locked label for {key!r}"
            )
    ordered_subset = [
        key for key in all_label_keys if key in official_key_set
    ]
    if ordered_subset != official_keys:
        raise TraceAlignmentError(
            "official keys are not the exact order-preserving locked-label subset"
        )
    if not eligible_keys:
        raise TraceAlignmentError(
            "no official event has an in-bounds exact source-text alignment"
        )
    exclusion_counts = {
        reason: sum(row["reason"] == reason for row in exclusions)
        for reason in ("span_out_of_bounds", "sentence_text_mismatch")
    }
    official_categories = [
        official_categories_by_key[key] for key in official_keys
    ]
    official_summary = _cohort_summary(
        official_keys,
        official_labels.tolist(),
        official_categories,
    )
    eligible_summary = _cohort_summary(
        eligible_keys,
        eligible_labels,
        eligible_categories,
    )
    return trace_specs, {
        "trace_count": len(trace_specs),
        "sentence_count": total_sentences,
        "official_rows": len(official_keys),
        "official_cohort": official_summary,
        "source_eligible_rows": len(eligible_keys),
        "source_eligible_sha256": eligible_summary["sha256"],
        "source_eligible_category_sha256": eligible_summary[
            "category_sha256"
        ],
        "source_eligible_class_counts": eligible_summary["class_counts"],
        "source_eligible_category_counts": eligible_summary[
            "category_counts"
        ],
        "source_eligible_category_class_counts": eligible_summary[
            "category_class_counts"
        ],
        "source_excluded_rows": len(exclusions),
        "source_exclusion_counts": exclusion_counts,
        "source_exclusions_sha256": _canonical_sha256(
            {"entries": exclusions}
        ),
        "_source_exclusion_entries": exclusions,
        "label_text_matches": total_text_matches,
        "label_text_mismatches": exclusion_counts[
            "sentence_text_mismatch"
        ],
        "label_span_out_of_bounds": exclusion_counts[
            "span_out_of_bounds"
        ],
        "ordered_official_key_join": True,
        "ordered_source_eligible_subset": True,
    }


def tokenize_full_response(tokenizer, full_response: str) -> dict[str, Any]:
    """Tokenize the full response with the pinned fast backend and BOS policy."""

    if not getattr(tokenizer, "is_fast", False):
        raise TraceAlignmentError("the pinned tokenizer must be fast")
    encoded = tokenizer(
        full_response,
        add_special_tokens=TOKENIZATION_ADD_SPECIAL_TOKENS,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [
        tuple(int(value) for value in pair)
        for pair in encoded["offset_mapping"]
    ]
    if len(input_ids) != len(offsets):
        raise TraceAlignmentError("token IDs and offsets have different lengths")
    previous_start = 0
    maximum_end = 0
    for token_index, (start, end) in enumerate(offsets):
        if not (0 <= start <= end <= len(full_response)):
            raise TraceAlignmentError(
                f"token {token_index}: invalid response offset [{start}, {end})"
            )
        if end > start:
            if start < previous_start:
                raise TraceAlignmentError("tokenizer offsets move backward")
            previous_start = start
            maximum_end = max(maximum_end, end)
    if full_response and maximum_end != len(full_response):
        raise TraceAlignmentError("tokenizer offsets do not cover full_response")
    decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
    if decoded != full_response:
        raise TokenizerRoundTripError(
            "pinned tokenizer does not exactly round-trip full_response"
        )
    return {"input_ids": input_ids, "offsets": offsets}


def token_containing_char(
    offsets: Sequence[tuple[int, int]],
    target_char: int,
) -> int:
    """Return the unique token whose nonempty offset contains target_char."""

    matches = [
        index
        for index, (start, end) in enumerate(offsets)
        if start <= target_char < end
    ]
    if len(matches) != 1:
        raise TraceAlignmentError(
            f"character {target_char} is covered by {len(matches)} tokens"
        )
    return matches[0]


def _float32_bit_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left_array = np.ascontiguousarray(left, dtype=np.float32)
    right_array = np.ascontiguousarray(right, dtype=np.float32)
    return (
        left_array.shape == right_array.shape
        and np.array_equal(
            left_array.view(np.uint32),
            right_array.view(np.uint32),
        )
    )


def _empty_trace_arrays(expected_width: int = EXPECTED_WIDTH) -> dict[str, np.ndarray]:
    return {
        "wide_X": np.empty(
            (0, len(ARTIFACT_OFFSETS), expected_width),
            dtype=OUTPUT_DTYPE,
        ),
        "wide_keys": np.asarray([], dtype=str),
        "wide_is_bt": np.asarray([], dtype=np.uint8),
        "wide_boundary_token": np.asarray([], dtype=np.int32),
        "tail_only_X": np.empty(
            (0, len(OFFICIAL_OFFSETS), expected_width),
            dtype=OUTPUT_DTYPE,
        ),
        "tail_only_keys": np.asarray([], dtype=str),
        "tail_only_is_bt": np.asarray([], dtype=np.uint8),
        "tail_only_boundary_token": np.asarray([], dtype=np.int32),
    }


def _event_exclusion(
    event: EventSpec,
    *,
    category: str,
    reason: str,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "key": event.key,
        "key_sha256": _event_key_sha256(event.key),
        "trace_idx": event.trace_idx,
        "sentence_idx": event.sentence_idx,
        "label": event.label,
        "category": category,
        "reason": reason,
    }
    if metrics:
        result["metrics"] = dict(metrics)
    return result


def _tail_mismatch_metrics(
    extracted_tail: np.ndarray,
    official_tail: np.ndarray,
) -> dict[str, Any]:
    extracted = np.ascontiguousarray(extracted_tail, dtype=np.float32)
    official = np.ascontiguousarray(official_tail, dtype=np.float32)
    if extracted.shape != official.shape:
        raise TraceAlignmentError(
            f"official tail shape {official.shape} != extracted {extracted.shape}"
        )
    bit_mismatch = extracted.view(np.uint32) != official.view(np.uint32)
    finite = np.isfinite(extracted) & np.isfinite(official)
    finite_difference = np.abs(extracted[finite] - official[finite])
    if finite_difference.size:
        max_abs: float | None = float(np.max(finite_difference))
        mean_abs: float | None = float(np.mean(finite_difference))
        rmse: float | None = float(
            np.sqrt(np.mean(np.square(finite_difference, dtype=np.float64)))
        )
    else:
        max_abs = None
        mean_abs = None
        rmse = None
    return {
        "compared_values": int(extracted.size),
        "bit_mismatched_values": int(np.count_nonzero(bit_mismatch)),
        "finite_pairs": int(np.count_nonzero(finite)),
        "nonfinite_pairs": int(extracted.size - np.count_nonzero(finite)),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
    }


def partition_trace_activations(
    hidden: np.ndarray,
    *,
    trace: TraceSpec,
    offsets: Sequence[tuple[int, int]],
    official_x_by_key: Mapping[str, np.ndarray],
    expected_width: int = EXPECTED_WIDTH,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Retain only bit-exact events and describe every exclusion."""

    hidden = np.asarray(hidden)
    if hidden.ndim != 2 or hidden.shape[1] != expected_width:
        raise TraceAlignmentError(
            f"captured hidden state must have shape (tokens, {expected_width}), "
            f"got {hidden.shape}"
        )
    if len(hidden) != len(offsets):
        raise TraceAlignmentError("captured hidden state and offsets disagree")

    arrays = _empty_trace_arrays(expected_width)
    wide_rows: list[np.ndarray] = []
    wide_keys: list[str] = []
    wide_labels: list[int] = []
    wide_boundaries: list[int] = []
    tail_rows: list[np.ndarray] = []
    tail_keys: list[str] = []
    tail_labels: list[int] = []
    tail_boundaries: list[int] = []
    exclusions: list[dict[str, Any]] = []
    exact_keys: list[str] = []
    exact_labels: list[int] = []
    exact_categories: list[str] = []
    compared_rows = 0
    bit_mismatched_values = 0
    compared_values = 0
    finite_pairs = 0
    nonfinite_pairs = 0
    finite_abs_sum = 0.0
    finite_sq_sum = 0.0
    maximum_abs = 0.0
    have_finite_difference = False

    for event in trace.events:
        try:
            boundary = token_containing_char(offsets, event.target_char)
        except TraceAlignmentError:
            exclusions.append(
                _event_exclusion(
                    event,
                    category=trace.category,
                    reason="boundary_unavailable",
                )
            )
            continue
        tail_positions = [boundary + value for value in OFFICIAL_OFFSETS]
        if min(tail_positions) < 0:
            exclusions.append(
                _event_exclusion(
                    event,
                    category=trace.category,
                    reason="tail_offsets_unavailable",
                )
            )
            continue
        extracted_tail = np.asarray(
            hidden[tail_positions],
            dtype=OUTPUT_DTYPE,
        )
        official_tail = np.asarray(
            official_x_by_key[event.key],
            dtype=OUTPUT_DTYPE,
        )
        compared_rows += 1
        if not _float32_bit_equal(extracted_tail, official_tail):
            metrics = _tail_mismatch_metrics(
                extracted_tail,
                official_tail,
            )
            exclusions.append(
                _event_exclusion(
                    event,
                    category=trace.category,
                    reason="tail_not_bit_exact",
                    metrics=metrics,
                )
            )
            bit_mismatched_values += int(metrics["bit_mismatched_values"])
            compared_values += int(metrics["compared_values"])
            finite_pairs += int(metrics["finite_pairs"])
            nonfinite_pairs += int(metrics["nonfinite_pairs"])
            finite_mask = np.isfinite(extracted_tail) & np.isfinite(
                official_tail
            )
            finite_difference = np.abs(
                extracted_tail[finite_mask] - official_tail[finite_mask]
            )
            if finite_difference.size:
                have_finite_difference = True
                finite_abs_sum += float(
                    np.sum(finite_difference, dtype=np.float64)
                )
                finite_sq_sum += float(
                    np.sum(
                        np.square(finite_difference, dtype=np.float64),
                        dtype=np.float64,
                    )
                )
                maximum_abs = max(
                    maximum_abs,
                    float(np.max(finite_difference)),
                )
            continue
        exact_keys.append(event.key)
        exact_labels.append(event.label)
        exact_categories.append(trace.category)
        wide_positions = [boundary + value for value in ARTIFACT_OFFSETS]
        if min(wide_positions) >= 0:
            wide_rows.append(
                np.asarray(hidden[wide_positions], dtype=OUTPUT_DTYPE)
            )
            wide_keys.append(event.key)
            wide_labels.append(event.label)
            wide_boundaries.append(boundary)
        else:
            tail_rows.append(extracted_tail)
            tail_keys.append(event.key)
            tail_labels.append(event.label)
            tail_boundaries.append(boundary)

    arrays.update({
        "wide_X": (
            np.stack(wide_rows).astype(OUTPUT_DTYPE, copy=False)
            if wide_rows
            else np.empty(
                (0, len(ARTIFACT_OFFSETS), expected_width),
                dtype=OUTPUT_DTYPE,
            )
        ),
        "wide_keys": np.asarray(wide_keys),
        "wide_is_bt": np.asarray(wide_labels, dtype=np.uint8),
        "wide_boundary_token": np.asarray(wide_boundaries, dtype=np.int32),
        "tail_only_X": (
            np.stack(tail_rows).astype(OUTPUT_DTYPE, copy=False)
            if tail_rows
            else np.empty(
                (0, len(OFFICIAL_OFFSETS), expected_width),
                dtype=OUTPUT_DTYPE,
            )
        ),
        "tail_only_keys": np.asarray(tail_keys),
        "tail_only_is_bt": np.asarray(tail_labels, dtype=np.uint8),
        "tail_only_boundary_token": np.asarray(
            tail_boundaries,
            dtype=np.int32,
        ),
    })
    unavailable_rows = sum(
        entry["reason"] in {
            "boundary_unavailable",
            "tail_offsets_unavailable",
        }
        for entry in exclusions
    )
    nonexact_rows = sum(
        entry["reason"] == "tail_not_bit_exact" for entry in exclusions
    )
    comparison = {
        "comparator": TAIL_COMPARATOR,
        "source_eligible_rows": len(trace.events),
        "tail_compared_rows": compared_rows,
        "exact_tail_rows": len(exact_keys),
        "nonexact_tail_rows": nonexact_rows,
        "unavailable_tail_rows": unavailable_rows,
        "wide_rows": len(wide_keys),
        "insufficient_t16_context_rows": len(tail_keys),
        "exact_tail_cohort": _cohort_summary(
            exact_keys,
            exact_labels,
            exact_categories,
        ),
        "nonexact_tail_metrics": {
            "events": nonexact_rows,
            "compared_values": compared_values,
            "bit_mismatched_values": bit_mismatched_values,
            "finite_pairs": finite_pairs,
            "nonfinite_pairs": nonfinite_pairs,
            "max_abs": maximum_abs if have_finite_difference else None,
            "mean_abs": (
                finite_abs_sum / finite_pairs if finite_pairs else None
            ),
            "rmse": (
                float(np.sqrt(finite_sq_sum / finite_pairs))
                if finite_pairs
                else None
            ),
        },
        "exclusions": exclusions,
    }
    if (
        comparison["source_eligible_rows"]
        != comparison["exact_tail_rows"]
        + comparison["nonexact_tail_rows"]
        + comparison["unavailable_tail_rows"]
    ):
        raise TraceAlignmentError("trace cohort accounting failed")
    return arrays, comparison


def _model_and_tokenizer(device: str):
    import torch
    from transformers import AutoModel, PreTrainedTokenizerFast

    if not device.startswith("cuda"):
        raise ValueError("teacher-force extraction requires an explicit CUDA device")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
        trust_remote_code=False,
    )
    if not tokenizer.is_fast:
        raise TraceAlignmentError("pinned tokenizer did not load as fast")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=ATTENTION_IMPLEMENTATION,
    )
    model.eval()
    backbone = model.model if hasattr(model, "model") else model
    if not hasattr(backbone, "layers"):
        raise TypeError("cannot locate transformer blocks on pinned model")
    if len(backbone.layers) <= LAYER:
        raise ValueError(f"pinned model does not have layer {LAYER}")
    if int(model.config.hidden_size) != EXPECTED_WIDTH:
        raise ValueError(
            f"model hidden size {model.config.hidden_size} != {EXPECTED_WIDTH}"
        )
    observed_model_revision = getattr(model.config, "_commit_hash", None)
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
    if observed_model_revision not in {None, MODEL_REVISION}:
        raise ProvenanceError(
            f"loaded model revision {observed_model_revision!r} != {MODEL_REVISION}"
        )
    if tokenizer_revision not in {None, TOKENIZER_REVISION}:
        raise ProvenanceError(
            f"loaded tokenizer revision {tokenizer_revision!r} "
            f"!= {TOKENIZER_REVISION}"
        )
    runtime = {
        "model_revision_requested": MODEL_REVISION,
        "model_revision_observed": observed_model_revision,
        "tokenizer_revision_requested": TOKENIZER_REVISION,
        "tokenizer_revision_observed": tokenizer_revision,
        "tokenizer_backend_sha256": _sha256_text(
            tokenizer.backend_tokenizer.to_str()
        ),
        "model_type": str(model.config.model_type),
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "max_position_embeddings": int(model.config.max_position_embeddings),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.device(device)),
    }
    return model, tokenizer, backbone.layers[LAYER], runtime


def _forward_trace(
    model,
    layer_module,
    *,
    input_ids: Sequence[int],
    device: str,
) -> np.ndarray:
    import torch

    values = torch.tensor(input_ids, dtype=torch.long, device=device)[None]
    attention_mask = torch.ones_like(values)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=values,
                attention_mask=attention_mask,
                use_cache=False,
            )
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("layer-10 block-output hook did not fire")
    hidden = (
        captured["hidden"][0]
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    del output, values, attention_mask, captured
    return hidden


def _trace_input_sha256(trace: TraceSpec) -> str:
    payload = {
        "question_id": trace.question_id,
        "trace_idx": trace.trace_idx,
        "category": trace.category,
        "full_response_sha256": trace.full_response_sha256,
        "thinking_char_start": trace.thinking_char_start,
        "thinking_char_end": trace.thinking_char_end,
        "events": [asdict(event) for event in trace.events],
    }
    return _canonical_sha256(payload)


def _required_shard_arrays() -> set[str]:
    return {
        "wide_X",
        "wide_keys",
        "wide_is_bt",
        "wide_boundary_token",
        "tail_only_X",
        "tail_only_keys",
        "tail_only_is_bt",
        "tail_only_boundary_token",
    }


def _validate_existing_final_output(
    artifact_path: Path,
    manifest_path: Path,
    *,
    output_dir: Path,
    traces: Sequence[TraceSpec],
    request_sha256: str,
    expected_keys: Sequence[str],
    expected_labels: Sequence[int],
    expected_width: int = EXPECTED_WIDTH,
) -> tuple[dict[str, Any], str]:
    """Validate a completed artifact against its current trace shards.

    The final manifest's artifact hash is not an independent trust anchor:
    both files can be edited together.  A valid resume therefore also proves
    every final activation row bit-for-bit against the already validated
    per-trace shard that produced it.
    """

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShardValidationError(
            "existing final manifest is unreadable"
        ) from error
    if not isinstance(manifest, dict):
        raise ShardValidationError("existing final manifest must be an object")
    expected_manifest_fields = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "status": "complete",
        "request_sha256": request_sha256,
    }
    manifest_drift = {
        key: (manifest.get(key), value)
        for key, value in expected_manifest_fields.items()
        if manifest.get(key) != value
    }
    if manifest_drift:
        raise ShardValidationError(
            f"existing final manifest drifted: {manifest_drift}"
        )
    claimed_sha = manifest.get("output", {}).get("sha256")
    observed_sha = sha256_file(artifact_path)
    if claimed_sha != observed_sha:
        raise ShardValidationError("existing final artifact hash failed")

    expected_key_list = [str(key) for key in expected_keys]
    expected_label_array = np.asarray(expected_labels, dtype=np.uint8)
    if len(expected_key_list) != len(expected_label_array):
        raise ShardValidationError(
            "existing final expected keys and labels do not align"
        )
    try:
        payload_context = np.load(artifact_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ShardValidationError(
            "existing final artifact is unreadable"
        ) from error
    with payload_context as payload:
        if set(payload.files) != {"X", "is_bt", "keys", "offsets"}:
            raise ShardValidationError(
                "existing final artifact arrays differ from contract"
            )
        artifact_x = payload["X"]
        artifact_labels = payload["is_bt"]
        artifact_keys = payload["keys"].astype(str)
        artifact_offsets = payload["offsets"]
        expected_shape = (
            len(expected_key_list),
            len(ARTIFACT_OFFSETS),
            expected_width,
        )
        if artifact_x.shape != expected_shape:
            raise ShardValidationError(
                f"existing final artifact shape {artifact_x.shape} "
                f"!= {expected_shape}"
            )
        if artifact_x.dtype != OUTPUT_DTYPE:
            raise ShardValidationError(
                "existing final artifact activations are not float32"
            )
        if artifact_labels.dtype != np.dtype(np.uint8):
            raise ShardValidationError(
                "existing final artifact labels are not uint8"
            )
        if artifact_offsets.dtype != np.dtype(np.int32):
            raise ShardValidationError(
                "existing final artifact offsets are not int32"
            )
        if artifact_keys.tolist() != expected_key_list:
            raise ShardValidationError(
                "existing final artifact key order drifted"
            )
        if not np.array_equal(artifact_labels, expected_label_array):
            raise ShardValidationError(
                "existing final artifact labels drifted"
            )
        if not np.array_equal(
            artifact_offsets,
            np.asarray(ARTIFACT_OFFSETS, dtype=np.int32),
        ):
            raise ShardValidationError(
                "existing final artifact offsets drifted"
            )

        cursor = 0
        for trace in traces:
            tensor_path, _, _ = _trace_paths(output_dir, trace.trace_idx)
            try:
                shard_context = np.load(tensor_path, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ShardValidationError(
                    f"trace {trace.trace_idx}: shard is unreadable during "
                    "final resume validation"
                ) from error
            with shard_context as shard:
                shard_keys = shard["wide_keys"].astype(str).tolist()
                shard_x = shard["wide_X"]
                stop = cursor + len(shard_keys)
                if expected_key_list[cursor:stop] != shard_keys:
                    raise ShardValidationError(
                        f"trace {trace.trace_idx}: final/shard key order drifted"
                    )
                if not _float32_bit_equal(
                    np.asarray(artifact_x[cursor:stop], dtype=OUTPUT_DTYPE),
                    np.asarray(shard_x, dtype=OUTPUT_DTYPE),
                ):
                    raise ShardValidationError(
                        f"trace {trace.trace_idx}: existing final artifact "
                        "activations drifted from trace shards"
                    )
                cursor = stop
        if cursor != len(expected_key_list):
            raise ShardValidationError(
                f"existing final artifact consumed {cursor} shard rows, "
                f"expected {len(expected_key_list)}"
            )
    return manifest, observed_sha


def _validate_trace_shard(
    tensor_path: Path,
    sidecar_path: Path,
    exclusion_path: Path,
    *,
    trace: TraceSpec,
    request_sha256: str,
    expected_width: int = EXPECTED_WIDTH,
    official_x_by_key: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any] | None:
    outcomes = (
        tensor_path.exists(),
        sidecar_path.exists(),
        exclusion_path.exists(),
    )
    if not any(outcomes):
        return None
    if not all(outcomes):
        raise ShardValidationError(
            f"incomplete shard outcome for trace {trace.trace_idx}"
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    exclusion_document = json.loads(
        exclusion_path.read_text(encoding="utf-8")
    )
    expected = {
        "schema_version": SHARD_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "request_sha256": request_sha256,
        "question_id": trace.question_id,
        "trace_idx": trace.trace_idx,
        "category": trace.category,
        "full_response_sha256": trace.full_response_sha256,
        "trace_input_sha256": _trace_input_sha256(trace),
        "tail_comparator": TAIL_COMPARATOR,
        "tail_exact_for_retained_events": True,
    }
    drift = {
        key: (sidecar.get(key), value)
        for key, value in expected.items()
        if sidecar.get(key) != value
    }
    if drift:
        raise ShardValidationError(
            f"trace {trace.trace_idx}: existing sidecar drifted: {drift}"
        )
    observed_sha = sha256_file(tensor_path)
    if sidecar.get("sha256") != observed_sha:
        raise ShardValidationError(
            f"trace {trace.trace_idx}: shard SHA-256 mismatch"
        )
    exclusion_expected = {
        "schema_version": EXCLUSION_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "request_sha256": request_sha256,
        "question_id": trace.question_id,
        "trace_idx": trace.trace_idx,
        "category": trace.category,
        "trace_input_sha256": _trace_input_sha256(trace),
    }
    exclusion_drift = {
        key: (exclusion_document.get(key), value)
        for key, value in exclusion_expected.items()
        if exclusion_document.get(key) != value
    }
    if exclusion_drift:
        raise ShardValidationError(
            f"trace {trace.trace_idx}: exclusion manifest drifted: "
            f"{exclusion_drift}"
        )
    entries = exclusion_document.get("entries")
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        raise ShardValidationError(
            f"trace {trace.trace_idx}: exclusions must be an entry list"
        )
    expected_entry_sha = _canonical_sha256({"entries": entries})
    if exclusion_document.get("entries_sha256") != expected_entry_sha:
        raise ShardValidationError(
            f"trace {trace.trace_idx}: exclusion entry hash drifted"
        )
    if sidecar.get("exclusion_manifest_sha256") != sha256_file(
        exclusion_path
    ):
        raise ShardValidationError(
            f"trace {trace.trace_idx}: exclusion file hash drifted"
        )
    event_by_key = {event.key: event for event in trace.events}
    excluded_keys: list[str] = []
    allowed_reasons = {
        "tail_not_bit_exact",
        "boundary_unavailable",
        "tail_offsets_unavailable",
        "tokenizer_roundtrip_failed",
    }
    for entry in entries:
        key = str(entry.get("key"))
        event = event_by_key.get(key)
        if event is None:
            raise ShardValidationError(
                f"trace {trace.trace_idx}: exclusion key is not source eligible"
            )
        if key in excluded_keys:
            raise ShardValidationError(
                f"trace {trace.trace_idx}: duplicate exclusion key"
            )
        if (
            entry.get("key_sha256") != _event_key_sha256(key)
            or entry.get("trace_idx") != event.trace_idx
            or entry.get("sentence_idx") != event.sentence_idx
            or entry.get("label") != event.label
            or entry.get("category") != trace.category
            or entry.get("reason") not in allowed_reasons
        ):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: exclusion entry metadata drifted"
            )
        metrics = entry.get("metrics")
        if entry.get("reason") == "tail_not_bit_exact":
            if (
                not isinstance(metrics, dict)
                or int(metrics.get("bit_mismatched_values", 0)) < 1
                or int(metrics.get("compared_values", 0))
                != len(OFFICIAL_OFFSETS) * expected_width
            ):
                raise ShardValidationError(
                    f"trace {trace.trace_idx}: mismatch metrics are invalid"
                )
        elif metrics is not None:
            raise ShardValidationError(
                f"trace {trace.trace_idx}: unexpected exclusion metrics"
            )
        excluded_keys.append(key)
    expected_excluded_order = [
        event.key for event in trace.events if event.key in set(excluded_keys)
    ]
    if excluded_keys != expected_excluded_order:
        raise ShardValidationError(
            f"trace {trace.trace_idx}: exclusion key order drifted"
        )
    trace_exclusion_reason = exclusion_document.get(
        "trace_exclusion_reason"
    )
    if trace_exclusion_reason is None:
        if any(
            entry["reason"] == "tokenizer_roundtrip_failed"
            for entry in entries
        ):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: tokenizer exclusion lacks "
                "trace disposition"
            )
    elif trace_exclusion_reason == "tokenizer_roundtrip_failed":
        if excluded_keys != [event.key for event in trace.events] or any(
            entry["reason"] != trace_exclusion_reason for entry in entries
        ):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: tokenizer exclusion is not complete"
            )
    else:
        raise ShardValidationError(
            f"trace {trace.trace_idx}: unknown trace exclusion reason"
        )
    with np.load(tensor_path, allow_pickle=False) as payload:
        if set(payload.files) != _required_shard_arrays():
            raise ShardValidationError(
                f"trace {trace.trace_idx}: shard arrays differ from contract"
            )
        wide_x = payload["wide_X"]
        tail_x = payload["tail_only_X"]
        wide_keys = payload["wide_keys"].astype(str)
        tail_keys = payload["tail_only_keys"].astype(str)
        if wide_x.shape[1:] != (len(ARTIFACT_OFFSETS), expected_width):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: invalid wide shape {wide_x.shape}"
            )
        if tail_x.shape[1:] != (len(OFFICIAL_OFFSETS), expected_width):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: invalid tail-only shape {tail_x.shape}"
            )
        if wide_x.dtype != OUTPUT_DTYPE or tail_x.dtype != OUTPUT_DTYPE:
            raise ShardValidationError(
                f"trace {trace.trace_idx}: shard dtype is not float32"
            )
        for prefix, keys in (("wide", wide_keys), ("tail_only", tail_keys)):
            if len(payload[f"{prefix}_is_bt"]) != len(keys):
                raise ShardValidationError(
                    f"trace {trace.trace_idx}: {prefix} label count drifted"
                )
            if len(payload[f"{prefix}_boundary_token"]) != len(keys):
                raise ShardValidationError(
                    f"trace {trace.trace_idx}: {prefix} boundary count drifted"
                )
        observed_keys = [*wide_keys.tolist(), *tail_keys.tolist()]
        if len(observed_keys) != len(set(observed_keys)):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: duplicate event key in shard"
            )
        retained_keys = [
            event.key
            for event in trace.events
            if event.key not in set(excluded_keys)
        ]
        if set(observed_keys) != set(retained_keys):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: retained event key set drifted"
            )
        expected_wide_order = [
            key for key in retained_keys if key in set(wide_keys.tolist())
        ]
        expected_tail_order = [
            key for key in retained_keys if key in set(tail_keys.tolist())
        ]
        if (
            wide_keys.tolist() != expected_wide_order
            or tail_keys.tolist() != expected_tail_order
        ):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: retained key order drifted"
            )
        observed_labels = {
            key: int(label)
            for key, label in zip(
                wide_keys.tolist(),
                payload["wide_is_bt"].tolist(),
                strict=True,
            )
        }
        observed_labels.update(
            {
                key: int(label)
                for key, label in zip(
                    tail_keys.tolist(),
                    payload["tail_only_is_bt"].tolist(),
                    strict=True,
                )
            }
        )
        if any(
            observed_labels[event.key] != event.label
            for event in trace.events
            if event.key in observed_labels
        ):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: retained labels drifted"
            )
        if int(sidecar.get("wide_rows", -1)) != len(wide_keys):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: wide row count drifted"
            )
        if int(sidecar.get("tail_only_rows", -1)) != len(tail_keys):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: tail-only row count drifted"
            )
        if int(sidecar.get("source_eligible_rows", -1)) != len(trace.events):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: source row count drifted"
            )
        if int(sidecar.get("exact_tail_rows", -1)) != len(retained_keys):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: exact-tail count drifted"
            )
        if int(sidecar.get("excluded_rows", -1)) != len(excluded_keys):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: exclusion count drifted"
            )
        if len(trace.events) != len(retained_keys) + len(excluded_keys):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: cohort accounting drifted"
            )
        retained_events = [
            event for event in trace.events if event.key in set(retained_keys)
        ]
        retained_summary = _cohort_summary(
            [event.key for event in retained_events],
            [event.label for event in retained_events],
            [trace.category] * len(retained_events),
        )
        if sidecar.get("exact_tail_cohort") != retained_summary:
            raise ShardValidationError(
                f"trace {trace.trace_idx}: exact-tail cohort drifted"
            )
        if official_x_by_key is not None:
            wide_index = {
                key: index for index, key in enumerate(wide_keys.tolist())
            }
            tail_index = {
                key: index for index, key in enumerate(tail_keys.tolist())
            }
            for key in retained_keys:
                if key in wide_index:
                    tail = np.asarray(
                        wide_x[wide_index[key]][-len(OFFICIAL_OFFSETS) :],
                        dtype=OUTPUT_DTYPE,
                    )
                else:
                    tail = np.asarray(
                        tail_x[tail_index[key]],
                        dtype=OUTPUT_DTYPE,
                    )
                official_tail = np.asarray(
                    official_x_by_key[key],
                    dtype=OUTPUT_DTYPE,
                )
                if not _float32_bit_equal(tail, official_tail):
                    raise ExactTailError(
                        f"trace {trace.trace_idx}: retained tail is not "
                        "bit-exact during shard validation"
                    )
    return sidecar


def _request_payload(request: Request) -> dict[str, Any]:
    payload = asdict(request)
    payload["offsets"] = list(request.offsets)
    payload["trailing_offsets"] = list(request.trailing_offsets)
    return payload


def _write_or_validate_request(path: Path, request: Request) -> str:
    payload = _request_payload(request)
    request_sha = _canonical_sha256(payload)
    document = {**payload, "request_sha256": request_sha}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != document:
            raise ShardValidationError(
                "existing extraction request does not match supplied inputs"
            )
    else:
        _atomic_json(document, path)
    return request_sha


def _public_validation(validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in validation.items()
        if not str(key).startswith("_")
    }


def _prepare_output_directory(
    output_dir: Path,
    *,
    request: Request,
    validation: Mapping[str, Any],
) -> str:
    request_payload = _request_payload(request)
    request_sha = _canonical_sha256(request_payload)
    sentinel_path = output_dir / "schema.json"
    if output_dir.exists() and not sentinel_path.exists():
        existing = list(output_dir.iterdir())
        if existing:
            raise ShardValidationError(
                "refusing nonempty output without the v4 schema sentinel"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "request_schema": REQUEST_SCHEMA,
        "request_sha256": request_sha,
    }
    if sentinel_path.exists():
        if json.loads(sentinel_path.read_text(encoding="utf-8")) != sentinel:
            raise ShardValidationError("output schema sentinel drifted")
    else:
        _atomic_json(sentinel, sentinel_path)
    observed_request_sha = _write_or_validate_request(
        output_dir / "request.json",
        request,
    )
    if observed_request_sha != request_sha:
        raise ShardValidationError("request hash changed during output setup")

    source_entries = validation.get("_source_exclusion_entries")
    if not isinstance(source_entries, list):
        raise ShardValidationError("source exclusion entries are unavailable")
    source_exclusions = {
        "schema_version": EXCLUSION_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "request_sha256": request_sha,
        "kind": "source_alignment",
        "entries": source_entries,
        "entries_sha256": _canonical_sha256({"entries": source_entries}),
    }
    source_exclusion_path = output_dir / "source_exclusions.json"
    if source_exclusion_path.exists():
        if (
            json.loads(source_exclusion_path.read_text(encoding="utf-8"))
            != source_exclusions
        ):
            raise ShardValidationError(
                "source exclusion manifest changed during resume"
            )
    else:
        _atomic_json(source_exclusions, source_exclusion_path)
    return request_sha


def _load_inputs(
    *,
    prompts_path: Path,
    labels_path: Path,
    traces_path: Path,
    official_path: Path,
    traces_sha256: str,
    source_path: str,
    source_commit: str,
    strict_counts: bool = True,
) -> tuple[
    list[TraceSpec],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    SourcePin,
    dict[str, Any],
    dict[str, Any],
]:
    provenance = {
        "prompts": verify_locked_file(prompts_path, LOCKED_PROMPTS),
        "labels": verify_locked_file(labels_path, LOCKED_LABELS),
        "official": verify_locked_file(official_path, LOCKED_OFFICIAL),
    }
    source_pin = validate_source_pin(
        traces_path,
        expected_sha256=traces_sha256,
        source_path=source_path,
        source_commit=source_commit,
    )
    prompts = _load_json_array(prompts_path, name="prompts")
    labels = _load_json_array(labels_path, name="labels")
    traces = _load_json_array(traces_path, name="traces")
    with np.load(official_path, allow_pickle=True) as official:
        official_x = np.asarray(official["X"])
        official_labels = np.asarray(official["is_bt"], dtype=np.uint8)
        official_keys = np.asarray(official["keys"]).astype(str)
    if strict_counts and tuple(official_x.shape) != EXPECTED_OFFICIAL_SHAPE:
        raise ProvenanceError(
            f"official X shape {official_x.shape} != {EXPECTED_OFFICIAL_SHAPE}"
        )
    trace_specs, validation = validate_trace_records(
        prompts,
        labels,
        traces,
        official_keys=official_keys,
        official_labels=official_labels,
        strict_counts=strict_counts,
    )
    return (
        trace_specs,
        official_x,
        official_labels,
        official_keys,
        source_pin,
        provenance,
        validation,
    )


def _request(
    source: SourcePin,
    *,
    trace_count: int,
    sentence_count: int,
    official_rows: int,
    validation: Mapping[str, Any],
) -> Request:
    return Request(
        schema_version=REQUEST_SCHEMA,
        protocol_version=TEACHER_FORCE_PROTOCOL,
        source=source,
        prompts_sha256=LOCKED_PROMPTS.sha256,
        labels_sha256=LOCKED_LABELS.sha256,
        official_sha256=LOCKED_OFFICIAL.sha256,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=TOKENIZER_ID,
        tokenizer_revision=TOKENIZER_REVISION,
        layer=LAYER,
        component=COMPONENT,
        model_dtype=MODEL_DTYPE,
        output_dtype=str(OUTPUT_DTYPE),
        attention_implementation=ATTENTION_IMPLEMENTATION,
        add_special_tokens=TOKENIZATION_ADD_SPECIAL_TOKENS,
        output_schema=OUTPUT_SCHEMA,
        exclusion_schema=EXCLUSION_SCHEMA,
        tail_comparator=TAIL_COMPARATOR,
        trace_roundtrip_policy=TRACE_ROUNDTRIP_POLICY,
        event_selection_policy=EVENT_SELECTION_POLICY,
        category_field=CATEGORY_FIELD,
        offsets=ARTIFACT_OFFSETS,
        trailing_offsets=OFFICIAL_OFFSETS,
        trace_count=trace_count,
        sentence_count=sentence_count,
        official_rows=official_rows,
        source_eligible_rows=int(validation["source_eligible_rows"]),
        source_eligible_sha256=str(
            validation["source_eligible_sha256"]
        ),
        source_exclusions_sha256=str(
            validation["source_exclusions_sha256"]
        ),
    )


def _trace_paths(
    output_dir: Path,
    trace_idx: int,
) -> tuple[Path, Path, Path]:
    tensor = output_dir / "shards" / f"trace_{trace_idx:05d}.npz"
    return (
        tensor,
        tensor.with_suffix(".manifest.json"),
        output_dir / "exclusions" / f"trace_{trace_idx:05d}.json",
    )


def _trace_exclusion_document(
    trace: TraceSpec,
    *,
    request_sha256: str,
    entries: Sequence[Mapping[str, Any]],
    trace_exclusion_reason: str | None,
) -> dict[str, Any]:
    ordered_entries = [dict(entry) for entry in entries]
    reason_counts: dict[str, int] = {}
    for entry in ordered_entries:
        reason = str(entry["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "schema_version": EXCLUSION_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "request_sha256": request_sha256,
        "question_id": trace.question_id,
        "trace_idx": trace.trace_idx,
        "category": trace.category,
        "trace_input_sha256": _trace_input_sha256(trace),
        "trace_exclusion_reason": trace_exclusion_reason,
        "entries": ordered_entries,
        "entries_sha256": _canonical_sha256(
            {"entries": ordered_entries}
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _remove_incomplete_trace_outcome(
    paths: Sequence[Path],
    *,
    trace_idx: int,
) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing or len(existing) == len(paths):
        return
    for path in existing:
        path.unlink()
    print(
        f"[ward-t16-extract] trace={trace_idx} "
        "recomputed_incomplete_outcome=1",
        flush=True,
    )


def run_extract(
    *,
    prompts_path: Path,
    labels_path: Path,
    traces_path: Path,
    official_path: Path,
    output_dir: Path,
    traces_sha256: str,
    source_path: str,
    source_commit: str,
    device: str,
    num_shards: int,
    shard_index: int,
    max_traces: int | None = None,
) -> dict[str, Any]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    (
        traces,
        official_x,
        official_labels,
        official_keys,
        source_pin,
        _provenance,
        validation,
    ) = _load_inputs(
        prompts_path=prompts_path,
        labels_path=labels_path,
        traces_path=traces_path,
        official_path=official_path,
        traces_sha256=traces_sha256,
        source_path=source_path,
        source_commit=source_commit,
    )
    request = _request(
        source_pin,
        trace_count=len(traces),
        sentence_count=validation["sentence_count"],
        official_rows=len(official_keys),
        validation=validation,
    )
    request_sha = _prepare_output_directory(
        output_dir,
        request=request,
        validation=validation,
    )
    official_x_by_key = {
        key: official_x[index]
        for index, key in enumerate(official_keys.tolist())
    }
    selected = [
        trace for trace in traces if trace.trace_idx % num_shards == shard_index
    ]
    if max_traces is not None:
        if max_traces < 1:
            raise ValueError("--max-traces must be positive")
        selected = selected[:max_traces]
    pending: list[tuple[TraceSpec, Path, Path, Path]] = []
    for trace in selected:
        tensor_path, sidecar_path, exclusion_path = _trace_paths(
            output_dir,
            trace.trace_idx,
        )
        _remove_incomplete_trace_outcome(
            (tensor_path, sidecar_path, exclusion_path),
            trace_idx=trace.trace_idx,
        )
        if (
            _validate_trace_shard(
                tensor_path,
                sidecar_path,
                exclusion_path,
                trace=trace,
                request_sha256=request_sha,
                official_x_by_key=official_x_by_key,
            )
            is None
        ):
            pending.append(
                (trace, tensor_path, sidecar_path, exclusion_path)
            )
    print(
        f"[ward-t16-extract] shard={shard_index}/{num_shards} "
        f"selected_traces={len(selected)} pending_traces={len(pending)}",
        flush=True,
    )
    if not pending:
        return {
            "status": "complete",
            "request_sha256": request_sha,
            "selected_traces": len(selected),
            "new_trace_shards": 0,
        }

    model, tokenizer, layer_module, runtime = _model_and_tokenizer(device)
    runtime_document = {
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "request_sha256": request_sha,
        **runtime,
    }
    runtime_path = output_dir / f"runtime_shard_{shard_index:03d}.json"
    if runtime_path.exists():
        if json.loads(runtime_path.read_text(encoding="utf-8")) != runtime_document:
            raise ShardValidationError("runtime provenance changed during resume")
    else:
        _atomic_json(runtime_document, runtime_path)

    completed = 0
    for trace, tensor_path, sidecar_path, exclusion_path in pending:
        trace_exclusion_reason: str | None = None
        forward_performed = False
        token_count: int | None = None
        hidden: np.ndarray | None = None
        try:
            tokenized = tokenize_full_response(
                tokenizer,
                trace.full_response,
            )
        except TokenizerRoundTripError:
            arrays = _empty_trace_arrays()
            trace_exclusion_reason = "tokenizer_roundtrip_failed"
            exclusions = [
                _event_exclusion(
                    event,
                    category=trace.category,
                    reason=trace_exclusion_reason,
                )
                for event in trace.events
            ]
            comparison = {
                "comparator": TAIL_COMPARATOR,
                "source_eligible_rows": len(trace.events),
                "tail_compared_rows": 0,
                "exact_tail_rows": 0,
                "nonexact_tail_rows": 0,
                "unavailable_tail_rows": 0,
                "wide_rows": 0,
                "insufficient_t16_context_rows": 0,
                "exact_tail_cohort": _cohort_summary([], [], []),
                "nonexact_tail_metrics": {
                    "events": 0,
                    "compared_values": 0,
                    "bit_mismatched_values": 0,
                    "finite_pairs": 0,
                    "nonfinite_pairs": 0,
                    "max_abs": None,
                    "mean_abs": None,
                    "rmse": None,
                },
                "exclusions": exclusions,
            }
        else:
            token_count = len(tokenized["input_ids"])
            if token_count > runtime["max_position_embeddings"]:
                raise TraceAlignmentError(
                    f"trace {trace.trace_idx}: token count exceeds "
                    "the pinned model context"
                )
            hidden = _forward_trace(
                model,
                layer_module,
                input_ids=tokenized["input_ids"],
                device=device,
            )
            forward_performed = True
            arrays, comparison = partition_trace_activations(
                hidden,
                trace=trace,
                offsets=tokenized["offsets"],
                official_x_by_key=official_x_by_key,
            )
        exclusion_document = _trace_exclusion_document(
            trace,
            request_sha256=request_sha,
            entries=comparison["exclusions"],
            trace_exclusion_reason=trace_exclusion_reason,
        )
        _atomic_json(exclusion_document, exclusion_path)
        _atomic_npz(tensor_path, **arrays)
        wide_keys = arrays["wide_keys"].astype(str).tolist()
        wide_labels = arrays["wide_is_bt"].astype(np.uint8).tolist()
        wide_cohort = _cohort_summary(
            wide_keys,
            wide_labels,
            [trace.category] * len(wide_keys),
        )
        sidecar = {
            "schema_version": SHARD_SCHEMA,
            "protocol_version": TEACHER_FORCE_PROTOCOL,
            "request_sha256": request_sha,
            "question_id": trace.question_id,
            "trace_idx": trace.trace_idx,
            "category": trace.category,
            "full_response_sha256": trace.full_response_sha256,
            "trace_input_sha256": _trace_input_sha256(trace),
            "thinking_char_start": trace.thinking_char_start,
            "thinking_char_end": trace.thinking_char_end,
            "label_sentence_count": trace.label_sentence_count,
            "label_text_matches": trace.label_text_matches,
            "token_count": token_count,
            "forward_performed": forward_performed,
            "trace_exclusion_reason": trace_exclusion_reason,
            "tail_comparator": TAIL_COMPARATOR,
            "source_eligible_rows": len(trace.events),
            "tail_compared_rows": comparison["tail_compared_rows"],
            "exact_tail_rows": comparison["exact_tail_rows"],
            "excluded_rows": len(comparison["exclusions"]),
            "wide_rows": len(arrays["wide_keys"]),
            "tail_only_rows": len(arrays["tail_only_keys"]),
            "exact_tail_cohort": comparison["exact_tail_cohort"],
            "wide_cohort": wide_cohort,
            "nonexact_tail_metrics": comparison[
                "nonexact_tail_metrics"
            ],
            "tail_exact_for_retained_events": True,
            "all_source_events_exact": not comparison["exclusions"],
            "exclusion_manifest_sha256": sha256_file(exclusion_path),
            "array_shapes": {
                name: list(array.shape) for name, array in arrays.items()
            },
            "array_dtypes": {
                name: str(array.dtype) for name, array in arrays.items()
            },
            "sha256": sha256_file(tensor_path),
            "size": tensor_path.stat().st_size,
        }
        _atomic_json(sidecar, sidecar_path)
        _validate_trace_shard(
            tensor_path,
            sidecar_path,
            exclusion_path,
            trace=trace,
            request_sha256=request_sha,
            official_x_by_key=official_x_by_key,
        )
        completed += 1
        print(
            f"[ward-t16-extract] trace={trace.trace_idx} "
            f"source={sidecar['source_eligible_rows']} "
            f"exact={sidecar['exact_tail_rows']} "
            f"wide={sidecar['wide_rows']} "
            f"excluded={sidecar['excluded_rows']} "
            f"trace_excluded={int(trace_exclusion_reason is not None)} "
            f"complete={completed}/{len(pending)}",
            flush=True,
        )
        del arrays
        if hidden is not None:
            del hidden
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    return {
        "status": "complete",
        "request_sha256": request_sha,
        "selected_traces": len(selected),
        "new_trace_shards": completed,
    }


def _manifest_locked_file(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key
        in {
            "repo_id",
            "revision",
            "filename",
            "sha256",
            "verified_sha256",
            "size",
            "verified_size",
            "local_path",
        }
    }


def _coverage_gate(
    cohort_name: str,
    summary: Mapping[str, Any],
    *,
    official_summary: Mapping[str, Any],
) -> dict[str, Any]:
    class_counts = summary["class_counts"]
    missing_classes = [
        label for label in ("0", "1") if int(class_counts.get(label, 0)) < 1
    ]
    required_categories = sorted(
        official_summary["category_counts"].keys()
    )
    observed_categories = summary["category_counts"]
    missing_categories = [
        category
        for category in required_categories
        if int(observed_categories.get(category, 0)) < 1
    ]
    missing_category_class_cells: list[str] = []
    for category, official_cells in sorted(
        official_summary["category_class_counts"].items()
    ):
        observed_cells = summary["category_class_counts"].get(
            category,
            {},
        )
        for label in ("0", "1"):
            if (
                int(official_cells.get(label, 0)) > 0
                and int(observed_cells.get(label, 0)) < 1
            ):
                missing_category_class_cells.append(
                    f"{category}|{label}"
                )
    passed = not (
        missing_classes
        or missing_categories
        or missing_category_class_cells
    )
    return {
        "cohort": cohort_name,
        "passed": passed,
        "missing_classes": missing_classes,
        "missing_categories": missing_categories,
        "missing_category_class_cells": missing_category_class_cells,
    }


def _require_coverage(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    official_summary: Mapping[str, Any],
) -> dict[str, Any]:
    report = {
        name: _coverage_gate(
            name,
            summary,
            official_summary=official_summary,
        )
        for name, summary in summaries.items()
    }
    failures = {
        name: gate for name, gate in report.items() if not gate["passed"]
    }
    if failures:
        raise ShardValidationError(
            "cohort class/category coverage gate failed: "
            + json.dumps(failures, sort_keys=True)
        )
    return report


def _aggregate_nonexact_metrics(
    sidecars: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    events = 0
    compared_values = 0
    bit_mismatched_values = 0
    finite_pairs = 0
    nonfinite_pairs = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    maxima: list[float] = []
    for sidecar in sidecars:
        metrics = sidecar["nonexact_tail_metrics"]
        events += int(metrics["events"])
        compared_values += int(metrics["compared_values"])
        bit_mismatched_values += int(metrics["bit_mismatched_values"])
        local_finite = int(metrics["finite_pairs"])
        finite_pairs += local_finite
        nonfinite_pairs += int(metrics["nonfinite_pairs"])
        if local_finite:
            absolute_sum += float(metrics["mean_abs"]) * local_finite
            squared_sum += float(metrics["rmse"]) ** 2 * local_finite
            maxima.append(float(metrics["max_abs"]))
    return {
        "events": events,
        "compared_values": compared_values,
        "bit_mismatched_values": bit_mismatched_values,
        "finite_pairs": finite_pairs,
        "nonfinite_pairs": nonfinite_pairs,
        "max_abs": max(maxima) if maxima else None,
        "mean_abs": absolute_sum / finite_pairs if finite_pairs else None,
        "rmse": (
            float(np.sqrt(squared_sum / finite_pairs))
            if finite_pairs
            else None
        ),
    }


def run_assemble(
    *,
    prompts_path: Path,
    labels_path: Path,
    traces_path: Path,
    official_path: Path,
    output_dir: Path,
    artifact_path: Path,
    manifest_path: Path,
    traces_sha256: str,
    source_path: str,
    source_commit: str,
) -> dict[str, Any]:
    (
        traces,
        official_x,
        official_labels,
        official_keys,
        source_pin,
        provenance,
        input_validation,
    ) = _load_inputs(
        prompts_path=prompts_path,
        labels_path=labels_path,
        traces_path=traces_path,
        official_path=official_path,
        traces_sha256=traces_sha256,
        source_path=source_path,
        source_commit=source_commit,
    )
    request = _request(
        source_pin,
        trace_count=len(traces),
        sentence_count=input_validation["sentence_count"],
        official_rows=len(official_keys),
        validation=input_validation,
    )
    request_sha = _prepare_output_directory(
        output_dir,
        request=request,
        validation=input_validation,
    )
    if artifact_path.exists() != manifest_path.exists():
        raise ShardValidationError(
            "final artifact and manifest must either both exist or both be absent"
        )
    existing_final = artifact_path.exists()

    official_x_by_key = {
        key: official_x[index]
        for index, key in enumerate(official_keys.tolist())
    }
    sidecars: dict[int, dict[str, Any]] = {}
    exclusion_documents: dict[int, dict[str, Any]] = {}
    source_keys: list[str] = []
    source_labels: list[int] = []
    source_categories: list[str] = []
    exact_keys: list[str] = []
    exact_labels: list[int] = []
    exact_categories: list[str] = []
    wide_keys_for_gate: list[str] = []
    wide_labels_for_gate: list[int] = []
    wide_categories_for_gate: list[str] = []
    extraction_exclusions: list[dict[str, Any]] = []
    trace_summary: list[dict[str, Any]] = []
    for trace in traces:
        tensor_path, sidecar_path, exclusion_path = _trace_paths(
            output_dir,
            trace.trace_idx,
        )
        sidecar = _validate_trace_shard(
            tensor_path,
            sidecar_path,
            exclusion_path,
            trace=trace,
            request_sha256=request_sha,
            official_x_by_key=official_x_by_key,
        )
        if sidecar is None:
            raise ShardValidationError(
                f"trace {trace.trace_idx}: extraction shard is missing"
            )
        sidecars[trace.trace_idx] = sidecar
        exclusion_document = json.loads(
            exclusion_path.read_text(encoding="utf-8")
        )
        exclusion_documents[trace.trace_idx] = exclusion_document
        excluded_set = {
            str(entry["key"]) for entry in exclusion_document["entries"]
        }
        retained_events = [
            event for event in trace.events if event.key not in excluded_set
        ]
        source_keys.extend(event.key for event in trace.events)
        source_labels.extend(event.label for event in trace.events)
        source_categories.extend([trace.category] * len(trace.events))
        exact_keys.extend(event.key for event in retained_events)
        exact_labels.extend(event.label for event in retained_events)
        exact_categories.extend([trace.category] * len(retained_events))
        with np.load(tensor_path, allow_pickle=False) as payload:
            shard_wide_keys = payload["wide_keys"].astype(str).tolist()
            shard_wide_labels = (
                payload["wide_is_bt"].astype(np.uint8).tolist()
            )
        wide_keys_for_gate.extend(shard_wide_keys)
        wide_labels_for_gate.extend(shard_wide_labels)
        wide_categories_for_gate.extend(
            [trace.category] * len(shard_wide_keys)
        )
        extraction_exclusions.extend(exclusion_document["entries"])
        trace_summary.append(
            {
                "trace_idx": trace.trace_idx,
                "category": trace.category,
                "source_eligible_rows": len(trace.events),
                "exact_tail_rows": len(retained_events),
                "wide_rows": len(shard_wide_keys),
                "source_class_counts": _cohort_counts(
                    [event.label for event in trace.events],
                    [trace.category] * len(trace.events),
                )["class_counts"],
                "exact_tail_class_counts": _cohort_counts(
                    [event.label for event in retained_events],
                    [trace.category] * len(retained_events),
                )["class_counts"],
                "wide_class_counts": _cohort_counts(
                    shard_wide_labels,
                    [trace.category] * len(shard_wide_labels),
                )["class_counts"],
                "trace_exclusion_reason": exclusion_document[
                    "trace_exclusion_reason"
                ],
                "exclusion_reason_counts": exclusion_document[
                    "reason_counts"
                ],
            }
        )
    source_summary = _cohort_summary(
        source_keys,
        source_labels,
        source_categories,
    )
    exact_summary = _cohort_summary(
        exact_keys,
        exact_labels,
        exact_categories,
    )
    wide_summary = _cohort_summary(
        wide_keys_for_gate,
        wide_labels_for_gate,
        wide_categories_for_gate,
    )
    if (
        source_summary["rows"] != input_validation["source_eligible_rows"]
        or source_summary["sha256"]
        != input_validation["source_eligible_sha256"]
    ):
        raise ShardValidationError("source-eligible cohort drifted at assembly")
    coverage = _require_coverage(
        {
            "source_eligible": source_summary,
            "exact_tail_retained": exact_summary,
            "wide_t16": wide_summary,
        },
        official_summary=input_validation["official_cohort"],
    )
    wide_rows = int(wide_summary["rows"])
    if wide_rows < 1:
        raise ShardValidationError("no official event supports all T16 offsets")

    existing_manifest: dict[str, Any] | None = None
    existing_artifact_sha: str | None = None
    if existing_final:
        existing_manifest, existing_artifact_sha = (
            _validate_existing_final_output(
                artifact_path,
                manifest_path,
                output_dir=output_dir,
                traces=traces,
                request_sha256=request_sha,
                expected_keys=wide_keys_for_gate,
                expected_labels=wide_labels_for_gate,
            )
        )

    temporary_x = artifact_path.with_name(
        f".{artifact_path.name}.assembling.npy"
    )
    temporary_x.parent.mkdir(parents=True, exist_ok=True)
    wide_x = None
    if not existing_final:
        wide_x = np.lib.format.open_memmap(
            temporary_x,
            mode="w+",
            dtype=OUTPUT_DTYPE,
            shape=(wide_rows, len(ARTIFACT_OFFSETS), EXPECTED_WIDTH),
        )
    wide_labels = np.empty(wide_rows, dtype=np.uint8)
    wide_keys: list[str] = []
    official_index = {
        key: index for index, key in enumerate(official_keys.tolist())
    }
    source_cursor = 0
    exact_cursor = 0
    previous_official_index = -1
    wide_cursor = 0
    try:
        for trace in traces:
            tensor_path, _, _ = _trace_paths(output_dir, trace.trace_idx)
            with np.load(tensor_path, allow_pickle=False) as payload:
                shard_wide_x = payload["wide_X"]
                shard_tail_x = payload["tail_only_X"]
                shard_wide_keys = payload["wide_keys"].astype(str)
                shard_tail_keys = payload["tail_only_keys"].astype(str)
                wide_index = {
                    key: index
                    for index, key in enumerate(shard_wide_keys.tolist())
                }
                tail_index = {
                    key: index
                    for index, key in enumerate(shard_tail_keys.tolist())
                }
                excluded_set = {
                    str(entry["key"])
                    for entry in exclusion_documents[trace.trace_idx][
                        "entries"
                    ]
                }
                for event in trace.events:
                    if source_cursor >= source_summary["rows"]:
                        raise ShardValidationError(
                            "extracted more source-eligible events than expected"
                        )
                    row_index = official_index.get(event.key)
                    if row_index is None:
                        raise ShardValidationError(
                            f"source-eligible key {event.key!r} is not official"
                        )
                    if row_index <= previous_official_index:
                        raise ShardValidationError(
                            f"source-eligible official key order drift at "
                            f"{event.key!r}"
                        )
                    if event.label != int(official_labels[row_index]):
                        raise ShardValidationError(
                            f"official label drift for {event.key!r}"
                        )
                    source_cursor += 1
                    if event.key in excluded_set:
                        continue
                    if event.key in wide_index:
                        row = np.asarray(
                            shard_wide_x[wide_index[event.key]],
                            dtype=OUTPUT_DTYPE,
                        )
                        tail = row[-len(OFFICIAL_OFFSETS) :]
                        if wide_x is not None:
                            wide_x[wide_cursor] = row
                        wide_labels[wide_cursor] = event.label
                        wide_keys.append(event.key)
                        wide_cursor += 1
                    elif event.key in tail_index:
                        tail = np.asarray(
                            shard_tail_x[tail_index[event.key]],
                            dtype=OUTPUT_DTYPE,
                        )
                    else:
                        raise ShardValidationError(
                            f"trace {trace.trace_idx}: exact event is missing"
                        )
                    official_tail = np.asarray(
                        official_x[row_index],
                        dtype=OUTPUT_DTYPE,
                    )
                    if not _float32_bit_equal(tail, official_tail):
                        raise ExactTailError(
                            f"assembly retained-tail mismatch at official row "
                            f"{row_index}"
                        )
                    previous_official_index = row_index
                    exact_cursor += 1
        if source_cursor != source_summary["rows"]:
            raise ShardValidationError(
                f"assembled {source_cursor} source-eligible rows, "
                f"expected {source_summary['rows']}"
            )
        if exact_cursor != exact_summary["rows"]:
            raise ShardValidationError(
                f"assembled {exact_cursor} exact-tail rows, "
                f"expected {exact_summary['rows']}"
            )
        if wide_cursor != wide_rows:
            raise ShardValidationError(
                f"assembled {wide_cursor} wide rows, expected {wide_rows}"
            )
        if wide_x is not None:
            wide_x.flush()
            key_array = np.asarray(wide_keys)
            _atomic_npz(
                artifact_path,
                X=wide_x,
                is_bt=wide_labels,
                keys=key_array,
                offsets=np.asarray(ARTIFACT_OFFSETS, dtype=np.int32),
            )
    finally:
        if wide_x is not None:
            del wide_x
        if temporary_x.exists():
            temporary_x.unlink()

    key_array = np.asarray(wide_keys)
    common_hash = cohort_sha256(key_array, wide_labels)
    if (
        wide_keys != wide_keys_for_gate
        or wide_labels.tolist() != wide_labels_for_gate
        or common_hash != wide_summary["sha256"]
    ):
        raise ShardValidationError("wide cohort changed while writing artifact")
    artifact_sha = (
        existing_artifact_sha
        if existing_artifact_sha is not None
        else sha256_file(artifact_path)
    )
    extraction_exclusions_sha = _canonical_sha256(
        {"entries": extraction_exclusions}
    )
    source_exclusions_sha = str(
        input_validation["source_exclusions_sha256"]
    )
    combined_exclusions_sha = _canonical_sha256(
        {
            "source_exclusions_sha256": source_exclusions_sha,
            "extraction_entries": extraction_exclusions,
        }
    )
    aggregate_exclusion_document = {
        "schema_version": EXCLUSION_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "request_sha256": request_sha,
        "kind": "tokenizer_and_tail",
        "entries": extraction_exclusions,
        "entries_sha256": extraction_exclusions_sha,
        "source_exclusions_sha256": source_exclusions_sha,
        "combined_exclusions_sha256": combined_exclusions_sha,
    }
    aggregate_exclusion_path = output_dir / "extraction_exclusions.json"
    if aggregate_exclusion_path.exists():
        if (
            json.loads(
                aggregate_exclusion_path.read_text(encoding="utf-8")
            )
            != aggregate_exclusion_document
        ):
            raise ShardValidationError(
                "aggregate extraction exclusions changed during resume"
            )
    else:
        _atomic_json(
            aggregate_exclusion_document,
            aggregate_exclusion_path,
        )
    category_trace_counts: dict[str, dict[str, int]] = {}
    for row in trace_summary:
        category = str(row["category"])
        counts = category_trace_counts.setdefault(
            category,
            {
                "traces": 0,
                "traces_with_source_eligible": 0,
                "traces_with_exact_tail": 0,
                "traces_with_wide_t16": 0,
                "tokenizer_excluded_traces": 0,
            },
        )
        counts["traces"] += 1
        counts["traces_with_source_eligible"] += int(
            row["source_eligible_rows"] > 0
        )
        counts["traces_with_exact_tail"] += int(
            row["exact_tail_rows"] > 0
        )
        counts["traces_with_wide_t16"] += int(row["wide_rows"] > 0)
        counts["tokenizer_excluded_traces"] += int(
            row["trace_exclusion_reason"]
            == "tokenizer_roundtrip_failed"
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "status": "complete",
        "request_sha256": request_sha,
        "offsets": list(ARTIFACT_OFFSETS),
        "dtype": str(OUTPUT_DTYPE),
        "source": {
            "path": source_pin.path,
            "local_path": str(traces_path.resolve()),
            "sha256": source_pin.sha256,
            "verified_sha256": source_pin.sha256,
            "commit": source_pin.commit,
            "field": "full_response",
        },
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "tokenizer": {
            "id": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "add_special_tokens": TOKENIZATION_ADD_SPECIAL_TOKENS,
        },
        "activation": {
            "layer": LAYER,
            "component": COMPONENT,
            "hook": "transformer block output",
            "model_dtype": MODEL_DTYPE,
            "output_dtype": str(OUTPUT_DTYPE),
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
        "provenance": {
            "prompts": _manifest_locked_file(provenance["prompts"]),
            "labels": _manifest_locked_file(provenance["labels"]),
            "official": _manifest_locked_file(provenance["official"]),
        },
        "official_artifact": {
            **_manifest_locked_file(provenance["official"]),
            "shape": list(official_x.shape),
        },
        "common_cohort_sha256": common_hash,
        "common_cohort": {
            **wide_summary,
            "exact_key_order": True,
            "selection": EVENT_SELECTION_POLICY,
        },
        "exact_key_order": True,
        "trailing_six": {
            "comparison": "exact_keyed_join",
            "comparator": TAIL_COMPARATOR,
            "keyed": True,
            "offsets": list(OFFICIAL_OFFSETS),
            "matched_keys": wide_rows,
            "source_eligible_rows": source_summary["rows"],
            "tail_compared_rows": sum(
                int(sidecar["tail_compared_rows"])
                for sidecar in sidecars.values()
            ),
            "exact_tail_retained_rows": exact_summary["rows"],
            "exact_equal": True,
            "max_abs": 0.0,
            "mismatched_values": 0,
        },
        "validation": {
            **_public_validation(input_validation),
            "source_eligible_cohort": source_summary,
            "exact_tail_retained_cohort": exact_summary,
            "wide_t16_cohort": wide_summary,
            "coverage": coverage,
            "wide_rows": wide_rows,
            "wide_rows_dropped_for_missing_early_offsets": (
                exact_summary["rows"] - wide_rows
            ),
            "source_rows_excluded_by_tokenizer_or_tail": (
                source_summary["rows"] - exact_summary["rows"]
            ),
            "complete_trace_shards": len(sidecars),
            "key_label_order_match": True,
            "retained_trailing_six_exact_equal": True,
            "nonexact_tail_metrics": _aggregate_nonexact_metrics(
                list(sidecars.values())
            ),
            "source_exclusions_sha256": source_exclusions_sha,
            "extraction_exclusions_sha256": extraction_exclusions_sha,
            "combined_exclusions_sha256": combined_exclusions_sha,
            "extraction_exclusion_rows": len(extraction_exclusions),
            "extraction_exclusion_manifest": {
                "path": str(aggregate_exclusion_path.resolve()),
                "size": aggregate_exclusion_path.stat().st_size,
                "sha256": sha256_file(aggregate_exclusion_path),
            },
            "trace_counts_by_category": dict(
                sorted(category_trace_counts.items())
            ),
            "trace_summary": trace_summary,
        },
        "output": {
            "path": str(artifact_path.resolve()),
            "size": artifact_path.stat().st_size,
            "sha256": artifact_sha,
            "shape": [
                wide_rows,
                len(ARTIFACT_OFFSETS),
                EXPECTED_WIDTH,
            ],
            "dtype": str(OUTPUT_DTYPE),
            "offsets": list(ARTIFACT_OFFSETS),
            "exact_key_order": True,
        },
    }
    if existing_manifest is not None:
        if existing_manifest != manifest:
            raise ShardValidationError(
                "existing final manifest differs from recomputed v4 contract"
            )
    else:
        _atomic_json(manifest, manifest_path)
    return manifest


def run_preflight(
    *,
    prompts_path: Path,
    labels_path: Path,
    traces_path: Path,
    official_path: Path,
    traces_sha256: str,
    source_path: str,
    source_commit: str,
) -> dict[str, Any]:
    (
        traces,
        _official_x,
        _official_labels,
        official_keys,
        source_pin,
        _provenance,
        validation,
    ) = _load_inputs(
        prompts_path=prompts_path,
        labels_path=labels_path,
        traces_path=traces_path,
        official_path=official_path,
        traces_sha256=traces_sha256,
        source_path=source_path,
        source_commit=source_commit,
    )
    request = _request(
        source_pin,
        trace_count=len(traces),
        sentence_count=validation["sentence_count"],
        official_rows=len(official_keys),
        validation=validation,
    )
    return {
        "status": "preflight-complete",
        "request": _request_payload(request),
        "request_sha256": _canonical_sha256(_request_payload(request)),
        "validation": _public_validation(validation),
        "gpu_loaded": False,
    }


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--traces-sha256", required=True)
    parser.add_argument(
        "--source-path",
        required=True,
        help="Repository-relative path of the explicitly supplied traces file",
    )
    parser.add_argument("--source-commit", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser(
        "preflight",
        help="validate all pinned joins without loading the model",
    )
    _add_common_inputs(preflight)

    extract = commands.add_parser(
        "extract",
        help="write resumable per-trace activation shards",
    )
    _add_common_inputs(extract)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--num-shards", type=int, default=1)
    extract.add_argument("--shard-index", type=int, default=0)
    extract.add_argument(
        "--max-traces",
        type=int,
        help="GPU smoke only: process at most N traces assigned to this shard",
    )

    assemble = commands.add_parser(
        "assemble",
        help="require every trace shard and emit the complete T16 artifact",
    )
    _add_common_inputs(assemble)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--artifact", type=Path, required=True)
    assemble.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    common = {
        "prompts_path": args.prompts,
        "labels_path": args.labels,
        "traces_path": args.traces,
        "official_path": args.official,
        "traces_sha256": args.traces_sha256,
        "source_path": args.source_path,
        "source_commit": args.source_commit,
    }
    if args.command == "preflight":
        result = run_preflight(**common)
    elif args.command == "extract":
        result = run_extract(
            **common,
            output_dir=args.output_dir,
            device=args.device,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            max_traces=args.max_traces,
        )
    else:
        result = run_assemble(
            **common,
            output_dir=args.output_dir,
            artifact_path=args.artifact,
            manifest_path=args.manifest,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
