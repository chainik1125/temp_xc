"""Teacher-force the pinned Ward traces into an exact T16 event artifact.

This builder never discovers or fetches source traces.  The caller must supply
one ``traces.json`` file together with its expected SHA-256, repository path,
and source commit.  The file is joined to the pinned prompts, sentence labels,
and official six-offset artifact by ``question_id``, ``trace_idx``, and event
key before a model is loaded.

Extraction is resumable at one file per trace.  Every extracted event's
offsets ``-13..-8`` are compared bit-for-bit against the official artifact
before its shard is committed.  Assembly repeats that keyed comparison over
the complete official cohort, then emits only events for which all offsets
``-23..-8`` exist.
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
    EXPECTED_PROMPTS,
    EXPECTED_SENTENCES,
    LOCKED_LABELS,
    LOCKED_OFFICIAL,
    LOCKED_PROMPTS,
    OFFICIAL_OFFSETS,
    ProvenanceError,
    LockedFile,
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
TOKENIZATION_ADD_SPECIAL_TOKENS = True
REQUEST_SCHEMA = "ward-c7-wide-teacher-force-request.v1"
SHARD_SCHEMA = "ward-c7-wide-teacher-force-shard.v1"
OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SourceValidationError(RuntimeError):
    """The supplied trace source does not match its explicit provenance."""


class TraceAlignmentError(RuntimeError):
    """A raw trace cannot be joined unambiguously to the locked labels."""


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
    offsets: tuple[int, ...]
    trailing_offsets: tuple[int, ...]
    trace_count: int
    sentence_count: int
    official_rows: int


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    sentences: Sequence[Mapping[str, Any]],
    *,
    trace_idx: int,
) -> tuple[int, int, int]:
    """Resolve whether label offsets include leading whitespace after <think>.

    Public labels contain gaps and a minority of conflicting overlapping
    annotations, so requiring every sentence string to agree would reject a
    source that the official keyed tail can still verify.  We instead choose
    uniquely among the explicit tag boundary and its leading-whitespace
    prefixes by the number of exact label-string matches.  The final
    activation comparison remains the authoritative alignment proof.
    """

    if full_response.count(OPEN_TAG) != 1:
        raise TraceAlignmentError(
            f"trace {trace_idx}: full_response must contain exactly one {OPEN_TAG!r}"
        )
    open_end = full_response.index(OPEN_TAG) + len(OPEN_TAG)
    close_start = full_response.find(CLOSE_TAG, open_end)
    if close_start < 0 or full_response.find(CLOSE_TAG, close_start + 1) >= 0:
        raise TraceAlignmentError(
            f"trace {trace_idx}: full_response must contain exactly one "
            f"{CLOSE_TAG!r} after {OPEN_TAG!r}"
        )

    candidates = [open_end]
    cursor = open_end
    while cursor < close_start and full_response[cursor].isspace():
        cursor += 1
        candidates.append(cursor)
    scored: list[tuple[int, int]] = []
    for candidate in candidates:
        matches = 0
        valid = True
        for sentence in sentences:
            start = int(sentence["char_start"])
            end = int(sentence["char_end"])
            text = sentence["sentence"]
            if start < 0 or end < start or candidate + end > close_start:
                valid = False
                break
            matches += full_response[candidate + start : candidate + end] == text
        if valid:
            scored.append((matches, candidate))
    if not scored:
        raise TraceAlignmentError(
            f"trace {trace_idx}: label spans exceed the thinking region"
        )
    best_score = max(score for score, _ in scored)
    best = [candidate for score, candidate in scored if score == best_score]
    if len(best) != 1:
        raise TraceAlignmentError(
            f"trace {trace_idx}: thinking-region base is ambiguous across {best}"
        )
    if sentences and best_score == 0:
        raise TraceAlignmentError(
            f"trace {trace_idx}: no locked sentence string agrees with "
            "full_response at any tag-anchored offset"
        )
    return best[0], close_start, best_score


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
            trace_idx = int(trace["trace_idx"])
        except (KeyError, TypeError, ValueError) as error:
            raise TraceAlignmentError(
                f"record {row_index}: trace_idx must be an integer"
            ) from error
        if label_trace_idx != row_index or trace_idx != row_index:
            raise TraceAlignmentError(
                f"record {row_index}: locked and raw trace_idx must equal row order"
            )
        if trace.get("question_id") != question_id:
            raise TraceAlignmentError(
                f"record {row_index}: raw question_id "
                f"{trace.get('question_id')!r} != {question_id!r}"
            )
        full_response = trace.get("full_response")
        if not isinstance(full_response, str) or not full_response:
            raise TraceAlignmentError(
                f"record {row_index}: full_response must be nonempty text"
            )
        sentences = label_row.get("sentences")
        if not isinstance(sentences, list):
            raise TraceAlignmentError(
                f"record {row_index}: sentences must be a list"
            )
        parsed_sentences = [
            _validate_sentence(
                sentence,
                trace_idx=row_index,
                sentence_idx=sentence_idx,
            )
            for sentence_idx, sentence in enumerate(sentences)
        ]
        thinking_start, thinking_end, match_count = _thinking_bounds(
            full_response,
            sentences,
            trace_idx=row_index,
        )
        total_text_matches += match_count
        total_sentences += len(sentences)
        question_ids.append(question_id)

        events: list[EventSpec] = []
        for sentence_idx, (text, start, end, label) in enumerate(parsed_sentences):
            key = f"{question_id}|{trace_idx}|{sentence_idx}"
            if key in all_label_by_key:
                raise TraceAlignmentError(f"duplicate locked event key {key!r}")
            all_label_keys.append(key)
            all_label_by_key[key] = label
            if key not in official_key_set:
                continue
            matches = (
                full_response[thinking_start + start : thinking_start + end]
                == text
            )
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
                    label_text_matches=matches,
                )
            )
        trace_specs.append(
            TraceSpec(
                question_id=question_id,
                trace_idx=trace_idx,
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
    return trace_specs, {
        "trace_count": len(trace_specs),
        "sentence_count": total_sentences,
        "official_rows": len(official_keys),
        "label_text_matches": total_text_matches,
        "label_text_mismatches": total_sentences - total_text_matches,
        "ordered_official_key_join": True,
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
    offsets = [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]]
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
        raise TraceAlignmentError(
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


def partition_trace_activations(
    hidden: np.ndarray,
    *,
    trace: TraceSpec,
    offsets: Sequence[tuple[int, int]],
    official_x_by_key: Mapping[str, np.ndarray],
    expected_width: int = EXPECTED_WIDTH,
) -> dict[str, np.ndarray]:
    """Slice one trace, proving the official tail before returning arrays."""

    hidden = np.asarray(hidden)
    if hidden.ndim != 2 or hidden.shape[1] != expected_width:
        raise TraceAlignmentError(
            f"captured hidden state must have shape (tokens, {expected_width}), "
            f"got {hidden.shape}"
        )
    if len(hidden) != len(offsets):
        raise TraceAlignmentError("captured hidden state and offsets disagree")

    wide_rows: list[np.ndarray] = []
    wide_keys: list[str] = []
    wide_labels: list[int] = []
    wide_boundaries: list[int] = []
    tail_rows: list[np.ndarray] = []
    tail_keys: list[str] = []
    tail_labels: list[int] = []
    tail_boundaries: list[int] = []

    for event in trace.events:
        boundary = token_containing_char(offsets, event.target_char)
        tail_positions = [boundary + value for value in OFFICIAL_OFFSETS]
        if min(tail_positions) < 0:
            raise TraceAlignmentError(
                f"official event {event.key!r} lacks offsets -13..-8"
            )
        extracted_tail = np.asarray(
            hidden[tail_positions],
            dtype=OUTPUT_DTYPE,
        )
        official_tail = np.asarray(
            official_x_by_key[event.key],
            dtype=OUTPUT_DTYPE,
        )
        if not np.array_equal(extracted_tail, official_tail):
            difference = np.abs(extracted_tail - official_tail)
            raise ExactTailError(
                f"event {event.key!r}: trailing six differ from official; "
                f"mismatched_values={int(np.count_nonzero(extracted_tail != official_tail))}, "
                f"max_abs={float(np.nanmax(difference))}"
            )
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

    return {
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
    }


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


def _validate_trace_shard(
    tensor_path: Path,
    sidecar_path: Path,
    *,
    trace: TraceSpec,
    request_sha256: str,
    expected_width: int = EXPECTED_WIDTH,
) -> dict[str, Any] | None:
    if not tensor_path.exists() and not sidecar_path.exists():
        return None
    if tensor_path.exists() != sidecar_path.exists():
        raise ShardValidationError(
            f"incomplete shard/sidecar pair for trace {trace.trace_idx}"
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SHARD_SCHEMA,
        "protocol_version": TEACHER_FORCE_PROTOCOL,
        "request_sha256": request_sha256,
        "question_id": trace.question_id,
        "trace_idx": trace.trace_idx,
        "full_response_sha256": trace.full_response_sha256,
        "trace_input_sha256": _trace_input_sha256(trace),
        "tail_exact_against_official": True,
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
        expected_keys = [event.key for event in trace.events]
        observed_keys = [*wide_keys.tolist(), *tail_keys.tolist()]
        if len(observed_keys) != len(set(observed_keys)):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: duplicate event key in shard"
            )
        if set(observed_keys) != set(expected_keys):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: shard event key set drifted"
            )
        if int(sidecar.get("wide_rows", -1)) != len(wide_keys):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: wide row count drifted"
            )
        if int(sidecar.get("tail_only_rows", -1)) != len(tail_keys):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: tail-only row count drifted"
            )
        if int(sidecar.get("official_rows_compared", -1)) != len(expected_keys):
            raise ShardValidationError(
                f"trace {trace.trace_idx}: compared row count drifted"
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
        offsets=ARTIFACT_OFFSETS,
        trailing_offsets=OFFICIAL_OFFSETS,
        trace_count=trace_count,
        sentence_count=sentence_count,
        official_rows=official_rows,
    )


def _trace_paths(output_dir: Path, trace_idx: int) -> tuple[Path, Path]:
    tensor = output_dir / "shards" / f"trace_{trace_idx:05d}.npz"
    return tensor, tensor.with_suffix(".manifest.json")


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
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    request_sha = _write_or_validate_request(
        output_dir / "request.json",
        request,
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
    pending: list[tuple[TraceSpec, Path, Path]] = []
    for trace in selected:
        tensor_path, sidecar_path = _trace_paths(output_dir, trace.trace_idx)
        if (
            _validate_trace_shard(
                tensor_path,
                sidecar_path,
                trace=trace,
                request_sha256=request_sha,
            )
            is None
        ):
            pending.append((trace, tensor_path, sidecar_path))
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
    for trace, tensor_path, sidecar_path in pending:
        tokenized = tokenize_full_response(tokenizer, trace.full_response)
        if len(tokenized["input_ids"]) > runtime["max_position_embeddings"]:
            raise TraceAlignmentError(
                f"trace {trace.trace_idx}: {len(tokenized['input_ids'])} tokens "
                f"exceed context {runtime['max_position_embeddings']}"
            )
        hidden = _forward_trace(
            model,
            layer_module,
            input_ids=tokenized["input_ids"],
            device=device,
        )
        arrays = partition_trace_activations(
            hidden,
            trace=trace,
            offsets=tokenized["offsets"],
            official_x_by_key=official_x_by_key,
        )
        _atomic_npz(tensor_path, **arrays)
        sidecar = {
            "schema_version": SHARD_SCHEMA,
            "protocol_version": TEACHER_FORCE_PROTOCOL,
            "request_sha256": request_sha,
            "question_id": trace.question_id,
            "trace_idx": trace.trace_idx,
            "full_response_sha256": trace.full_response_sha256,
            "trace_input_sha256": _trace_input_sha256(trace),
            "thinking_char_start": trace.thinking_char_start,
            "thinking_char_end": trace.thinking_char_end,
            "label_sentence_count": trace.label_sentence_count,
            "label_text_matches": trace.label_text_matches,
            "token_count": len(tokenized["input_ids"]),
            "wide_rows": len(arrays["wide_keys"]),
            "tail_only_rows": len(arrays["tail_only_keys"]),
            "official_rows_compared": len(trace.events),
            "tail_exact_against_official": True,
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
            trace=trace,
            request_sha256=request_sha,
        )
        completed += 1
        print(
            f"[ward-t16-extract] trace={trace.trace_idx} "
            f"wide={sidecar['wide_rows']} tail_only={sidecar['tail_only_rows']} "
            f"complete={completed}/{len(pending)}",
            flush=True,
        )
        del hidden, arrays
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
    )
    request_sha = _write_or_validate_request(
        output_dir / "request.json",
        request,
    )
    if artifact_path.exists() != manifest_path.exists():
        raise ShardValidationError(
            "final artifact and manifest must either both exist or both be absent"
        )
    if artifact_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("output", {}).get("sha256") != sha256_file(artifact_path):
            raise ShardValidationError("existing final artifact hash failed")
        if manifest.get("request_sha256") != request_sha:
            raise ShardValidationError("existing final manifest request drifted")
        return manifest

    sidecars: dict[int, dict[str, Any]] = {}
    wide_rows = 0
    for trace in traces:
        tensor_path, sidecar_path = _trace_paths(output_dir, trace.trace_idx)
        sidecar = _validate_trace_shard(
            tensor_path,
            sidecar_path,
            trace=trace,
            request_sha256=request_sha,
        )
        if sidecar is None:
            raise ShardValidationError(
                f"trace {trace.trace_idx}: extraction shard is missing"
            )
        sidecars[trace.trace_idx] = sidecar
        wide_rows += int(sidecar["wide_rows"])
    if wide_rows < 1:
        raise ShardValidationError("no official event supports all T16 offsets")

    temporary_x = artifact_path.with_name(
        f".{artifact_path.name}.assembling.npy"
    )
    temporary_x.parent.mkdir(parents=True, exist_ok=True)
    wide_x = np.lib.format.open_memmap(
        temporary_x,
        mode="w+",
        dtype=OUTPUT_DTYPE,
        shape=(wide_rows, len(ARTIFACT_OFFSETS), EXPECTED_WIDTH),
    )
    wide_labels = np.empty(wide_rows, dtype=np.uint8)
    wide_keys: list[str] = []
    official_cursor = 0
    wide_cursor = 0
    try:
        for trace in traces:
            tensor_path, _ = _trace_paths(output_dir, trace.trace_idx)
            with np.load(tensor_path, allow_pickle=False) as payload:
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
                for event in trace.events:
                    if official_cursor >= len(official_keys):
                        raise ShardValidationError(
                            "extracted more official events than expected"
                        )
                    if event.key != official_keys[official_cursor]:
                        raise ShardValidationError(
                            f"official key order drift at row {official_cursor}: "
                            f"{event.key!r} != {official_keys[official_cursor]!r}"
                        )
                    if event.label != int(official_labels[official_cursor]):
                        raise ShardValidationError(
                            f"official label drift for {event.key!r}"
                        )
                    if event.key in wide_index:
                        row = np.asarray(
                            payload["wide_X"][wide_index[event.key]],
                            dtype=OUTPUT_DTYPE,
                        )
                        tail = row[-len(OFFICIAL_OFFSETS) :]
                        wide_x[wide_cursor] = row
                        wide_labels[wide_cursor] = event.label
                        wide_keys.append(event.key)
                        wide_cursor += 1
                    elif event.key in tail_index:
                        tail = np.asarray(
                            payload["tail_only_X"][tail_index[event.key]],
                            dtype=OUTPUT_DTYPE,
                        )
                    else:
                        raise ShardValidationError(
                            f"trace {trace.trace_idx}: missing event {event.key!r}"
                        )
                    official_tail = np.asarray(
                        official_x[official_cursor],
                        dtype=OUTPUT_DTYPE,
                    )
                    if not np.array_equal(tail, official_tail):
                        raise ExactTailError(
                            f"assembly tail mismatch at official row "
                            f"{official_cursor}, key {event.key!r}"
                        )
                    official_cursor += 1
        if official_cursor != len(official_keys):
            raise ShardValidationError(
                f"assembled {official_cursor} official rows, "
                f"expected {len(official_keys)}"
            )
        if wide_cursor != wide_rows:
            raise ShardValidationError(
                f"assembled {wide_cursor} wide rows, expected {wide_rows}"
            )
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
        del wide_x
        if temporary_x.exists():
            temporary_x.unlink()

    key_array = np.asarray(wide_keys)
    common_hash = cohort_sha256(key_array, wide_labels)
    artifact_sha = sha256_file(artifact_path)
    manifest = {
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
            "rows": wide_rows,
            "sha256": common_hash,
            "exact_key_order": True,
            "selection": "official events with all offsets -23..-8",
        },
        "exact_key_order": True,
        "trailing_six": {
            "comparison": "exact_keyed_join",
            "keyed": True,
            "offsets": list(OFFICIAL_OFFSETS),
            "matched_keys": wide_rows,
            "official_rows_compared": len(official_keys),
            "exact_equal": True,
            "max_abs": 0.0,
            "mismatched_values": 0,
        },
        "validation": {
            **input_validation,
            "official_rows_compared": len(official_keys),
            "wide_rows": wide_rows,
            "wide_rows_dropped_for_missing_early_offsets": (
                len(official_keys) - wide_rows
            ),
            "complete_trace_shards": len(sidecars),
            "key_label_order_match": True,
            "trailing_six_exact_equal": True,
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
    )
    return {
        "status": "preflight-complete",
        "request": _request_payload(request),
        "request_sha256": _canonical_sha256(_request_payload(request)),
        "validation": validation,
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
