"""Fail-closed provenance audit and wide C7 event-artifact builder.

The public Ward Stage-A label file and Stage-B dictionary-training cache do
not contain an explicit mapping from labelled traces to cache rows. In
particular, ``3300 == 300 * 11`` is not evidence that eleven adjacent cache
rows belong to each labelled trace. This module therefore has two modes:

``audit``
    Verify the small, pinned public inputs and explain why they are
    insufficient to reconstruct the official event-aligned artifact.

``build``
    Build offsets ``-23..-8`` only when an additional, provenance-locked
    event-coordinate map is supplied. Before writing an output, reconstruct
    every official ``-13..-8`` row from the source cache and require exact
    equality by event key. Missing or ambiguous coordinates fail closed.

No text, prompt, or sentence content is written to the audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


STAGE_B_REPO = "aniketdesh/ward-stage-b-cache"
STAGE_B_REVISION = "4a7afcc5de12614d2c46872c21b068761e6bbe6a"
OFFICIAL_REPO = "han1823123123/temp-bench-data"
OFFICIAL_REVISION = "6ef9b1debf863dedcef9555cad3a4903fb9e8c43"

WIDE_OFFSETS = tuple(range(-23, -7))
OFFICIAL_OFFSETS = tuple(range(-13, -7))
EVENT_MAP_SCHEMA = "ward-c7-event-coordinate-map.v1"
AUDIT_PROTOCOL = "ward-c7-wide-artifact-audit.v1"


@dataclass(frozen=True)
class LockedFile:
    repo_id: str
    revision: str
    filename: str
    size: int
    sha256: str


LOCKED_PROMPTS = LockedFile(
    repo_id=STAGE_B_REPO,
    revision=STAGE_B_REVISION,
    filename="stageA_prompts.json",
    size=115_246,
    sha256="f718d76c1be63bddb83cfb7a9fe03ebde0bf5036a02defb1addc104f8829dd6a",
)
LOCKED_LABELS = LockedFile(
    repo_id=STAGE_B_REPO,
    revision=STAGE_B_REVISION,
    filename="stageA_sentence_labels.json",
    size=4_904_727,
    sha256="329891b9a0858d1b2d58cc864624ebc6fc63d0a1e156efe4ca0fc8d020dff39c",
)
LOCKED_TOKEN_IDS = LockedFile(
    repo_id=STAGE_B_REPO,
    revision=STAGE_B_REVISION,
    filename="activations/token_ids.npy",
    size=6_758_528,
    sha256="e87fbff903c521ee810a91ee68f25cd707d0f3486940aa73cdae826a5c4f1d97",
)
LOCKED_RESIDUAL = LockedFile(
    repo_id=STAGE_B_REPO,
    revision=STAGE_B_REVISION,
    filename="activations/resid_L10.npy",
    size=6_920_601_728,
    sha256="bf36e55f6af3e7bd06d5568689d49f84c91c65191e9908d13749d61ff6087f5a",
)
LOCKED_OFFICIAL = LockedFile(
    repo_id=OFFICIAL_REPO,
    revision=OFFICIAL_REVISION,
    filename="c7_backtracking/stage_a/sentence_acts_L10.npz",
    size=1_137_333_114,
    sha256="1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810",
)

EXPECTED_PROMPTS = 300
EXPECTED_LABEL_RECORDS = 300
EXPECTED_SENTENCES = 25_528
EXPECTED_CACHE_SHAPE = (3_300, 256)
EXPECTED_RESIDUAL_SHAPE = (3_300, 256, 4_096)
EXPECTED_OFFICIAL_SHAPE = (25_204, 6, 4_096)


class ProvenanceError(RuntimeError):
    """An input is not the pinned public artifact."""


class AmbiguousTraceError(RuntimeError):
    """Sentence spans do not determine one exact trace string."""


class MappingError(RuntimeError):
    """An event-to-cache mapping is missing, incomplete, or ambiguous."""


class TailValidationError(RuntimeError):
    """The source cache does not exactly reproduce the official six offsets."""


@dataclass(frozen=True)
class SpanAudit:
    trace_idx: int
    question_id: str
    sentence_count: int
    covered_chars: int
    gap_chars: int
    conflicting_chars: int
    labeled_extent: int

    @property
    def exact_labeled_region(self) -> bool:
        return self.gap_chars == 0 and self.conflicting_chars == 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_file(path: Path, spec: LockedFile) -> dict[str, Any]:
    if not path.is_file():
        raise ProvenanceError(f"missing pinned input: {path}")
    size = path.stat().st_size
    if size != spec.size:
        raise ProvenanceError(
            f"size mismatch for {path}: observed {size}, expected {spec.size}"
        )
    digest = sha256_file(path)
    if digest != spec.sha256:
        raise ProvenanceError(
            f"SHA-256 mismatch for {path}: observed {digest}, expected {spec.sha256}"
        )
    return {
        **asdict(spec),
        "local_path": str(path.resolve()),
        "verified_size": size,
        "verified_sha256": digest,
    }


def _load_json_array(path: Path, *, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ProvenanceError(f"{name} must be a JSON array of objects")
    return payload


def audit_sentence_spans(record: Mapping[str, Any]) -> SpanAudit:
    trace_idx = int(record["trace_idx"])
    question_id = str(record["question_id"])
    sentences = record.get("sentences")
    if not isinstance(sentences, list):
        raise ProvenanceError(f"trace {trace_idx}: sentences must be a list")

    extent = max((int(sentence["char_end"]) for sentence in sentences), default=0)
    if extent < 0:
        raise ProvenanceError(f"trace {trace_idx}: negative labelled extent")
    chars: list[str | None] = [None] * extent
    conflicts: set[int] = set()

    for sentence_idx, sentence in enumerate(sentences):
        text = sentence.get("sentence")
        if not isinstance(text, str):
            raise ProvenanceError(
                f"trace {trace_idx} sentence {sentence_idx}: sentence must be text"
            )
        start = int(sentence["char_start"])
        end = int(sentence["char_end"])
        if start < 0 or end < start or end > extent:
            raise ProvenanceError(
                f"trace {trace_idx} sentence {sentence_idx}: invalid span [{start}, {end})"
            )
        if end - start != len(text):
            raise ProvenanceError(
                f"trace {trace_idx} sentence {sentence_idx}: span length "
                f"{end - start} != text length {len(text)}"
            )
        if not isinstance(sentence.get("is_backtracking"), bool):
            raise ProvenanceError(
                f"trace {trace_idx} sentence {sentence_idx}: label must be boolean"
            )
        for position, character in enumerate(text, start=start):
            existing = chars[position]
            if existing is None:
                chars[position] = character
            elif existing != character:
                conflicts.add(position)

    covered = sum(character is not None for character in chars)
    return SpanAudit(
        trace_idx=trace_idx,
        question_id=question_id,
        sentence_count=len(sentences),
        covered_chars=covered,
        gap_chars=extent - covered,
        conflicting_chars=len(conflicts),
        labeled_extent=extent,
    )


def reconstruct_labeled_region(record: Mapping[str, Any]) -> str:
    """Reconstruct the labelled character interval, or reject ambiguity.

    Even a successful return is only the labelled thinking-region interval;
    the labels do not contain the prompt, ``<think>`` wrapper, or proof that
    the final labelled character is the end of the full response.
    """

    audit = audit_sentence_spans(record)
    if not audit.exact_labeled_region:
        raise AmbiguousTraceError(
            f"trace {audit.trace_idx}: {audit.gap_chars} uncovered and "
            f"{audit.conflicting_chars} conflicting character positions"
        )
    chars: list[str | None] = [None] * audit.labeled_extent
    for sentence in record["sentences"]:
        start = int(sentence["char_start"])
        for position, character in enumerate(sentence["sentence"], start=start):
            chars[position] = character
    if any(character is None for character in chars):
        raise AssertionError("span audit accepted an uncovered character")
    return "".join(character for character in chars if character is not None)


def audit_public_inputs(
    prompts: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    token_ids: np.ndarray,
) -> dict[str, Any]:
    if len(prompts) != EXPECTED_PROMPTS:
        raise ProvenanceError(
            f"prompt count {len(prompts)} != pinned count {EXPECTED_PROMPTS}"
        )
    if len(labels) != EXPECTED_LABEL_RECORDS:
        raise ProvenanceError(
            f"label record count {len(labels)} != pinned count {EXPECTED_LABEL_RECORDS}"
        )
    if tuple(token_ids.shape) != EXPECTED_CACHE_SHAPE:
        raise ProvenanceError(
            f"token cache shape {tuple(token_ids.shape)} != {EXPECTED_CACHE_SHAPE}"
        )
    if token_ids.dtype.kind not in "iu":
        raise ProvenanceError(f"token IDs must be integers, got {token_ids.dtype}")

    prompt_ids: list[str] = []
    audits: list[SpanAudit] = []
    total_sentences = 0
    for index, (prompt, label_record) in enumerate(zip(prompts, labels, strict=True)):
        prompt_id = str(prompt.get("id"))
        question_id = str(label_record.get("question_id"))
        trace_idx = int(label_record.get("trace_idx", -1))
        if prompt_id != question_id:
            raise ProvenanceError(
                f"record {index}: prompt ID {prompt_id!r} != label ID {question_id!r}"
            )
        if trace_idx != index:
            raise ProvenanceError(
                f"record {index}: trace_idx {trace_idx} is not the locked row order"
            )
        prompt_ids.append(prompt_id)
        audit = audit_sentence_spans(label_record)
        audits.append(audit)
        total_sentences += audit.sentence_count
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ProvenanceError("prompt/question IDs must be unique")
    if total_sentences != EXPECTED_SENTENCES:
        raise ProvenanceError(
            f"sentence count {total_sentences} != pinned count {EXPECTED_SENTENCES}"
        )

    gap_only = sum(audit.gap_chars > 0 and audit.conflicting_chars == 0 for audit in audits)
    conflict_only = sum(
        audit.gap_chars == 0 and audit.conflicting_chars > 0 for audit in audits
    )
    gap_and_conflict = sum(
        audit.gap_chars > 0 and audit.conflicting_chars > 0 for audit in audits
    )
    exact_regions = sum(audit.exact_labeled_region for audit in audits)
    rows_per_trace = token_ids.shape[0] / len(labels)
    all_rows_share_first_token = bool(
        len(token_ids) > 0 and np.all(token_ids[:, 0] == token_ids[0, 0])
    )

    blockers = [
        {
            "code": "sentence_spans_do_not_determine_text",
            "detail": (
                f"{len(audits) - exact_regions}/{len(audits)} records contain "
                "uncovered or conflicting labelled character positions"
            ),
        },
        {
            "code": "full_responses_absent",
            "detail": (
                "sentence labels contain thinking-region spans, not the full_response "
                "strings used for token offset mapping"
            ),
        },
        {
            "code": "trace_cache_map_absent",
            "detail": (
                "the public cache has no trace ID, global token coordinate, or "
                "event-to-cache-row sidecar"
            ),
        },
        {
            "code": "rows_per_trace_arithmetic_is_not_mapping",
            "detail": (
                f"{token_ids.shape[0]} cache rows / {len(labels)} labels = "
                f"{rows_per_trace:g}, but the rows are independent dictionary-training "
                "sequences and this ratio supplies no alignment"
            ),
        },
    ]
    return {
        "protocol_version": AUDIT_PROTOCOL,
        "stage_b_repo": STAGE_B_REPO,
        "stage_b_revision": STAGE_B_REVISION,
        "prompt_count": len(prompts),
        "label_record_count": len(labels),
        "sentence_count": total_sentences,
        "span_audit": {
            "exact_labeled_regions": exact_regions,
            "gap_only_records": gap_only,
            "conflict_only_records": conflict_only,
            "gap_and_conflict_records": gap_and_conflict,
            "total_uncovered_characters": sum(audit.gap_chars for audit in audits),
            "total_conflicting_character_positions": sum(
                audit.conflicting_chars for audit in audits
            ),
        },
        "token_cache": {
            "shape": list(token_ids.shape),
            "dtype": str(token_ids.dtype),
            "rows_per_label_record_arithmetic": rows_per_trace,
            "all_rows_share_first_token": all_rows_share_first_token,
            "shared_first_token_id": (
                int(token_ids[0, 0]) if all_rows_share_first_token else None
            ),
        },
        "full_response_reconstruction_possible": False,
        "exact_official_six_reconstruction_possible": False,
        "wide_artifact_build_authorized": False,
        "blockers": blockers,
        "required_additional_artifact": {
            "schema_version": EVENT_MAP_SCHEMA,
            "description": (
                "one unique source-cache coordinate for every official key and offset "
                "-13..-8, plus coordinates for -23..-14 where available"
            ),
        },
    }


def _parse_location(raw: Any, *, key: str, offset: int) -> tuple[int, int] | None:
    if raw is None:
        return None
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or not all(isinstance(value, int) for value in raw)
    ):
        raise MappingError(
            f"event {key!r} offset {offset}: location must be [cache_row, token_position]"
        )
    return int(raw[0]), int(raw[1])


def parse_event_map(
    payload: Mapping[str, Any],
    *,
    official_sha256: str = LOCKED_OFFICIAL.sha256,
) -> dict[str, dict[int, tuple[int, int] | None]]:
    if payload.get("schema_version") != EVENT_MAP_SCHEMA:
        raise MappingError(
            f"event map schema must be {EVENT_MAP_SCHEMA!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    if payload.get("source_repo") != STAGE_B_REPO:
        raise MappingError("event map source_repo does not match the locked cache")
    if payload.get("source_revision") != STAGE_B_REVISION:
        raise MappingError("event map source_revision does not match the locked cache")
    if payload.get("official_sha256") != official_sha256:
        raise MappingError("event map official_sha256 does not match the locked artifact")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise MappingError("event map entries must be a list")

    parsed: dict[str, dict[int, tuple[int, int] | None]] = {}
    for entry_index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise MappingError(f"event map entry {entry_index} must be an object")
        key = entry.get("key")
        if not isinstance(key, str) or not key:
            raise MappingError(f"event map entry {entry_index} has no string key")
        if key in parsed:
            raise MappingError(f"duplicate event-map key {key!r}")
        locations = entry.get("locations")
        if not isinstance(locations, dict):
            raise MappingError(f"event {key!r}: locations must be an offset-keyed object")
        extra = set(locations) - {str(offset) for offset in WIDE_OFFSETS}
        if extra:
            raise MappingError(f"event {key!r}: unsupported offsets {sorted(extra)}")
        parsed[key] = {
            offset: _parse_location(locations.get(str(offset)), key=key, offset=offset)
            for offset in WIDE_OFFSETS
        }
        missing_tail = [
            offset for offset in OFFICIAL_OFFSETS if parsed[key][offset] is None
        ]
        if missing_tail:
            raise MappingError(
                f"event {key!r}: official-tail coordinates missing at {missing_tail}"
            )
    return parsed


def _to_float32(values: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        if values.dtype.itemsize != 2:
            raise
        words = np.asarray(values).view(np.uint16).astype(np.uint32)
        return (words << 16).view(np.float32)


def _require_contiguous_single_row(
    locations: Sequence[tuple[int, int]],
    *,
    key: str,
    offsets: Sequence[int],
) -> None:
    cache_rows = {location[0] for location in locations}
    positions = [location[1] for location in locations]
    expected_positions = list(range(positions[0], positions[0] + len(positions)))
    if len(cache_rows) != 1 or positions != expected_positions:
        raise MappingError(
            f"event {key!r} offsets {offsets[0]}..{offsets[-1]} do not map to "
            "one contiguous cache row"
        )


def reconstruct_from_event_map(
    source: np.ndarray,
    *,
    official_x: np.ndarray,
    official_y: np.ndarray,
    official_keys: Sequence[str],
    event_map: Mapping[str, Mapping[int, tuple[int, int] | None]],
    token_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct and validate a wide artifact from explicit coordinates."""

    if source.ndim != 3:
        raise ProvenanceError(f"source residual cache must be rank 3, got {source.shape}")
    if official_x.ndim != 3 or official_x.shape[1] != len(OFFICIAL_OFFSETS):
        raise ProvenanceError(
            f"official X must have six offsets, got shape {official_x.shape}"
        )
    if official_x.shape[0] != len(official_y) or len(official_y) != len(official_keys):
        raise ProvenanceError("official X, labels, and keys have inconsistent row counts")
    if official_x.shape[2] != source.shape[2]:
        raise ProvenanceError("source and official hidden widths differ")
    if token_ids is not None:
        if token_ids.dtype.kind not in "iu":
            raise ProvenanceError(f"token IDs must be integers, got {token_ids.dtype}")
        if tuple(token_ids.shape) != tuple(source.shape[:2]):
            raise ProvenanceError(
                f"token/source cache mismatch: {token_ids.shape} vs {source.shape[:2]}"
            )
    keys = [str(key) for key in official_keys]
    if len(set(keys)) != len(keys):
        raise ProvenanceError("official keys must be unique")
    if set(event_map) != set(keys):
        missing = sorted(set(keys) - set(event_map))
        extra = sorted(set(event_map) - set(keys))
        raise MappingError(
            f"event-map key set mismatch: {len(missing)} missing, {len(extra)} extra"
        )

    wide_rows: list[np.ndarray] = []
    wide_labels: list[Any] = []
    wide_keys: list[str] = []
    tail_mismatch_rows = 0

    for row_index, key in enumerate(keys):
        locations = event_map[key]
        tail_vectors = []
        for offset in OFFICIAL_OFFSETS:
            location = locations[offset]
            if location is None:
                raise MappingError(f"event {key!r}: tail offset {offset} is missing")
            cache_row, token_position = location
            if not (0 <= cache_row < source.shape[0]):
                raise MappingError(f"event {key!r}: cache row {cache_row} out of bounds")
            if not (0 <= token_position < source.shape[1]):
                raise MappingError(
                    f"event {key!r}: token position {token_position} out of bounds"
                )
            tail_vectors.append(_to_float32(source[cache_row, token_position]))
        _require_contiguous_single_row(
            [locations[offset] for offset in OFFICIAL_OFFSETS if locations[offset] is not None],
            key=key,
            offsets=OFFICIAL_OFFSETS,
        )
        reconstructed_tail = np.stack(tail_vectors)
        official_tail = _to_float32(official_x[row_index])
        if not np.array_equal(reconstructed_tail, official_tail):
            tail_mismatch_rows += 1
            difference = np.abs(reconstructed_tail - official_tail)
            raise TailValidationError(
                f"event {key!r}: trailing six do not exactly match; "
                f"max_abs={float(np.nanmax(difference))}"
            )

        wide_locations = [locations[offset] for offset in WIDE_OFFSETS]
        if any(location is None for location in wide_locations):
            continue
        complete_wide_locations = [
            location for location in wide_locations if location is not None
        ]
        _require_contiguous_single_row(
            complete_wide_locations,
            key=key,
            offsets=WIDE_OFFSETS,
        )
        vectors = []
        for offset, location in zip(WIDE_OFFSETS, wide_locations, strict=True):
            assert location is not None
            cache_row, token_position = location
            if not (0 <= cache_row < source.shape[0]):
                raise MappingError(
                    f"event {key!r} offset {offset}: cache row out of bounds"
                )
            if not (0 <= token_position < source.shape[1]):
                raise MappingError(
                    f"event {key!r} offset {offset}: token position out of bounds"
                )
            vectors.append(_to_float32(source[cache_row, token_position]))
        wide_rows.append(np.stack(vectors))
        wide_labels.append(official_y[row_index])
        wide_keys.append(key)

    if tail_mismatch_rows:
        raise AssertionError("tail mismatch should fail at the first row")
    if not wide_rows:
        raise MappingError("event map validates the six-offset artifact but maps no T16 rows")
    wide_x = np.stack(wide_rows)
    return (
        wide_x,
        np.asarray(wide_labels, dtype=official_y.dtype),
        np.asarray(wide_keys),
        {
            "official_rows_reconstructed": len(keys),
            "wide_rows": len(wide_rows),
            "wide_rows_dropped_for_missing_early_offsets": len(keys) - len(wide_rows),
            "trailing_six_exact_equal": True,
            "trailing_six_max_abs": 0.0,
        },
    )


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npz(
    path: Path,
    *,
    x: np.ndarray,
    labels: np.ndarray,
    keys: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            X=x,
            is_bt=labels,
            keys=keys,
            offsets=np.asarray(WIDE_OFFSETS, dtype=np.int32),
        )
    os.replace(temporary, path)


def run_audit(
    *,
    prompts_path: Path,
    labels_path: Path,
    token_ids_path: Path,
) -> dict[str, Any]:
    provenance = {
        "prompts": verify_locked_file(prompts_path, LOCKED_PROMPTS),
        "labels": verify_locked_file(labels_path, LOCKED_LABELS),
        "token_ids": verify_locked_file(token_ids_path, LOCKED_TOKEN_IDS),
    }
    prompts = _load_json_array(prompts_path, name="prompts")
    labels = _load_json_array(labels_path, name="labels")
    token_ids = np.load(token_ids_path, mmap_mode="r")
    report = audit_public_inputs(prompts, labels, token_ids)
    report["provenance"] = provenance
    return report


def run_build(
    *,
    prompts_path: Path,
    labels_path: Path,
    token_ids_path: Path,
    residual_path: Path,
    official_path: Path,
    event_map_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    audit = run_audit(
        prompts_path=prompts_path,
        labels_path=labels_path,
        token_ids_path=token_ids_path,
    )
    if not event_map_path.is_file():
        raise MappingError(
            "the public datasets contain no event-coordinate map; refusing to "
            "infer one from 3300 / 300 = 11"
        )
    provenance = {
        **audit["provenance"],
        "residual": verify_locked_file(residual_path, LOCKED_RESIDUAL),
        "official": verify_locked_file(official_path, LOCKED_OFFICIAL),
        "event_map": {
            "local_path": str(event_map_path.resolve()),
            "size": event_map_path.stat().st_size,
            "sha256": sha256_file(event_map_path),
        },
    }
    source = np.load(residual_path, mmap_mode="r")
    if tuple(source.shape) != EXPECTED_RESIDUAL_SHAPE:
        raise ProvenanceError(
            f"residual shape {tuple(source.shape)} != {EXPECTED_RESIDUAL_SHAPE}"
        )
    token_ids = np.load(token_ids_path, mmap_mode="r")
    with np.load(official_path, allow_pickle=True) as official:
        official_x = official["X"]
        official_y = official["is_bt"]
        official_keys = np.asarray(official["keys"]).astype(str)
        if tuple(official_x.shape) != EXPECTED_OFFICIAL_SHAPE:
            raise ProvenanceError(
                f"official shape {tuple(official_x.shape)} != {EXPECTED_OFFICIAL_SHAPE}"
            )
        event_payload = json.loads(event_map_path.read_text(encoding="utf-8"))
        if not isinstance(event_payload, dict):
            raise MappingError("event map must be a JSON object")
        event_map = parse_event_map(event_payload)
        wide_x, wide_y, wide_keys, validation = reconstruct_from_event_map(
            source,
            official_x=official_x,
            official_y=official_y,
            official_keys=official_keys,
            event_map=event_map,
            token_ids=token_ids,
        )

    _atomic_npz(output_path, x=wide_x, labels=wide_y, keys=wide_keys)
    manifest = {
        "protocol_version": AUDIT_PROTOCOL,
        "status": "complete",
        "offsets": list(WIDE_OFFSETS),
        "shape": list(wide_x.shape),
        "dtype": str(wide_x.dtype),
        "provenance": provenance,
        "audit": audit,
        "validation": validation,
        "output": {
            "path": str(output_path.resolve()),
            "size": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }
    _atomic_json(manifest, manifest_path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_small_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--prompts", type=Path, required=True)
        command.add_argument("--labels", type=Path, required=True)
        command.add_argument("--token-ids", type=Path, required=True)

    audit = subparsers.add_parser("audit", help="audit the pinned small public inputs")
    add_small_inputs(audit)
    audit.add_argument("--report", type=Path)

    build = subparsers.add_parser(
        "build",
        help="build only with an explicit event-coordinate map and exact tail validation",
    )
    add_small_inputs(build)
    build.add_argument("--residual", type=Path, required=True)
    build.add_argument("--official", type=Path, required=True)
    build.add_argument("--event-map", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "audit":
        report = run_audit(
            prompts_path=args.prompts,
            labels_path=args.labels,
            token_ids_path=args.token_ids,
        )
        if args.report is not None:
            _atomic_json(report, args.report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    manifest = run_build(
        prompts_path=args.prompts,
        labels_path=args.labels,
        token_ids_path=args.token_ids,
        residual_path=args.residual,
        official_path=args.official,
        event_map_path=args.event_map,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
