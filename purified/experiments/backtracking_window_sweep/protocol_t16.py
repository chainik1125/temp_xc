"""Isolated protocol for the common-cohort C7 ``T<=16`` extension.

This module deliberately does not modify the frozen
``protocol.PROTOCOL_VERSION == 2026-07-23.2`` contract. Every T in this
extension is evaluated on the strict subset of Ward sentence events for which
all offsets ``-23..-8`` exist.

The primary artifact path is teacher-forced from pinned ``full_response``
traces. The unrelated Stage-B dictionary-training cache is accepted only by
the legacy explicit-coordinate-map builder and is not part of the
teacher-force provenance contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..backtracking_assets import CACHE_SHA256, CACHE_SHAPE
from .protocol import (
    EXPECTED_ARTIFACT_SHA256,
    EXPECTED_ARTIFACT_SHAPE,
    FULL_SEEDS,
    SweepProfile,
    sha256,
)


PROTOCOL_VERSION = "2026-07-26.t16.1"
ARTIFACT_OFFSETS = tuple(range(-23, -7))
FULL_WINDOWS = (1, 2, 4, 6, 8, 10, 12, 14, 16)
DEFAULT_ARTIFACT_NAME = "sentence_acts_L10_T16.npz"
DEFAULT_MANIFEST_NAME = "sentence_acts_L10_T16.manifest.json"
EXPECTED_WIDTH = 4_096
COORDINATE_MAP_PROTOCOL = "ward-c7-wide-artifact-audit.v1"
TEACHER_FORCE_PROTOCOL = "ward-c7-wide-teacher-force.v1"
TEACHER_FORCE_MANIFEST_SCHEMA = "ward-c7-wide-teacher-force-manifest.v4"
TEACHER_FORCE_TAIL_COMPARATOR = "float32-uint32-bit-exact.v1"
TEACHER_FORCE_SELECTION_POLICY = (
    "source-aligned-and-bit-exact-tail-and-t16.v1"
)
ACCEPTED_BUILDER_PROTOCOLS = (
    TEACHER_FORCE_PROTOCOL,
    COORDINATE_MAP_PROTOCOL,
)
EXPECTED_BUILDER_PROVENANCE = {
    "prompts": {
        "repo_id": "aniketdesh/ward-stage-b-cache",
        "revision": "4a7afcc5de12614d2c46872c21b068761e6bbe6a",
        "filename": "stageA_prompts.json",
        "sha256": "f718d76c1be63bddb83cfb7a9fe03ebde0bf5036a02defb1addc104f8829dd6a",
    },
    "labels": {
        "repo_id": "aniketdesh/ward-stage-b-cache",
        "revision": "4a7afcc5de12614d2c46872c21b068761e6bbe6a",
        "filename": "stageA_sentence_labels.json",
        "sha256": "329891b9a0858d1b2d58cc864624ebc6fc63d0a1e156efe4ca0fc8d020dff39c",
    },
    "token_ids": {
        "repo_id": "aniketdesh/ward-stage-b-cache",
        "revision": "4a7afcc5de12614d2c46872c21b068761e6bbe6a",
        "filename": "activations/token_ids.npy",
        "sha256": "e87fbff903c521ee810a91ee68f25cd707d0f3486940aa73cdae826a5c4f1d97",
    },
    "residual": {
        "repo_id": "aniketdesh/ward-stage-b-cache",
        "revision": "4a7afcc5de12614d2c46872c21b068761e6bbe6a",
        "filename": "activations/resid_L10.npy",
        "sha256": "bf36e55f6af3e7bd06d5568689d49f84c91c65191e9908d13749d61ff6087f5a",
    },
    "official": {
        "repo_id": "han1823123123/temp-bench-data",
        "revision": "6ef9b1debf863dedcef9555cad3a4903fb9e8c43",
        "filename": "c7_backtracking/stage_a/sentence_acts_L10.npz",
        "sha256": EXPECTED_ARTIFACT_SHA256,
    },
}


def profile(mode: str) -> SweepProfile:
    if mode == "smoke":
        return SweepProfile(
            mode=mode,
            windows=(1, 16),
            seeds=(42,),
            d_sae=128,
            k_pos=2,
            steps=2,
            batch_size=8,
            learning_rate=3e-4,
            warmup_steps=0,
            checkpoint_every=1,
            folds=2,
            s_grid=(4, 8),
            max_rows=800,
            bootstrap_repeats=50,
            amp=False,
        )
    if mode == "memory-smoke":
        return SweepProfile(
            mode=mode,
            windows=(16,),
            seeds=(42,),
            d_sae=32_768,
            k_pos=20,
            steps=1,
            batch_size=8,
            learning_rate=3e-4,
            warmup_steps=0,
            checkpoint_every=1,
            folds=2,
            s_grid=(8, 16, 32),
            max_rows=None,
            bootstrap_repeats=1,
            amp=True,
        )
    if mode == "full":
        return SweepProfile(
            mode=mode,
            windows=FULL_WINDOWS,
            seeds=FULL_SEEDS,
            d_sae=32_768,
            k_pos=20,
            steps=20_000,
            batch_size=1_024,
            learning_rate=3e-4,
            warmup_steps=1_000,
            checkpoint_every=1_000,
            folds=5,
            s_grid=(8, 16, 32),
            max_rows=None,
            bootstrap_repeats=2_000,
            amp=True,
        )
    raise ValueError(
        f"mode must be smoke, memory-smoke, or full, got {mode!r}"
    )


def validate_axes(windows: tuple[int, ...], seeds: tuple[int, ...]) -> None:
    if len(set(windows)) != len(windows):
        raise ValueError(f"duplicate windows: {windows}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")
    invalid = [window for window in windows if window not in FULL_WINDOWS]
    if invalid:
        raise ValueError(
            f"T16 protocol supports only {FULL_WINDOWS}; got {invalid}"
        )


def physical_offsets(window: int) -> tuple[int, ...]:
    if window not in FULL_WINDOWS:
        raise ValueError(f"window must be in {FULL_WINDOWS}, got {window}")
    return ARTIFACT_OFFSETS[-window:]


def window_queue(windows: tuple[int, ...]) -> tuple[int, ...]:
    """Run the two endpoints first, then the paper-faithful T=6 anchor."""

    priority = {1: 0, 16: 1, 6: 2}
    return tuple(
        sorted(windows, key=lambda window: (priority.get(window, 3), window))
    )


def cohort_sha256(keys: np.ndarray, labels: np.ndarray) -> str:
    """Hash the ordered common cohort without retaining any raw text."""

    if len(keys) != len(labels):
        raise ValueError("keys and labels must have the same length")
    digest = hashlib.sha256()
    for key, label in zip(keys, labels):
        encoded = str(key).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(int(label).to_bytes(1, "little", signed=False))
    return digest.hexdigest()


def _shape(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("shape")
    if not isinstance(value, (list, tuple)):
        return None
    return tuple(int(item) for item in value)


def _sha(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        candidate = value.get("sha256")
        return str(candidate) if candidate is not None else None
    return None


def _builder_provenance_ok(manifest: dict) -> bool:
    observed = manifest.get("provenance", {})
    for name, expected in EXPECTED_BUILDER_PROVENANCE.items():
        record = observed.get(name, {})
        if any(record.get(key) != value for key, value in expected.items()):
            return False
        if record.get("verified_sha256") != expected["sha256"]:
            return False
    return True


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256_string(value: object) -> bool:
    if not _nonempty_string(value) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _first_mapping(*values: object) -> dict:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def _identity(record: dict) -> tuple[object, object]:
    return (
        record.get("id", record.get("repo_id", record.get("model_id"))),
        record.get(
            "revision",
            record.get("commit", record.get("resolved_revision")),
        ),
    )


def _cohort_metadata_ok(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        rows = int(record["rows"])
        class_counts = record["class_counts"]
        category_counts = record["category_counts"]
        category_class_counts = record["category_class_counts"]
    except (KeyError, TypeError, ValueError):
        return False
    if (
        rows < 1
        or not _sha256_string(record.get("sha256"))
        or not _sha256_string(record.get("category_sha256"))
        or not isinstance(class_counts, dict)
        or not isinstance(category_counts, dict)
        or not isinstance(category_class_counts, dict)
        or set(class_counts) != {"0", "1"}
        or set(category_counts) != set(category_class_counts)
        or not category_counts
    ):
        return False
    try:
        class_values = {
            label: int(class_counts[label]) for label in ("0", "1")
        }
        category_values = {
            str(category): int(count)
            for category, count in category_counts.items()
        }
        cell_values = {
            str(category): {
                label: int(cells[label]) for label in ("0", "1")
            }
            for category, cells in category_class_counts.items()
            if isinstance(cells, dict) and set(cells) == {"0", "1"}
        }
    except (KeyError, TypeError, ValueError):
        return False
    if (
        set(cell_values) != set(category_values)
        or any(value < 0 for value in class_values.values())
        or any(value < 0 for value in category_values.values())
        or any(
            value < 0
            for cells in cell_values.values()
            for value in cells.values()
        )
    ):
        return False
    return (
        sum(class_values.values()) == rows
        and sum(category_values.values()) == rows
        and all(
            sum(cell_values[category].values())
            == category_values[category]
            for category in category_values
        )
        and all(
            sum(
                cell_values[category][label]
                for category in cell_values
            )
            == class_values[label]
            for label in ("0", "1")
        )
    )


def _coverage_metadata_ok(
    cohort: object,
    official: object,
) -> bool:
    if not (
        _cohort_metadata_ok(cohort)
        and _cohort_metadata_ok(official)
    ):
        return False
    assert isinstance(cohort, dict)
    assert isinstance(official, dict)
    if any(int(cohort["class_counts"][label]) < 1 for label in ("0", "1")):
        return False
    official_categories = set(official["category_counts"])
    if set(cohort["category_counts"]) != official_categories:
        return False
    for category in official_categories:
        for label in ("0", "1"):
            if (
                int(
                    official["category_class_counts"][category][label]
                )
                > 0
                and int(cohort["category_class_counts"][category][label])
                < 1
            ):
                return False
    return True


def _trace_accounting_ok(
    trace_summary: object,
    *,
    source_rows: int,
    exact_rows: int,
    wide_rows: int,
    expected_traces: int,
) -> bool:
    if (
        not isinstance(trace_summary, list)
        or len(trace_summary) != expected_traces
        or not all(isinstance(row, dict) for row in trace_summary)
    ):
        return False
    observed_source = 0
    observed_exact = 0
    observed_wide = 0
    for row in trace_summary:
        try:
            local_source = int(row["source_eligible_rows"])
            local_exact = int(row["exact_tail_rows"])
            local_wide = int(row["wide_rows"])
            reason_counts = row["exclusion_reason_counts"]
            source_class = row["source_class_counts"]
            exact_class = row["exact_tail_class_counts"]
            wide_class = row["wide_class_counts"]
        except (KeyError, TypeError, ValueError):
            return False
        try:
            local_ok = (
                min(local_source, local_exact, local_wide) >= 0
                and local_source >= local_exact >= local_wide
                and isinstance(reason_counts, dict)
                and sum(int(value) for value in reason_counts.values())
                == local_source - local_exact
                and sum(int(value) for value in source_class.values())
                == local_source
                and sum(int(value) for value in exact_class.values())
                == local_exact
                and sum(int(value) for value in wide_class.values())
                == local_wide
            )
        except (AttributeError, TypeError, ValueError):
            return False
        if not local_ok:
            return False
        observed_source += local_source
        observed_exact += local_exact
        observed_wide += local_wide
    return (
        observed_source == source_rows
        and observed_exact == exact_rows
        and observed_wide == wide_rows
    )


def _teacher_force_checks(
    manifest: dict,
    *,
    x_shape: tuple[int, ...],
    x_dtype: str,
    common_cohort_sha256: str,
    exact_key_order: bool,
) -> dict[str, bool]:
    """Validate the self-contained teacher-force extraction provenance."""

    provenance = manifest.get("provenance", {})
    extraction = manifest.get("extraction", {})
    source = _first_mapping(
        manifest.get("source"),
        provenance.get("source"),
        provenance.get("full_responses"),
    )
    model = _first_mapping(
        manifest.get("model"),
        extraction.get("model"),
        provenance.get("model"),
    )
    if not model:
        model = {
            "id": manifest.get("model_id"),
            "revision": manifest.get("model_revision"),
        }
    tokenizer = _first_mapping(
        manifest.get("tokenizer"),
        extraction.get("tokenizer"),
        provenance.get("tokenizer"),
    )
    if not tokenizer:
        tokenizer = {
            "id": manifest.get("tokenizer_id"),
            "revision": manifest.get("tokenizer_revision"),
        }
    activation = _first_mapping(
        manifest.get("activation"),
        extraction.get("activation"),
        extraction,
        manifest,
    )
    output = manifest.get("output", {})
    cohort = _first_mapping(
        manifest.get("common_cohort"),
        manifest.get("cohort"),
    )
    validation = manifest.get("validation", {})
    official_cohort = validation.get("official_cohort", {})
    source_cohort = validation.get("source_eligible_cohort", {})
    exact_cohort = validation.get("exact_tail_retained_cohort", {})
    wide_cohort = validation.get("wide_t16_cohort", {})
    coverage = validation.get("coverage", {})
    trailing = manifest.get("trailing_six", {})

    source_path = source.get(
        "path",
        source.get(
            "filename",
            source.get("file", source.get("relative_path")),
        ),
    )
    source_sha = source.get("sha256", source.get("file_sha256"))
    source_commit = source.get(
        "commit",
        source.get(
            "revision",
            source.get("git_commit", source.get("commit_sha")),
        ),
    )
    source_field = source.get(
        "field",
        source.get(
            "text_field",
            source.get("response_field", source.get("field_name")),
        ),
    )
    model_id, model_revision = _identity(model)
    tokenizer_id, tokenizer_revision = _identity(tokenizer)
    layer = activation.get("layer", activation.get("layer_index"))
    component = activation.get(
        "component",
        activation.get(
            "hook_semantics", activation.get("activation_component")
        ),
    )
    recorded_cohort = manifest.get(
        "common_cohort_sha256",
        cohort.get(
            "sha256",
            cohort.get("cohort_sha256"),
        ),
    )
    recorded_key_order = manifest.get(
        "exact_key_order",
        output.get("exact_key_order", cohort.get("exact_key_order")),
    )
    recorded_dtype = manifest.get("dtype", output.get("dtype"))
    try:
        official_rows = int(validation["official_rows"])
        source_excluded_rows = int(validation["source_excluded_rows"])
        source_rows = int(source_cohort["rows"])
        exact_rows = int(exact_cohort["rows"])
        wide_rows = int(wide_cohort["rows"])
        extraction_excluded_rows = int(
            validation["extraction_exclusion_rows"]
        )
        missing_early_rows = int(
            validation["wide_rows_dropped_for_missing_early_offsets"]
        )
        complete_trace_shards = int(
            validation["complete_trace_shards"]
        )
        trace_count = int(validation["trace_count"])
    except (KeyError, TypeError, ValueError):
        official_rows = source_excluded_rows = source_rows = -1
        exact_rows = wide_rows = extraction_excluded_rows = -1
        missing_early_rows = complete_trace_shards = trace_count = -1
    expected_coverage_names = {
        "source_eligible",
        "exact_tail_retained",
        "wide_t16",
    }
    coverage_records_ok = (
        isinstance(coverage, dict)
        and set(coverage) == expected_coverage_names
        and all(
            isinstance(record, dict)
            and record.get("cohort") == name
            and record.get("passed") is True
            and record.get("missing_classes") == []
            and record.get("missing_categories") == []
            and record.get("missing_category_class_cells") == []
            for name, record in coverage.items()
        )
    )
    cohort_accounting_ok = (
        official_rows >= 0
        and official_rows == source_rows + source_excluded_rows
        and source_rows == exact_rows + extraction_excluded_rows
        and exact_rows == wide_rows + missing_early_rows
        and wide_rows == x_shape[0]
        and int(validation.get("wide_rows", -1)) == wide_rows
    )
    common_metadata_ok = (
        _cohort_metadata_ok(cohort)
        and _cohort_metadata_ok(wide_cohort)
        and all(
            cohort.get(key) == wide_cohort.get(key)
            for key in (
                "rows",
                "sha256",
                "category_sha256",
                "class_counts",
                "category_counts",
                "category_class_counts",
            )
        )
        and cohort.get("selection") == TEACHER_FORCE_SELECTION_POLICY
        and cohort.get("exact_key_order") is True
    )
    structural_coverage_ok = all(
        _coverage_metadata_ok(candidate, official_cohort)
        for candidate in (source_cohort, exact_cohort, wide_cohort)
    )
    trace_accounting_ok = _trace_accounting_ok(
        validation.get("trace_summary"),
        source_rows=source_rows,
        exact_rows=exact_rows,
        wide_rows=wide_rows,
        expected_traces=trace_count,
    ) and complete_trace_shards == trace_count
    return {
        "teacher_manifest_schema_ok": (
            manifest.get("schema_version")
            == TEACHER_FORCE_MANIFEST_SCHEMA
        ),
        "teacher_source_pin_ok": (
            _nonempty_string(source_path)
            and _sha256_string(source_sha)
            and _nonempty_string(source_commit)
            and source.get("verified_sha256", source_sha) == source_sha
        ),
        "teacher_source_field_ok": source_field == "full_response",
        "teacher_model_pin_ok": (
            _nonempty_string(model_id)
            and _nonempty_string(model_revision)
        ),
        "teacher_tokenizer_pin_ok": (
            _nonempty_string(tokenizer_id)
            and _nonempty_string(tokenizer_revision)
        ),
        "teacher_activation_contract_ok": (
            int(layer) == 10
            if layer is not None
            else False
        )
        and component in {"resid_post", "residual_post"},
        "teacher_common_cohort_hash_ok": (
            recorded_cohort == common_cohort_sha256
        ),
        "teacher_key_order_ok": (
            recorded_key_order is True and exact_key_order
        ),
        "teacher_output_dtype_ok": (
            recorded_dtype == x_dtype == "float32"
        ),
        "teacher_output_rows_ok": (
            int(cohort.get("rows", x_shape[0])) == x_shape[0]
        ),
        "teacher_exact_subset_policy_ok": (
            trailing.get("comparator") == TEACHER_FORCE_TAIL_COMPARATOR
            and cohort.get("selection") == TEACHER_FORCE_SELECTION_POLICY
            and trailing.get("exact_equal") is True
            and int(trailing.get("mismatched_values", -1)) == 0
        ),
        "teacher_cohort_metadata_ok": (
            _cohort_metadata_ok(official_cohort)
            and _cohort_metadata_ok(source_cohort)
            and _cohort_metadata_ok(exact_cohort)
            and common_metadata_ok
        ),
        "teacher_cohort_accounting_ok": cohort_accounting_ok,
        "teacher_coverage_gates_ok": (
            coverage_records_ok and structural_coverage_ok
        ),
        "teacher_trace_accounting_ok": trace_accounting_ok,
        "teacher_exclusion_hashes_ok": all(
            _sha256_string(validation.get(key))
            for key in (
                "source_exclusions_sha256",
                "extraction_exclusions_sha256",
                "combined_exclusions_sha256",
            )
        ),
    }


def artifact_inventory(
    artifact: Path,
    manifest_path: Path,
    reference_artifact: Path,
    activation_cache: Path,
    *,
    strict_full: bool,
) -> dict:
    """Validate the wider artifact and its exact keyed relation to v2026-07-23.2."""

    result: dict = {
        "artifact": str(artifact),
        "artifact_manifest": str(manifest_path),
        "reference_artifact": str(reference_artifact),
        "activation_cache": str(activation_cache),
        "missing": [],
    }
    for path in (artifact, manifest_path, reference_artifact, activation_cache):
        if not path.exists():
            result["missing"].append(str(path))
    if result["missing"]:
        return result

    manifest = json.loads(manifest_path.read_text())
    with np.load(artifact, allow_pickle=True) as payload:
        artifact_keys = sorted(payload.files)
        x_array = payload["X"]
        x_shape = tuple(int(value) for value in x_array.shape)
        x_dtype = str(x_array.dtype)
        labels = payload["is_bt"].astype(np.uint8, copy=True)
        keys = payload["keys"].astype(str, copy=True)
        offsets = tuple(int(value) for value in payload["offsets"])
        del x_array
    with np.load(reference_artifact, allow_pickle=True) as payload:
        reference_shape = tuple(int(value) for value in payload["X"].shape)
        reference_labels = payload["is_bt"].astype(np.uint8, copy=True)
        reference_keys = payload["keys"].astype(str, copy=True)
    cache = np.load(activation_cache, mmap_mode="r")
    cache_shape = tuple(int(value) for value in cache.shape)
    cache_dtype = str(cache.dtype)
    cache_digest = sha256(activation_cache)

    result.update(
        {
            "artifact_keys": artifact_keys,
            "artifact_x_shape": list(x_shape),
            "artifact_x_dtype": x_dtype,
            "artifact_offsets": list(offsets),
            "artifact_shape_ok": (
                len(x_shape) == 3
                and x_shape[0] > 0
                and x_shape[1:] == (16, EXPECTED_WIDTH)
                and len(labels) == x_shape[0]
                and len(keys) == x_shape[0]
            ),
            "artifact_offsets_ok": offsets == ARTIFACT_OFFSETS,
            "artifact_keys_unique": len(np.unique(keys)) == len(keys),
            "reference_shape": list(reference_shape),
            "reference_shape_ok": (
                reference_shape == EXPECTED_ARTIFACT_SHAPE
                if strict_full
                else len(reference_shape) == 3
                and reference_shape[1:] == (6, EXPECTED_WIDTH)
            ),
            "reference_keys_unique": (
                len(np.unique(reference_keys)) == len(reference_keys)
            ),
            "activation_cache_shape": list(cache_shape),
            "activation_cache_dtype": cache_dtype,
            "activation_cache_sha256": cache_digest,
            "activation_cache_shape_ok": (
                cache_shape == CACHE_SHAPE
                if strict_full
                else cache.ndim == 3 and cache.shape[-1] == EXPECTED_WIDTH
            ),
            "activation_cache_dtype_ok": cache_dtype == "float16",
            "activation_cache_sha256_ok": cache_digest == CACHE_SHA256,
        }
    )

    reference_index = {
        key: index for index, key in enumerate(reference_keys.tolist())
    }
    positions = np.asarray(
        [reference_index.get(key, -1) for key in keys.tolist()],
        dtype=np.int64,
    )
    subset_ok = bool(np.all(positions >= 0))
    labels_match = bool(
        subset_ok and np.array_equal(labels, reference_labels[positions])
    )
    exact_key_order = bool(
        subset_ok and (len(positions) < 2 or np.all(np.diff(positions) > 0))
    )
    result.update(
        {
            "common_cohort_rows": int(len(keys)),
            "common_cohort_sha256": cohort_sha256(keys, labels),
            "official_key_subset_ok": subset_ok,
            "labels_match_official": labels_match,
            "exact_key_order": exact_key_order,
        }
    )

    validation = manifest.get("validation", {})
    trailing = manifest.get(
        "trailing_six",
        manifest.get(
            "trailing_six_keyed",
            validation.get(
                "trailing_six",
                {
                    "offsets": list(range(-13, -7)),
                    "exact_equal": validation.get(
                        "trailing_six_exact_equal"
                    ),
                    "max_abs": validation.get("trailing_six_max_abs"),
                    "mismatched_values": (
                        0
                        if validation.get("trailing_six_exact_equal") is True
                        else -1
                    ),
                },
            ),
        ),
    )
    manifest_output = manifest.get("output", {})
    manifest_official = manifest.get(
        "official_artifact",
        manifest.get("provenance", {}).get("official", {}),
    )
    manifest_offsets = manifest.get(
        "offsets", manifest_output.get("offsets")
    )
    builder_protocol = manifest.get("protocol_version")
    manifest_common_cohort_sha = manifest.get(
        "common_cohort_sha256",
        manifest.get("common_cohort", {}).get(
            "sha256",
            manifest.get("cohort", {}).get("sha256"),
        ),
    )
    trailing_rows = trailing.get(
        "matched_keys",
        trailing.get(
            "rows_compared",
            trailing.get("n_rows", x_shape[0]),
        ),
    )
    trailing_exact = trailing.get(
        "exact_equal", trailing.get("exact_keyed_equal")
    )
    trailing_mismatches = trailing.get(
        "mismatched_values",
        trailing.get(
            "mismatch_count",
            trailing.get("mismatched_elements", -1),
        ),
    )
    trailing_max_abs = trailing.get(
        "max_abs", trailing.get("max_abs_difference", float("inf"))
    )
    trailing_comparison = trailing.get(
        "comparison", trailing.get("mode")
    )
    trailing_is_keyed = (
        trailing.get("keyed") is True
        or trailing_comparison
        in {"exact_keyed", "keyed_exact", "exact_keyed_join"}
    )
    if builder_protocol == TEACHER_FORCE_PROTOCOL:
        validation_counts_ok = (
            int(trailing_rows) == x_shape[0]
            and int(
                validation.get(
                    "wide_rows",
                    manifest.get("common_cohort", {}).get(
                        "rows", x_shape[0]
                    ),
                )
            )
            == x_shape[0]
        )
    else:
        validation_counts_ok = (
            int(validation.get("official_rows_reconstructed", -1))
            == EXPECTED_ARTIFACT_SHAPE[0]
            and int(validation.get("wide_rows", -1)) == x_shape[0]
            and int(
                validation.get(
                    "wide_rows_dropped_for_missing_early_offsets", -1
                )
            )
            == EXPECTED_ARTIFACT_SHAPE[0] - x_shape[0]
        )
    result.update(
        {
            "manifest_protocol_version": builder_protocol,
            "manifest_status_ok": manifest.get("status") == "complete",
            "manifest_output_shape_ok": (
                _shape(manifest_output) == x_shape
                or _shape(manifest.get("output_shape")) == x_shape
                or _shape(manifest.get("shape")) == x_shape
            ),
            "manifest_offsets_ok": (
                tuple(int(value) for value in manifest_offsets)
                == ARTIFACT_OFFSETS
                if manifest_offsets is not None
                else False
            ),
            "manifest_exact_key_order_ok": (
                manifest.get(
                    "exact_key_order",
                    manifest_output.get(
                        "exact_key_order",
                        manifest.get("common_cohort", {}).get(
                            "exact_key_order",
                            exact_key_order,
                        ),
                    ),
                )
                is True
            ),
            "manifest_trailing_six_ok": (
                trailing_exact is True
                and int(trailing_mismatches) == 0
                and float(trailing_max_abs) == 0.0
                and tuple(int(value) for value in trailing.get("offsets", ()))
                == tuple(range(-13, -7))
                and (
                    trailing_is_keyed
                    if builder_protocol == TEACHER_FORCE_PROTOCOL
                    else True
                )
            ),
            "manifest_validation_counts_ok": validation_counts_ok,
            "manifest_common_cohort_sha256": manifest_common_cohort_sha,
        }
    )
    teacher_force_checks = (
        _teacher_force_checks(
            manifest,
            x_shape=x_shape,
            x_dtype=x_dtype,
            common_cohort_sha256=result["common_cohort_sha256"],
            exact_key_order=exact_key_order,
        )
        if builder_protocol == TEACHER_FORCE_PROTOCOL
        else {}
    )
    result.update(teacher_force_checks)
    artifact_digest = sha256(artifact)
    result.update(
        {
            "artifact_sha256": artifact_digest,
            "artifact_sha256_ok": (
                _sha(manifest_output)
                or _sha(manifest.get("output_sha256"))
            )
            == artifact_digest,
        }
    )
    if strict_full:
        reference_digest = sha256(reference_artifact)
        result.update(
            {
                "manifest_protocol_version_ok": (
                    builder_protocol in ACCEPTED_BUILDER_PROTOCOLS
                ),
                "manifest_source_provenance_ok": (
                    all(teacher_force_checks.values())
                    if builder_protocol == TEACHER_FORCE_PROTOCOL
                    else _builder_provenance_ok(manifest)
                ),
                "reference_artifact_sha256": reference_digest,
                "reference_artifact_sha256_ok": (
                    reference_digest == EXPECTED_ARTIFACT_SHA256
                    and (
                        _sha(manifest_official)
                        or _sha(manifest.get("official_artifact_sha256"))
                    )
                    == EXPECTED_ARTIFACT_SHA256
                ),
                "activation_cache_contract_ok": (
                    cache_shape == CACHE_SHAPE
                    and cache_dtype == "float16"
                    and cache_digest == CACHE_SHA256
                ),
            }
        )
    return result


def assert_inventory(inventory: dict, *, strict_full: bool) -> None:
    if inventory["missing"]:
        raise FileNotFoundError(
            "missing required T16 artifact(s): "
            + ", ".join(inventory["missing"])
        )
    checks = [
        "artifact_shape_ok",
        "artifact_offsets_ok",
        "artifact_keys_unique",
        "reference_shape_ok",
        "reference_keys_unique",
        "activation_cache_shape_ok",
        "official_key_subset_ok",
        "labels_match_official",
        "exact_key_order",
        "manifest_output_shape_ok",
        "manifest_offsets_ok",
        "manifest_exact_key_order_ok",
        "manifest_trailing_six_ok",
        "artifact_sha256_ok",
    ]
    if strict_full:
        checks.extend(
            (
                "artifact_sha256_ok",
                "manifest_status_ok",
                "manifest_protocol_version_ok",
                "manifest_source_provenance_ok",
                "manifest_validation_counts_ok",
                "reference_artifact_sha256_ok",
                "activation_cache_dtype_ok",
                "activation_cache_sha256_ok",
                "activation_cache_contract_ok",
            )
        )
    failures = {
        key: inventory.get(key) for key in checks if not inventory.get(key)
    }
    if failures:
        raise ValueError(f"T16 artifact provenance mismatch: {failures}")
