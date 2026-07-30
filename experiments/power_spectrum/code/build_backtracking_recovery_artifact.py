"""Build a transparent T=16 recovery artifact from replay candidates.

This is a sensitivity artifact, not a replacement for Aniket's bit-exact
artifact.  It matches the published row count and class balance by retaining
the lowest replay-RMSE candidates within each class, preserves source order,
and replaces the trailing six offsets with the official activations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import zipfile

import numpy as np


SCHEMA = "power-spectrum.backtracking-recovery-artifact.v1"
OFFSETS = tuple(range(-23, -7))
EXPECTED_REFERENCE_COHORT_SHA256 = (
    "f397f4caf6212825bd98b1b82be932ae634f01a716fd7e3642fd3d7640b27c0b"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cohort_sha256(keys: list[str], labels: list[int]) -> str:
    if len(keys) != len(labels):
        raise ValueError("keys and labels must have the same length")
    digest = hashlib.sha256()
    for key, label in zip(keys, labels):
        encoded = str(key).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(int(label).to_bytes(1, "little", signed=False))
    return digest.hexdigest()


def stratified_lowest_score(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    target_rows: int,
    target_positive_rows: int,
) -> np.ndarray:
    """Return an order-preserving mask with exact class targets."""

    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be same-length vectors")
    target_negative_rows = target_rows - target_positive_rows
    targets = {0: target_negative_rows, 1: target_positive_rows}
    selected = np.zeros(len(labels), dtype=bool)
    for label, count in targets.items():
        candidates = np.flatnonzero(labels == label)
        if count < 0 or count > len(candidates):
            raise ValueError(
                f"cannot select {count} rows for label {label} from "
                f"{len(candidates)} candidates"
            )
        order = np.argsort(scores[candidates], kind="stable")
        selected[candidates[order[:count]]] = True
    if int(selected.sum()) != target_rows:
        raise AssertionError("stratified selection returned the wrong size")
    return selected


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(values, [0.0, 0.5, 0.9, 0.99, 1.0])
    return {
        name: float(value)
        for name, value in zip(
            ("min", "median", "p90", "p99", "max"),
            points,
        )
    }


def _stored_npz_memmap(path: Path, name: str) -> np.memmap:
    """Memory-map an array saved by ``np.savez`` without loading the archive.

    NumPy ignores ``mmap_mode`` for ``.npz`` files and silently materializes
    the complete array.  Recovery artifacts are several GiB, so parse the
    stored ZIP member and NPY header to map its data bytes directly.
    """

    member = f"{name}.npy"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{member} is compressed and cannot be memory-mapped")
        with archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = (
                    np.lib.format.read_array_header_1_0(stream)
                )
            elif version == (2, 0):
                shape, fortran_order, dtype = (
                    np.lib.format.read_array_header_2_0(stream)
                )
            else:
                raise ValueError(f"unsupported NPY version {version} in {member}")
            npy_header_bytes = stream.tell()

    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
    if len(header) != 30:
        raise ValueError(f"truncated ZIP header for {member}")
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50:
        raise ValueError(f"invalid ZIP header for {member}")
    filename_bytes, extra_bytes = fields[-2:]
    data_offset = info.header_offset + 30 + filename_bytes + extra_bytes
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=data_offset + npy_header_bytes,
        shape=shape,
        order="F" if fortran_order else "C",
    )


def _tail_matches_official(
    recovered_x: np.ndarray,
    official_x: np.ndarray,
    official_rows: np.ndarray,
    *,
    block_rows: int = 256,
) -> bool:
    for start in range(0, len(official_rows), block_rows):
        stop = min(start + block_rows, len(official_rows))
        if not np.array_equal(
            recovered_x[start:stop, -6:],
            official_x[official_rows[start:stop]],
        ):
            return False
    return True


def _head_matches_candidates(
    recovered_x: np.ndarray,
    candidate_paths: list[Path],
    selected: np.ndarray,
) -> bool:
    """Verify that an existing artifact contains the selected replay prefix."""

    candidate_cursor = 0
    recovered_cursor = 0
    for path in candidate_paths:
        with np.load(path, allow_pickle=False) as payload:
            local_x = payload["candidate_X"]
            local_stop = candidate_cursor + len(local_x)
            local_selected = selected[candidate_cursor:local_stop]
            expected = local_x[local_selected, :-6]
            recovered_stop = recovered_cursor + len(expected)
            if not np.array_equal(
                recovered_x[recovered_cursor:recovered_stop, :-6],
                expected,
            ):
                return False
            candidate_cursor = local_stop
            recovered_cursor = recovered_stop
    return (
        candidate_cursor == len(selected)
        and recovered_cursor == int(selected.sum())
    )


def _candidate_metadata(
    candidate_paths: list[Path],
    *,
    official_index: dict[str, int],
    official_x: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    keys: list[str] = []
    labels: list[int] = []
    rmse: list[float] = []
    mean_abs: list[float] = []
    for path in candidate_paths:
        with np.load(path, allow_pickle=False) as payload:
            local_keys = payload["candidate_keys"].astype(str)
            local_labels = payload["candidate_is_bt"].astype(np.uint8)
            local_x = payload["candidate_X"]
            if local_x.shape[1:] != (len(OFFSETS), 4096):
                raise ValueError(f"invalid candidate shape {local_x.shape} at {path}")
            if len(local_keys):
                official_rows = np.asarray(
                    [official_index[key] for key in local_keys.tolist()]
                )
                difference = (
                    local_x[:, -6:].astype(np.float64)
                    - official_x[official_rows].astype(np.float64)
                )
                rmse.extend(
                    np.sqrt(np.square(difference).mean(axis=(1, 2))).tolist()
                )
                mean_abs.extend(
                    np.abs(difference).mean(axis=(1, 2)).tolist()
                )
            keys.extend(local_keys.tolist())
            labels.extend(local_labels.astype(int).tolist())
    if len(keys) != len(set(keys)):
        raise ValueError("candidate keys are not unique")
    return (
        keys,
        np.asarray(labels, dtype=np.uint8),
        np.asarray(rmse, dtype=np.float64),
        np.asarray(mean_abs, dtype=np.float64),
    )


def build(args: argparse.Namespace) -> dict:
    candidate_paths = sorted(args.candidate_dir.glob("trace_*.npz"))
    if not candidate_paths:
        raise ValueError(f"no candidate shards under {args.candidate_dir}")

    with np.load(args.official_artifact, allow_pickle=True) as official:
        official_keys = official["keys"].astype(str)
        official_labels = official["is_bt"].astype(np.uint8)
        official_x = official["X"]
        official_index = {
            key: index for index, key in enumerate(official_keys.tolist())
        }
        keys, labels, rmse, mean_abs = _candidate_metadata(
            candidate_paths,
            official_index=official_index,
            official_x=official_x,
        )
        if any(
            int(label) != int(official_labels[official_index[key]])
            for key, label in zip(keys, labels)
        ):
            raise ValueError("candidate labels disagree with official artifact")

        selected = stratified_lowest_score(
            labels,
            rmse,
            target_rows=args.target_rows,
            target_positive_rows=args.target_positive_rows,
        )
        selected_keys = [
            key for key, keep in zip(keys, selected) if bool(keep)
        ]
        selected_labels = labels[selected].astype(np.uint8)
        key_width = max(len(key) for key in selected_keys)
        selected_key_array = np.asarray(
            selected_keys,
            dtype=f"<U{key_width}",
        )

        args.output_artifact.parent.mkdir(parents=True, exist_ok=True)
        if not args.output_artifact.exists():
            temporary_x = args.output_artifact.with_suffix(".X.tmp.npy")
            temporary_npz = args.output_artifact.with_suffix(".npz.tmp")
            output_x = np.lib.format.open_memmap(
                temporary_x,
                mode="w+",
                dtype=np.float32,
                shape=(args.target_rows, len(OFFSETS), 4096),
            )
            cursor = 0
            global_row = 0
            for path in candidate_paths:
                with np.load(path, allow_pickle=False) as payload:
                    local_keys = payload["candidate_keys"].astype(str)
                    local_x = payload["candidate_X"]
                    for local_row, key in enumerate(local_keys.tolist()):
                        if selected[global_row]:
                            row = np.asarray(
                                local_x[local_row],
                                dtype=np.float32,
                            ).copy()
                            row[-6:] = official_x[official_index[key]]
                            output_x[cursor] = row
                            cursor += 1
                        global_row += 1
            if cursor != args.target_rows or global_row != len(keys):
                raise AssertionError("artifact assembly row accounting failed")
            output_x.flush()
            with temporary_npz.open("wb") as handle:
                np.savez(
                    handle,
                    X=output_x,
                    is_bt=selected_labels,
                    keys=selected_key_array,
                    offsets=np.asarray(OFFSETS, dtype=np.int32),
                )
            del output_x
            temporary_x.unlink()
            os.replace(temporary_npz, args.output_artifact)

        recovered_x = _stored_npz_memmap(args.output_artifact, "X")
        with np.load(args.output_artifact, allow_pickle=False) as recovered:
            recovered_keys = recovered["keys"].astype(str).tolist()
            if recovered_x.shape != (args.target_rows, len(OFFSETS), 4096):
                raise ValueError("recovered artifact shape changed on write")
            if (
                recovered_keys != selected_keys
                or not np.array_equal(recovered["is_bt"], selected_labels)
                or not np.array_equal(
                    recovered["offsets"],
                    np.asarray(OFFSETS, dtype=np.int32),
                )
            ):
                raise ValueError("recovered artifact cohort changed on write")
            recovered_official_rows = np.asarray(
                [official_index[key] for key in recovered_keys]
            )
            if not _tail_matches_official(
                recovered_x,
                official_x,
                recovered_official_rows,
            ):
                raise ValueError("recovered trailing six rows are not exact")
            if not _head_matches_candidates(
                recovered_x,
                candidate_paths,
                selected,
            ):
                raise ValueError(
                    "recovered replay prefix differs from selected candidates"
                )

    recovered_cohort = cohort_sha256(
        selected_keys,
        selected_labels.astype(int).tolist(),
    )
    candidate_cohort = cohort_sha256(keys, labels.astype(int).tolist())
    manifest = {
        "schema_version": SCHEMA,
        "status": "complete",
        "artifact": str(args.output_artifact),
        "artifact_sha256": sha256(args.output_artifact),
        "artifact_shape": [args.target_rows, len(OFFSETS), 4096],
        "artifact_dtype": "float32",
        "offsets": list(OFFSETS),
        "cohort": {
            "rows": args.target_rows,
            "positive_rows": int(selected_labels.sum()),
            "negative_rows": int(args.target_rows - selected_labels.sum()),
            "sha256": recovered_cohort,
            "expected_reference_sha256": (
                EXPECTED_REFERENCE_COHORT_SHA256
            ),
            "matches_reference": (
                recovered_cohort == EXPECTED_REFERENCE_COHORT_SHA256
            ),
        },
        "candidate_cohort": {
            "rows": len(keys),
            "positive_rows": int(labels.sum()),
            "negative_rows": int(len(labels) - labels.sum()),
            "sha256": candidate_cohort,
            "shards": len(candidate_paths),
        },
        "selection": {
            "method": (
                "lowest trailing-six replay RMSE within each label; "
                "stable ties; source order preserved"
            ),
            "target_rows": args.target_rows,
            "target_positive_rows": args.target_positive_rows,
            "selected_rmse_quantiles": _quantiles(rmse[selected]),
            "all_candidate_rmse_quantiles": _quantiles(rmse),
            "selected_mean_abs_quantiles": _quantiles(mean_abs[selected]),
            "all_candidate_mean_abs_quantiles": _quantiles(mean_abs),
        },
        "tail_replacement": {
            "offsets": list(OFFSETS[-6:]),
            "source_artifact": str(args.official_artifact),
            "source_artifact_sha256": sha256(args.official_artifact),
            "bit_exact_after_replacement": True,
        },
        "source_request": (
            json.loads(args.request.read_text())
            if args.request is not None
            else None
        ),
        "provenance_warning": (
            "Sensitivity artifact only: row count and class balance match "
            "Aniket's published evaluation, but the ordered cohort hash and "
            "the first ten replayed activation offsets are not bit-exact."
        ),
    }
    _atomic_json(manifest, args.output_manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--official-artifact", type=Path, required=True)
    parser.add_argument("--output-artifact", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--target-rows", type=int, default=20_335)
    parser.add_argument("--target-positive-rows", type=int, default=2_498)
    return parser


def main() -> None:
    manifest = build(_parser().parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
