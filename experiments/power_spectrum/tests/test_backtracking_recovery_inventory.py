from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.power_spectrum.code import run_backtracking_fourier as runner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    official_labels: np.ndarray,
) -> tuple[Path, Path, Path]:
    keys = np.asarray(["question:0", "question:1"])
    labels = np.asarray([0, 1], dtype=np.uint8)
    official_x = np.arange(2 * 6 * 4096, dtype=np.float32).reshape(
        2,
        6,
        4096,
    )
    recovered_x = np.zeros((2, len(runner.ARTIFACT_OFFSETS), 4096), np.float32)
    recovered_x[:, -6:] = official_x

    artifact = tmp_path / "recovered.npz"
    reference = tmp_path / "official.npz"
    manifest_path = tmp_path / "recovered.manifest.json"
    np.savez(
        artifact,
        X=recovered_x,
        is_bt=labels,
        keys=keys,
        offsets=np.asarray(runner.ARTIFACT_OFFSETS, dtype=np.int32),
    )
    np.savez(
        reference,
        X=official_x,
        is_bt=official_labels,
        keys=keys,
    )

    artifact_digest = _sha256(artifact)
    reference_digest = _sha256(reference)
    cohort_digest = runner._cohort_sha256(keys, labels)
    manifest = {
        "schema_version": runner.RECOVERED_ARTIFACT_SCHEMA,
        "status": "complete",
        "artifact_sha256": artifact_digest,
        "cohort": {
            "rows": 2,
            "positive_rows": 1,
            "sha256": cohort_digest,
            "expected_reference_sha256": (
                runner.EXPECTED_REFERENCE_COHORT_SHA256
            ),
            "matches_reference": False,
        },
        "tail_replacement": {
            "source_artifact_sha256": reference_digest,
            "bit_exact_after_replacement": True,
        },
        "provenance_warning": "Fixture is not the reference cohort.",
    }
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(runner, "RECOVERED_ROWS", 2)
    monkeypatch.setattr(runner, "RECOVERED_POSITIVE_ROWS", 1)
    monkeypatch.setattr(
        runner,
        "EXPECTED_RECOVERED_ARTIFACT_SHA256",
        artifact_digest,
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_RECOVERED_MANIFEST_SHA256",
        _sha256(manifest_path),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_RECOVERED_COHORT_SHA256",
        cohort_digest,
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_OFFICIAL_ARTIFACT_SHA256",
        reference_digest,
    )
    monkeypatch.setattr(runner, "_reference_imports", lambda: {"sha256": _sha256})
    monkeypatch.setattr(
        runner,
        "activation_cache_inventory",
        lambda path: {"activation_cache": str(path)},
    )
    return artifact, manifest_path, reference


def test_recovered_inventory_verifies_pinned_artifact_and_official_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, manifest, reference = _fixture(
        tmp_path,
        monkeypatch,
        official_labels=np.asarray([0, 1], dtype=np.uint8),
    )

    inventory = runner.recovered_artifact_inventory(
        artifact,
        manifest,
        reference,
        tmp_path / "cache.npy",
    )

    assert inventory["artifact_sha256_pinned"]
    assert inventory["manifest_sha256_pinned"]
    assert inventory["cohort_sha256_pinned"]
    assert inventory["official_tail_bit_exact"]
    assert inventory["labels_match_official_artifact"]


def test_recovered_inventory_rejects_official_label_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, manifest, reference = _fixture(
        tmp_path,
        monkeypatch,
        official_labels=np.asarray([1, 1], dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="contract failed"):
        runner.recovered_artifact_inventory(
            artifact,
            manifest,
            reference,
            tmp_path / "cache.npy",
        )
