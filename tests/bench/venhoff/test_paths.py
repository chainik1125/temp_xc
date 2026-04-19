"""Resume / artifact-path invariants for the Venhoff pipeline.

These tests run without GPU and without live API calls. They verify:
  - `ArtifactPaths` produces stable, per-identity subtrees
  - `write_with_metadata` + `can_resume` round-trip correctly
  - Hash mismatch defeats resume
  - Missing sidecar defeats resume

If any of these break, the resume semantics stop being safe and the
`bash scripts/runpod_venhoff_launch.sh` would silently re-run stages
it should be skipping (or worse, skip stages it should be rerunning).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bench.venhoff.paths import (
    ArtifactPaths,
    RunIdentity,
    can_resume,
    write_with_metadata,
)


@pytest.fixture
def identity() -> RunIdentity:
    return RunIdentity(
        model="deepseek-r1-distill-llama-8b",
        dataset="mmlu-pro",
        dataset_split="test",
        n_traces=1000,
        layer=6,
        seed=42,
    )


@pytest.fixture
def paths(tmp_path: Path, identity: RunIdentity) -> ArtifactPaths:
    return ArtifactPaths(root=tmp_path, identity=identity)


def test_slug_encodes_identity(identity: RunIdentity) -> None:
    slug = identity.slug()
    assert "deepseek-r1-distill-llama-8b" in slug
    assert "n1000" in slug
    assert "L6" in slug
    assert "seed42" in slug


def test_run_dir_under_root(paths: ArtifactPaths, tmp_path: Path) -> None:
    assert paths.run_dir.parent == tmp_path
    assert paths.run_dir.name == paths.identity.slug()


def test_per_stage_paths_unique(paths: ArtifactPaths) -> None:
    p1_act = paths.activations_pkl("path1")
    p3_act = paths.activations_pkl("path3")
    assert p1_act != p3_act

    ck1 = paths.ckpt("sae", 15, "path1")
    ck2 = paths.ckpt("sae", 20, "path1")
    ck3 = paths.ckpt("tempxc", 15, "path3")
    assert len({ck1, ck2, ck3}) == 3


def test_write_with_metadata_creates_artifact_and_sidecar(paths: ArtifactPaths) -> None:
    paths.ensure_dirs()
    out = paths.traces_json
    config = {"stage": "generate_traces", "n_traces": 1000}

    write_with_metadata(out, json.dumps({"hello": "world"}), config)

    assert out.exists()
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    assert meta_path.exists()
    saved_meta = json.loads(meta_path.read_text())
    assert "config_hash" in saved_meta
    assert saved_meta["stage"] == "generate_traces"


def test_can_resume_hits_on_same_config(paths: ArtifactPaths) -> None:
    paths.ensure_dirs()
    out = paths.traces_json
    config = {"stage": "generate_traces", "n_traces": 1000, "seed": 42}
    write_with_metadata(out, "payload", config)

    assert can_resume(out, config) is True


def test_can_resume_misses_on_different_config(paths: ArtifactPaths) -> None:
    paths.ensure_dirs()
    out = paths.traces_json
    write_with_metadata(out, "payload", {"stage": "x", "n_traces": 1000})

    assert can_resume(out, {"stage": "x", "n_traces": 2000}) is False


def test_can_resume_misses_without_sidecar(paths: ArtifactPaths) -> None:
    paths.ensure_dirs()
    out = paths.traces_json
    out.write_text("naked payload with no sidecar")

    assert can_resume(out, {"stage": "x"}) is False


def test_can_resume_misses_without_artifact(paths: ArtifactPaths) -> None:
    paths.ensure_dirs()
    out = paths.traces_json
    # Write only the sidecar, no artifact.
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps({"config_hash": "abc"}))

    assert can_resume(out, {"stage": "x"}) is False
