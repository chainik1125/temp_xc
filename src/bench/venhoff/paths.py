"""Artifact path registry for the Venhoff pipeline.

Every stage in the pipeline produces files under a single root. The
root encodes the experiment identity — model, dataset, n_traces, layer,
seed — so two runs with different configs never collide on disk. Within
a root, each stage has a deterministic filename.

Resume semantics: each stage checks `ArtifactPaths` for its expected
output before running. If the file exists AND its sidecar metadata
matches the current config, the stage is skipped. If the file exists
but metadata disagrees, the orchestrator raises unless `--force` or
`--force-stage {stage}` is set. This mirrors the SAEBench runpod
launcher's resume behavior (`scripts/runpod_saebench_launch.sh`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _short_hash(payload: dict[str, Any]) -> str:
    """Stable 12-char hash of a JSON-serializable config."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class RunIdentity:
    """Identifies a full pipeline run. Every artifact path is under this."""

    model: str          # e.g. "deepseek-r1-distill-llama-8b"
    dataset: str        # e.g. "mmlu-pro"
    dataset_split: str  # e.g. "test"
    n_traces: int       # e.g. 1000 (smoke) or 5000 (full)
    layer: int          # e.g. 6
    seed: int           # e.g. 42

    def slug(self) -> str:
        """Human-readable directory name."""
        return (
            f"{self.model}_{self.dataset}-{self.dataset_split}_"
            f"n{self.n_traces}_L{self.layer}_seed{self.seed}"
        )


@dataclass(frozen=True)
class ArtifactPaths:
    """Per-stage artifact paths under a fixed root + identity."""

    root: Path
    identity: RunIdentity

    @property
    def run_dir(self) -> Path:
        return self.root / self.identity.slug()

    # ── Stage 1: trace generation
    @property
    def traces_json(self) -> Path:
        return self.run_dir / "traces.json"

    # ── Stage 2: activation collection
    def activations_pkl(self, path: str) -> Path:
        """`path` is "path1" (per-sentence-mean) or "path3" (T-window)."""
        return self.run_dir / f"activations_{path}.pkl"

    def activation_mean_pkl(self, path: str) -> Path:
        return self.run_dir / f"activations_{path}_mean.pkl"

    def sentences_json(self, path: str) -> Path:
        """Sentence list aligned with activations — one row per act."""
        return self.run_dir / f"sentences_{path}.json"

    # ── Stage 3: small-k dictionary training
    def ckpt(self, arch: str, cluster_size: int, path: str) -> Path:
        return self.run_dir / "ckpts" / f"{arch}_k{cluster_size}_{path}.pt"

    # ── Stage 4: annotation (cluster assignment per sentence)
    def assignments_json(self, arch: str, cluster_size: int, path: str, aggregation: str) -> Path:
        return self.run_dir / "assignments" / f"{arch}_k{cluster_size}_{path}_{aggregation}.json"

    # ── Stage 5: cluster labeling (title + description)
    def labels_json(self, arch: str, cluster_size: int, path: str, aggregation: str) -> Path:
        return self.run_dir / "labels" / f"{arch}_k{cluster_size}_{path}_{aggregation}.json"

    # ── Stage 6: taxonomy scoring
    def scores_json(self, arch: str, cluster_size: int, path: str, aggregation: str) -> Path:
        return self.run_dir / "scores" / f"{arch}_k{cluster_size}_{path}_{aggregation}.json"

    # ── Stage 7: judge-bridge drift
    @property
    def bridge_json(self) -> Path:
        return self.run_dir / "bridge_drift.json"

    # ── Sidecar metadata for resume-correctness
    def metadata_for(self, artifact: Path) -> Path:
        return artifact.with_suffix(artifact.suffix + ".meta.json")

    def ensure_dirs(self) -> None:
        """Make all subdirectories eagerly."""
        for sub in ("ckpts", "assignments", "labels", "scores"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)


def write_with_metadata(artifact: Path, payload_bytes_or_text: bytes | str, meta: dict[str, Any]) -> None:
    """Atomic write of `artifact` + its sidecar metadata.

    Writes to `.tmp` then renames so partial writes never look valid
    to the resume check.
    """
    artifact.parent.mkdir(parents=True, exist_ok=True)
    tmp = artifact.with_suffix(artifact.suffix + ".tmp")
    if isinstance(payload_bytes_or_text, bytes):
        tmp.write_bytes(payload_bytes_or_text)
    else:
        tmp.write_text(payload_bytes_or_text)
    tmp.rename(artifact)

    meta_path = artifact.with_suffix(artifact.suffix + ".meta.json")
    meta_with_hash = {**meta, "config_hash": _short_hash(meta)}
    meta_tmp = meta_path.with_suffix(".tmp")
    meta_tmp.write_text(json.dumps(meta_with_hash, indent=2, sort_keys=True))
    meta_tmp.rename(meta_path)


def can_resume(artifact: Path, current_config: dict[str, Any]) -> bool:
    """Return True iff the artifact exists AND its sidecar hash matches.

    The current_config should be the same dict that will be passed to
    `write_with_metadata` when producing the artifact — this is the
    resume key.
    """
    if not artifact.exists():
        return False
    meta_path = artifact.with_suffix(artifact.suffix + ".meta.json")
    if not meta_path.exists():
        return False
    try:
        saved = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected_hash = _short_hash(current_config)
    return saved.get("config_hash") == expected_hash
