"""One-shot upload of Phase 5 reproducibility caches to the dataset repo.

Uploads:
  - /workspace/temp_xc/data/cached_activations/gemma-2-2b-it/fineweb/
    (5 layer files + token_ids.npy, ~17 GB)
  - /workspace/temp_xc/experiments/phase5_downstream_utility/results/probe_cache/
    (36 task dirs, ~66 GB)

Target: han1823123123/txcdr-data (dataset repo).

Mirror-paths: the folder layout in the HF repo matches the local path
under the github repo root, so a downloader using
`huggingface-cli download --local-dir .` lands every file at the
canonical location. Safe to re-run — upload_folder skips unchanged
files by hash.

Usage:
    .venv/bin/python scripts/hf_upload_data.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from huggingface_hub import HfApi


REPO = "han1823123123/txcdr-data"
REPO_TYPE = "dataset"

REPO_ROOT = Path("/workspace/temp_xc")
ACT_CACHE = REPO_ROOT / "data/cached_activations/gemma-2-2b-it/fineweb"
PROBE_CACHE = REPO_ROOT / "experiments/phase5_downstream_utility/results/probe_cache"
PHASE6_CORPORA = REPO_ROOT / "experiments/phase6_qualitative_latents/concat_corpora"
PHASE6_Z_CACHE = REPO_ROOT / "experiments/phase6_qualitative_latents/z_cache"


def _du(folder: Path) -> tuple[int, int]:
    files = [p for p in folder.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    return len(files), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--only",
        choices=["activations", "probe_cache", "phase6_corpora", "phase6_z"],
        default=None, help="Upload only one subdir.",
    )
    args = ap.parse_args()

    api = HfApi()

    n_act, s_act = _du(ACT_CACHE) if ACT_CACHE.exists() else (0, 0)
    n_pc, s_pc = _du(PROBE_CACHE) if PROBE_CACHE.exists() else (0, 0)
    n_p6c, s_p6c = _du(PHASE6_CORPORA) if PHASE6_CORPORA.exists() else (0, 0)
    n_p6z, s_p6z = _du(PHASE6_Z_CACHE) if PHASE6_Z_CACHE.exists() else (0, 0)
    print(f"Activations:       {n_act} files, {s_act / 1e9:.2f} GB")
    print(f"Probe cache:       {n_pc} files, {s_pc / 1e9:.2f} GB")
    print(f"Phase 6 corpora:   {n_p6c} files, {s_p6c / 1e9:.4f} GB")
    print(f"Phase 6 z_cache:   {n_p6z} files, {s_p6z / 1e9:.2f} GB")
    total = s_act + s_pc + s_p6c + s_p6z
    print(f"Total:             {total / 1e9:.2f} GB")

    if args.dry_run:
        return

    if args.only in (None, "activations"):
        print(f"\nUploading activations -> {REPO} / "
              f"data/cached_activations/gemma-2-2b-it/fineweb/")
        r = api.upload_folder(
            folder_path=str(ACT_CACHE),
            path_in_repo="data/cached_activations/gemma-2-2b-it/fineweb",
            repo_id=REPO, repo_type=REPO_TYPE,
            allow_patterns=["*.npy", "*.json", "token_ids*"],
            ignore_patterns=["*.progress.json"],
            commit_message=(
                f"Phase 5 activations: Gemma-2-2B-IT fineweb cache "
                f"({n_act} files, {s_act / 1e9:.1f} GB)"
            ),
        )
        print(f"  commit: {r}")

    if args.only in (None, "probe_cache") and PROBE_CACHE.exists():
        print(f"\nUploading probe_cache -> {REPO} / "
              f"experiments/phase5_downstream_utility/results/probe_cache/")
        r = api.upload_folder(
            folder_path=str(PROBE_CACHE),
            path_in_repo="experiments/phase5_downstream_utility/results/probe_cache",
            repo_id=REPO, repo_type=REPO_TYPE,
            allow_patterns=["*.npz", "*.json"],
            commit_message=(
                f"Phase 5 probe_cache: 36 tasks × (anchor, mlc, mlc_tail) "
                f"({n_pc} files, {s_pc / 1e9:.1f} GB)"
            ),
        )
        print(f"  commit: {r}")

    if args.only in (None, "phase6_corpora") and PHASE6_CORPORA.exists():
        print(f"\nUploading phase6 concat corpora -> {REPO} / "
              f"experiments/phase6_qualitative_latents/concat_corpora/")
        r = api.upload_folder(
            folder_path=str(PHASE6_CORPORA),
            path_in_repo="experiments/phase6_qualitative_latents/concat_corpora",
            repo_id=REPO, repo_type=REPO_TYPE,
            allow_patterns=["*.json"],
            ignore_patterns=["sources/*"],  # raw fetches are re-downloadable
            commit_message=(
                f"Phase 6 concat corpora: A, B, C token IDs + provenance "
                f"({n_p6c} files)"
            ),
        )
        print(f"  commit: {r}")

    if args.only in (None, "phase6_z") and PHASE6_Z_CACHE.exists():
        print(f"\nUploading phase6 z_cache -> {REPO} / "
              f"experiments/phase6_qualitative_latents/z_cache/")
        r = api.upload_folder(
            folder_path=str(PHASE6_Z_CACHE),
            path_in_repo="experiments/phase6_qualitative_latents/z_cache",
            repo_id=REPO, repo_type=REPO_TYPE,
            allow_patterns=["*.npy", "*.json"],
            commit_message=(
                f"Phase 6 z_cache: per-arch encoded latents on concat A/B/C "
                f"({n_p6z} files, {s_p6z / 1e9:.2f} GB)"
            ),
        )
        print(f"  commit: {r}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
