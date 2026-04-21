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


def _du(folder: Path) -> tuple[int, int]:
    files = [p for p in folder.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    return len(files), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["activations", "probe_cache"],
                    default=None, help="Upload only one subdir.")
    args = ap.parse_args()

    api = HfApi()

    n_act, s_act = _du(ACT_CACHE)
    n_pc, s_pc = _du(PROBE_CACHE)
    print(f"Activations: {n_act} files, {s_act / 1e9:.1f} GB")
    print(f"Probe cache: {n_pc} files, {s_pc / 1e9:.1f} GB")
    print(f"Total: {(s_act + s_pc) / 1e9:.1f} GB")

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

    if args.only in (None, "probe_cache"):
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

    print("\nDONE.")


if __name__ == "__main__":
    main()
