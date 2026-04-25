"""Upload Phase 7 caches (activation cache + probe cache) to
HuggingFace Hub.

Targets `han1823123123/txcdr-base-data` (dataset repo, gemma-2-2b
BASE caches only). Hardcoded — there is no env-var override path.

Idempotent: re-runs skip already-uploaded files.

Two upload modes (mutually exclusive):

    --activation-cache  push data/cached_activations/gemma-2-2b/fineweb/*
    --probe-cache       push experiments/phase7_unification/results/probe_cache/*

Default (no flag) pushes BOTH.

Usage (from repo root):

    .venv/bin/python scripts/hf_upload_phase7_data.py [--dry-run]
                                                     [--activation-cache]
                                                     [--probe-cache]

Local-aware: resolves repo root from `PHASE7_REPO` env var, falling
back to the script's grandparent.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ─── Hardcoded targets — DO NOT CHANGE without updating phase7_unification docs.

REPO = "han1823123123/txcdr-base-data"
REPO_TYPE = "dataset"

ACTIVATION_CACHE_DIR_REL = "data/cached_activations/gemma-2-2b/fineweb"
PROBE_CACHE_DIR_REL = "experiments/phase7_unification/results/probe_cache"

ACTIVATION_PATH_IN_REPO = "activation_cache"
PROBE_PATH_IN_REPO = "probe_cache"


def repo_root() -> Path:
    env = os.environ.get("PHASE7_REPO")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def upload_dir(folder: Path, path_in_repo: str,
                allow_patterns: list[str], dry_run: bool):
    if not folder.exists():
        print(f"  skip: {folder} does not exist")
        return
    files = []
    for pat in allow_patterns:
        files.extend(folder.rglob(pat))
    if not files:
        print(f"  skip: no files matching {allow_patterns} in {folder}")
        return
    total_gb = sum(p.stat().st_size for p in files) / 1e9
    print(f"  found {len(files)} files, {total_gb:.1f} GB in {folder}")
    if dry_run:
        for p in sorted(files)[:20]:
            print(f"    {p.stat().st_size / 1e6:7.1f} MB  {p.relative_to(folder)}")
        if len(files) > 20:
            print(f"    ... and {len(files) - 20} more")
        return
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"  → uploading to {REPO}/{path_in_repo}/")
    r = api.upload_folder(
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        repo_id=REPO, repo_type=REPO_TYPE,
        allow_patterns=allow_patterns,
        commit_message=f"Phase 7 {path_in_repo} ({len(files)} files, {total_gb:.1f} GB)",
    )
    print(f"  done — commit {r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--activation-cache", action="store_true",
                    help="push only the activation cache")
    ap.add_argument("--probe-cache", action="store_true",
                    help="push only the probe cache")
    args = ap.parse_args()

    do_act = args.activation_cache or not args.probe_cache
    do_probe = args.probe_cache or not args.activation_cache

    root = repo_root()
    print(f"Repo root: {root}")
    print(f"Target: {REPO} ({REPO_TYPE})")

    if do_act:
        print(f"\n=== Activation cache (gemma-2-2b base, FineWeb) ===")
        upload_dir(
            folder=root / ACTIVATION_CACHE_DIR_REL,
            path_in_repo=ACTIVATION_PATH_IN_REPO,
            allow_patterns=["*.npy", "*.json"],
            dry_run=args.dry_run,
        )

    if do_probe:
        print(f"\n=== Probe cache ===")
        upload_dir(
            folder=root / PROBE_CACHE_DIR_REL,
            path_in_repo=PROBE_PATH_IN_REPO,
            allow_patterns=["*.npz", "*.json"],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
