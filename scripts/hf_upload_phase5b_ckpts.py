"""Upload Phase 5B ckpts to HuggingFace Hub.

Uploads <repo_root>/experiments/phase5b_t_scaling_explore/results/ckpts/*.pt
to han1823123123/txcdr under path `phase5b_ckpts/`. Idempotent — re-runs
skip files that already match.

Local-aware: resolves repo root from the env var `PHASE5B_REPO`, falling
back to the script's grandparent (assumes scripts/ → repo root).

Usage:
    .venv/bin/python scripts/hf_upload_phase5b_ckpts.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("PHASE5B_REPO")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


REPO = "han1823123123/txcdr"
REPO_TYPE = "model"
PATH_IN_REPO = "phase5b_ckpts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be uploaded; do not upload.")
    args = ap.parse_args()

    root = repo_root()
    ckpt_dir = root / "experiments/phase5b_t_scaling_explore/results/ckpts"
    if not ckpt_dir.exists():
        print(f"FATAL: {ckpt_dir} does not exist.")
        sys.exit(1)

    ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        print(f"No ckpts found in {ckpt_dir}. Nothing to upload.")
        return

    total_gb = sum(p.stat().st_size for p in ckpts) / 1e9
    print(f"Found {len(ckpts)} ckpts, {total_gb:.1f} GB total in {ckpt_dir}")
    if args.dry_run:
        for p in ckpts:
            print(f"  {p.stat().st_size / 1e6:6.0f} MB  {p.name}")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Uploading {len(ckpts)} ckpts ({total_gb:.1f} GB) to "
            f"{REPO}/{PATH_IN_REPO}/")
    r = api.upload_folder(
        folder_path=str(ckpt_dir),
        path_in_repo=PATH_IN_REPO,
        repo_id=REPO, repo_type=REPO_TYPE,
        allow_patterns=["*.pt"],
        commit_message=f"Phase 5B ckpts ({len(ckpts)} archs, {total_gb:.1f} GB)",
    )
    print(f"DONE — commit {r}")


if __name__ == "__main__":
    main()
