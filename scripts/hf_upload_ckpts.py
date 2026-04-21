"""One-shot upload of Phase 5 + Phase 5.7 ckpts to HuggingFace Hub.

Uploads /workspace/temp_xc/experiments/phase5_downstream_utility/results/ckpts/*.pt
plus README.md to han1823123123/txcdr. Safe to re-run — upload_folder
skips unchanged files.

Usage:
    .venv/bin/python scripts/hf_upload_ckpts.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from huggingface_hub import HfApi


REPO = "han1823123123/txcdr"
REPO_TYPE = "model"

REPO_ROOT = Path("/workspace/temp_xc")
CKPT_DIR = REPO_ROOT / "experiments/phase5_downstream_utility/results/ckpts"
README_SRC = Path("/tmp/hf_readme.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be uploaded; do not upload.")
    args = ap.parse_args()

    ckpts = sorted(CKPT_DIR.glob("*.pt"))
    if not ckpts:
        print(f"FATAL: no ckpts found in {CKPT_DIR}")
        sys.exit(1)

    total_gb = sum(p.stat().st_size for p in ckpts) / 1e9
    print(f"Found {len(ckpts)} ckpts, {total_gb:.1f} GB total.")
    if args.dry_run:
        for p in ckpts:
            print(f"  {p.stat().st_size / 1e6:6.0f} MB  {p.name}")
        return

    api = HfApi()

    # 1. README intentionally NOT pushed from here — we keep a minimal
    # stub README on the HF repo (just a link back to github) per Han's
    # request that HF READMEs not duplicate research narrative.

    # 2. Upload ckpts folder. upload_folder handles the chunking + parallel
    # uploads and skips already-uploaded files by hash.
    print(f"Uploading {len(ckpts)} ckpts ({total_gb:.1f} GB) to {REPO}/ckpts/")
    r = api.upload_folder(
        folder_path=str(CKPT_DIR),
        path_in_repo="ckpts",
        repo_id=REPO, repo_type=REPO_TYPE,
        allow_patterns=["*.pt"],
        commit_message=(
            f"Phase 5 + 5.7 ckpts ({len(ckpts)} archs, {total_gb:.1f} GB)"
        ),
    )
    print(f"DONE — commit {r}")


if __name__ == "__main__":
    main()
