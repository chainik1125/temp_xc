"""Upload Phase 7 ckpts to HuggingFace Hub.

Targets `han1823123123/txcdr-base` (model repo, gemma-2-2b BASE
ckpts only). Hardcoded — there is no env-var override path.

Idempotent: re-runs skip already-uploaded files (HfApi compares
SHA-1 hashes). Safe to call after every training cell.

Subject-model verification: each ckpt's training-log JSON is
inspected for `meta["subject_model"] == "google/gemma-2-2b"` (or
equivalent fields). If the metadata is missing or mismatched the
upload is ABORTED with an error. This prevents accidentally pushing
IT-trained ckpts to the base repo.

Usage (from repo root):

    .venv/bin/python scripts/hf_upload_phase7_ckpts.py [--dry-run]

Local-aware: resolves repo root from `PHASE7_REPO` env var, falling
back to the script's grandparent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ─── Hardcoded targets — DO NOT CHANGE without updating phase7_unification docs.

REPO = "han1823123123/txcdr-base"
REPO_TYPE = "model"
PATH_IN_REPO = "ckpts"

EXPECTED_SUBJECT_MODEL = "google/gemma-2-2b"
EXPECTED_LAYER_ANCHOR = 12  # 0-indexed


def repo_root() -> Path:
    env = os.environ.get("PHASE7_REPO")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def verify_ckpt_metadata(ckpt_path: Path, log_path: Path) -> tuple[bool, str]:
    """Return (ok, message). Reads the sibling training_log JSON to
    check subject-model + anchor-layer fields. Aborts if mismatched.
    """
    if not log_path.exists():
        return False, f"missing training log {log_path.name}"
    try:
        log = json.loads(log_path.read_text())
    except Exception as e:
        return False, f"unreadable training log: {type(e).__name__}: {e}"
    sm = log.get("subject_model") or log.get("model") or log.get("hf_path")
    if sm != EXPECTED_SUBJECT_MODEL:
        return False, (
            f"subject_model mismatch: expected '{EXPECTED_SUBJECT_MODEL}', "
            f"got '{sm}'. This ckpt does not belong in {REPO}."
        )
    layer = log.get("layer") or log.get("anchor_layer")
    if layer is not None and int(layer) != EXPECTED_LAYER_ANCHOR:
        return False, (
            f"anchor_layer mismatch: expected {EXPECTED_LAYER_ANCHOR}, "
            f"got {layer}. This ckpt does not belong in {REPO}."
        )
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be uploaded; do not upload.")
    ap.add_argument("--skip-verify", action="store_true",
                    help="DANGEROUS: skip subject-model verification. "
                          "Only use if metadata is missing for a legitimate "
                          "reason (e.g., backfilling from an old run).")
    args = ap.parse_args()

    root = repo_root()
    ckpt_dir = root / "experiments/phase7_unification/results/ckpts"
    logs_dir = root / "experiments/phase7_unification/results/training_logs"
    if not ckpt_dir.exists():
        print(f"FATAL: {ckpt_dir} does not exist. Has Agent A trained anything yet?")
        sys.exit(1)

    ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        print(f"No ckpts found in {ckpt_dir}. Nothing to upload.")
        return

    # Verify each ckpt's metadata BEFORE uploading.
    if not args.skip_verify:
        bad = []
        for ckpt in ckpts:
            log = logs_dir / f"{ckpt.stem}.json"
            ok, msg = verify_ckpt_metadata(ckpt, log)
            if not ok:
                bad.append((ckpt.name, msg))
        if bad:
            print("FATAL: subject-model verification failed for the following:")
            for name, msg in bad:
                print(f"  {name}: {msg}")
            print("\nAborting. The Phase 7 base repo accepts ONLY ckpts whose")
            print(f"training logs declare subject_model = '{EXPECTED_SUBJECT_MODEL}'")
            print(f"and anchor_layer = {EXPECTED_LAYER_ANCHOR} (0-indexed).")
            print("\nIf metadata is missing for a legitimate reason, re-run with")
            print("--skip-verify (you are responsible for correctness).")
            sys.exit(2)

    total_gb = sum(p.stat().st_size for p in ckpts) / 1e9
    print(f"Found {len(ckpts)} ckpts, {total_gb:.1f} GB total in {ckpt_dir}")
    print(f"Target: {REPO} ({REPO_TYPE}) at path-in-repo `{PATH_IN_REPO}/`")

    if args.dry_run:
        for p in ckpts:
            print(f"  {p.stat().st_size / 1e6:6.0f} MB  {p.name}")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    print(f"\nUploading {len(ckpts)} ckpts ({total_gb:.1f} GB) "
          f"to {REPO}/{PATH_IN_REPO}/")
    r = api.upload_folder(
        folder_path=str(ckpt_dir),
        path_in_repo=PATH_IN_REPO,
        repo_id=REPO, repo_type=REPO_TYPE,
        allow_patterns=["*.pt"],
        commit_message=f"Phase 7 ckpts ({len(ckpts)} archs, {total_gb:.1f} GB)",
    )
    print(f"DONE — commit {r}")


if __name__ == "__main__":
    main()
