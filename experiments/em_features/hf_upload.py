"""HuggingFace Hub upload helper for em_features checkpoints.

Opt-in: reads ``HF_UPLOAD_REPO`` (repo id like ``chainik1125/temp-xc-em-features``)
and ``HF_TOKEN`` from environment. If either is missing, ``upload_if_enabled``
is a no-op — training scripts can unconditionally call it after saving a
checkpoint and it'll only actually upload when the env is configured.

    upload_if_enabled(Path("/root/.../qwen_l15_txc_small_step300000.pt"),
                      category="txc")
      → uploads to $HF_UPLOAD_REPO/txc/qwen_l15_txc_small_step300000.pt

    python -m experiments.em_features.hf_upload \\
        --ckpt /root/.../qwen_l15_txc_small_step300000.pt --category txc

"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def upload_if_enabled(ckpt_path: Path, *, category: str, repo_id: str | None = None,
                     token: str | None = None, path_in_repo: str | None = None) -> None:
    """Upload ``ckpt_path`` to HF Hub under ``{category}/{ckpt_name}`` unless
    env is not configured, in which case this is a no-op.

    `category` should be one of ``txc``, ``mlc``, ``mlc_all``, ``sae`` — used
    as a top-level folder inside the repo.
    """
    repo_id = repo_id or os.environ.get("HF_UPLOAD_REPO")
    token = token or os.environ.get("HF_TOKEN")
    if not repo_id:
        print(f"[hf_upload] HF_UPLOAD_REPO not set — skipping upload of {ckpt_path.name}")
        return
    if not token:
        print(f"[hf_upload] HF_TOKEN not set — skipping upload of {ckpt_path.name}")
        return
    if not ckpt_path.exists():
        print(f"[hf_upload] WARN {ckpt_path} does not exist")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[hf_upload] huggingface_hub not installed — skipping")
        return

    api = HfApi(token=token)
    target = path_in_repo or f"{category}/{ckpt_path.name}"

    t0 = time.time()
    # Ensure repo exists (idempotent).
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    except Exception as e:
        print(f"[hf_upload] create_repo failed (continuing): {e}")

    try:
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo=target,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"upload {target}",
        )
        sz = ckpt_path.stat().st_size / 1e9
        dt = time.time() - t0
        print(f"[hf_upload] pushed {ckpt_path.name} → {repo_id}:{target} "
              f"({sz:.2f} GB in {dt:.1f}s)")
    except Exception as e:
        # Never let a failed upload kill training; just warn.
        print(f"[hf_upload] upload failed for {ckpt_path.name}: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--category", required=True,
                   choices=["txc", "mlc", "mlc_all", "sae", "sae_custom"])
    p.add_argument("--repo_id", default=None)
    p.add_argument("--path_in_repo", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    upload_if_enabled(args.ckpt, category=args.category,
                      repo_id=args.repo_id, path_in_repo=args.path_in_repo)


if __name__ == "__main__":
    main()
