"""Batch-upload every already-saved checkpoint under
``experiments/em_features/checkpoints/`` to HuggingFace Hub.

Intended to be run on whichever pod has checkpoints on disk that predate
the addition of the in-training auto-upload. Requires ``HF_UPLOAD_REPO``
and ``HF_TOKEN`` in the environment.

    uv run python -m experiments.em_features.hf_upload_existing \\
        --ckpt_dir /root/temp_xc/experiments/em_features/checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.em_features.hf_upload import upload_if_enabled


def infer_category(ckpt_path: Path) -> str:
    # Peek at the checkpoint's config dict to decide.
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[skip] {ckpt_path.name}: cannot load ({e})")
        return "unknown"
    ccfg = ckpt.get("config", {})
    if "T" in ccfg:
        return "txc"
    if "L" in ccfg and "layers" in ccfg:
        return "mlc_all" if ccfg["L"] > 5 else "mlc"
    if "hookpoint" in ccfg:
        return "sae_custom"
    if "d_sae" in ccfg:
        return "sae"
    return "unknown"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=Path, required=True)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ckpt_dir.is_dir():
        raise SystemExit(f"{args.ckpt_dir} is not a directory")

    pts = sorted(args.ckpt_dir.glob("*.pt"))
    if not pts:
        print(f"no checkpoints found under {args.ckpt_dir}")
        return

    for pt in pts:
        category = infer_category(pt)
        if category == "unknown":
            print(f"[skip] {pt.name}: category unknown")
            continue
        size_gb = pt.stat().st_size / 1e9
        print(f"[plan] {pt.name}  →  {category}/{pt.name}  ({size_gb:.2f} GB)")
        if not args.dry_run:
            upload_if_enabled(pt, category=category)


if __name__ == "__main__":
    main()
