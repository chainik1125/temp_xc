#!/usr/bin/env python3
"""
push_to_hf.py — Upload trained model checkpoints to HuggingFace Hub.

Reads checkpoints from NLP/checkpoints/ and uploads them to a HF repo.
Does NOT run automatically — invoke manually when ready.

Usage:
    python push_to_hf.py --repo-id YOUR_HF_USERNAME/temporal-crosscoders-nlp
    python push_to_hf.py --repo-id antebe/temporal-crosscoders-nlp --only-best
    python push_to_hf.py --dry-run   # just list what would be uploaded
"""

import argparse
import glob
import json
import os
import sys

from config import CHECKPOINT_DIR, LOG_DIR


def find_checkpoints(checkpoint_dir: str) -> list[str]:
    """Find all .pt checkpoint files."""
    pattern = os.path.join(checkpoint_dir, "*.pt")
    return sorted(glob.glob(pattern))


def find_best_checkpoints(log_dir: str, checkpoint_dir: str) -> dict[str, str]:
    """Find the best checkpoint per architecture type (lowest loss)."""
    summary_path = os.path.join(log_dir, "sweep_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No sweep summary at {summary_path}")

    with open(summary_path) as f:
        summary = json.load(f)

    best: dict[str, dict] = {}
    for row in summary:
        model_type = row["model"]
        loss = row["final_loss"]
        ckpt_name = f"{model_type}__{row['layer']}__k{row['k']}__T{row['T']}.pt"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            continue

        if model_type not in best or loss < best[model_type]["loss"]:
            best[model_type] = {"path": ckpt_path, "loss": loss, "name": ckpt_name}

    return {k: v["path"] for k, v in best.items()}


def create_model_card(checkpoint_dir: str, log_dir: str) -> str:
    """Generate a README model card for the HF repo."""
    summary_path = os.path.join(log_dir, "sweep_summary.json")
    summary_text = ""
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        lines = ["| Model | Layer | k | T | Loss | FVU |", "|---|---|---|---|---|---|"]
        for row in sorted(summary, key=lambda r: r["final_loss"]):
            lines.append(
                f"| {row['model']} | {row['layer']} | {row['k']} | {row['T']} "
                f"| {row['final_loss']:.4f} | {row.get('final_fvu', 'N/A')} |"
            )
        summary_text = "\n".join(lines)

    return f"""---
tags:
  - sparse-autoencoder
  - temporal-crosscoder
  - mechanistic-interpretability
  - gemma-2-2b
license: mit
---

# Temporal Crosscoders — NLP (Gemma 2 2B)

Trained StackedSAE and TemporalCrosscoder dictionaries (expansion factor 8)
on Gemma 2 2B activations extracted from FineWeb.

## Architecture

- **Base model**: google/gemma-2-2b
- **Dictionary size**: 16384 (16k, ~7.1x expansion)
- **Sparsity**: TopK activation
- **Loss**: MSE reconstruction (no L1 penalty)

## Results

{summary_text}

## Usage

```python
import torch
from temporal_crosscoders.models import StackedSAE, TemporalCrosscoder

# Load a checkpoint
state = torch.load("stacked_sae__mid_res__k50__T10.pt", weights_only=True)
model = StackedSAE(d_in=2304, d_sae=16384, T=10, k=50)
model.load_state_dict(state)
```
"""


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints to HuggingFace Hub")
    parser.add_argument(
        "--repo-id", type=str, required=True,
        help="HuggingFace repo ID (e.g. antebe/temporal-crosscoders-nlp)",
    )
    parser.add_argument(
        "--only-best", action="store_true",
        help="Upload only the best checkpoint per architecture",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files to upload without uploading",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
    )
    parser.add_argument(
        "--log-dir", type=str, default=LOG_DIR,
    )
    args = parser.parse_args()

    # Gather files to upload
    if args.only_best:
        best = find_best_checkpoints(args.log_dir, args.checkpoint_dir)
        files_to_upload = list(best.values())
        print(f"Best checkpoints: {list(best.keys())}")
    else:
        files_to_upload = find_checkpoints(args.checkpoint_dir)

    if not files_to_upload:
        print("No checkpoint files found.")
        sys.exit(1)

    print(f"Files to upload ({len(files_to_upload)}):")
    total_size = 0
    for f in files_to_upload:
        size = os.path.getsize(f) / 1e9
        total_size += size
        print(f"  {os.path.basename(f)}  ({size:.2f} GB)")
    print(f"  Total: {total_size:.2f} GB")

    if args.dry_run:
        print("\n(dry run — nothing uploaded)")
        return

    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=True)

    # Upload model card
    model_card = create_model_card(args.checkpoint_dir, args.log_dir)
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print("  Uploaded README.md (model card)")

    # Upload sweep summary
    summary_path = os.path.join(args.log_dir, "sweep_summary.json")
    if os.path.exists(summary_path):
        api.upload_file(
            path_or_fileobj=summary_path,
            path_in_repo="sweep_summary.json",
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("  Uploaded sweep_summary.json")

    # Upload checkpoints
    for filepath in files_to_upload:
        filename = os.path.basename(filepath)
        print(f"  Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=f"checkpoints/{filename}",
            repo_id=args.repo_id,
            repo_type="model",
        )

    print(f"\nDone. Repo: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
