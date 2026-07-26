"""Download the public activation cache used by the backtracking dictionaries."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download


REPO_ID = "han1823123123/temp-bench-data"
REPO_REVISION = "6ef9b1debf863dedcef9555cad3a4903fb9e8c43"
FILES = (
    "act_cache/fb2a74be884e512a/resid_post_L10.npy",
    "act_cache/fb2a74be884e512a/token_ids.npy",
    "act_cache/fb2a74be884e512a/corpus.json",
    "act_cache/fb2a74be884e512a/layer_specs.json",
)
CACHE_FILE = FILES[0]
CACHE_SHA256 = (
    "dc34dfb117f77abddef4b4396d0d00afc707c39876d0ee36015de1e7b8406914"
)
CACHE_SHAPE = (4_044, 128, 4_096)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cache(path: Path) -> None:
    digest = _sha256(path)
    cache = np.load(path, mmap_mode="r")
    checks = {
        "sha256": digest == CACHE_SHA256,
        "shape": tuple(cache.shape) == CACHE_SHAPE,
        "dtype": cache.dtype == np.float16,
    }
    if not all(checks.values()):
        raise ValueError(
            "training-cache provenance mismatch: "
            f"checks={checks}, sha256={digest}, shape={cache.shape}, "
            f"dtype={cache.dtype}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--revision", default=REPO_REVISION)
    parser.add_argument("--training-cache-only", action="store_true")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    files = (CACHE_FILE,) if args.training_cache_only else FILES
    for filename in files:
        path = Path(
            hf_hub_download(
                REPO_ID,
                filename=filename,
                repo_type="dataset",
                revision=args.revision,
                local_dir=args.destination,
            )
        )
        if filename == CACHE_FILE:
            _validate_cache(path)
        print(path, flush=True)


if __name__ == "__main__":
    main()
