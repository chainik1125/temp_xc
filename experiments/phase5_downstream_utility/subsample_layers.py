"""Subsample each L11-L15 cache to the first 6000 sequences.

My training pipeline preloads only 6000 sequences from disk to GPU; the
other 18000 of each 24000-seq cache are dead weight on a tight MooseFS
quota. This script rewrites each layer cache with just the first 6000
rows, saving ~52 GB total across 5 layers.

Run once, before the probe cache build. After this, the training driver
is unchanged — `_preload_single` / `_preload_multilayer` still take the
first 6000 rows; they're just the whole file now.

Usage:
    .venv/bin/python /tmp/subsample_layers.py
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np

CACHE_DIR = Path("/workspace/temp_xc/data/cached_activations/gemma-2-2b-it/fineweb")
KEEP_N = 6000

layers = [11, 12, 13, 14, 15]

for layer in layers:
    src = CACHE_DIR / f"resid_L{layer}.npy"
    if not src.exists():
        print(f"  L{layer}: missing, skip")
        continue
    arr = np.load(src, mmap_mode="r")
    N, seq_len, d = arr.shape
    if N <= KEEP_N:
        print(f"  L{layer}: already at {N} rows, skip")
        continue

    t0 = time.time()
    print(f"  L{layer}: {N} -> {KEEP_N} rows (dtype={arr.dtype})")
    # Materialize the slice to a temp file in /tmp (local, no MooseFS
    # write-quota issues on the transient file), then atomic rename.
    tmp = Path(f"/tmp/resid_L{layer}_{KEEP_N}.npy")
    np.save(tmp, np.asarray(arr[:KEEP_N]))
    # Delete original BEFORE moving to free quota headroom
    src.unlink()
    shutil.move(str(tmp), str(src))
    new_size = src.stat().st_size
    print(f"  L{layer}: done ({new_size/1e9:.2f} GB, {time.time()-t0:.1f}s)")

# Also subsample token_ids for consistency (not strictly needed for
# training but keeps alignment semantics clean).
tok_path = CACHE_DIR / "token_ids.npy"
if tok_path.exists():
    tok = np.load(tok_path, mmap_mode="r")
    if tok.shape[0] > KEEP_N:
        tmp = Path("/tmp/token_ids_6000.npy")
        np.save(tmp, np.asarray(tok[:KEEP_N]))
        tok_path.unlink()
        shutil.move(str(tmp), str(tok_path))
        print(f"  token_ids: {tok.shape[0]} -> {KEEP_N}")
print("Done.")
