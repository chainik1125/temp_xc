"""Rebuild probe cache at S=32 by per-example slicing the existing
LAST_N=128 right-padded cache. Output is left-aligned (real tokens
right-aligned, zeros left-padded for short sentences).

Per-example transformation:
  src (128-frame, right-padded): real at [0, last_idx[i]], pad at rest
  dst (32-frame, left-aligned):  real at [first_real[i], 31], zero at start
  where first_real[i] = max(0, 32 - n_real[i]) and n_real[i] = min(32, last_idx[i]+1)

Saves to experiments/phase7_unification/results/probe_cache_S32/<task>/
{acts_anchor.npz, acts_mlc.npz, acts_mlc_tail.npz, meta.json}

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.rebuild_probe_cache_s32
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import (
    PROBE_CACHE_DIR, OUT_DIR, banner,
)


S_NEW = 32
NEW_CACHE_DIR = OUT_DIR / "probe_cache_S32"


def _slice_per_example(arr: np.ndarray, last_idx: np.ndarray, S: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-example slice the last min(S, n_real) real tokens.

    Args:
      arr: (N, LAST_N=128, ...) right-padded numpy array (any extra dims preserved)
      last_idx: (N,) position of last real token in 128-frame
      S: new tail length

    Returns:
      (sliced, first_real_S):
        sliced: (N, S, ...) — real tokens right-aligned, zeros at start for short examples
        first_real_S: (N,) — position of first real token in S-frame
    """
    N = arr.shape[0]
    rest = arr.shape[2:]  # any dims after (N, LAST_N)
    out = np.zeros((N, S) + rest, dtype=arr.dtype)
    first_real_S = np.zeros(N, dtype=np.int64)
    for i in range(N):
        li = int(last_idx[i])
        n_real = min(li + 1, S)
        src_start = li - n_real + 1
        out[i, S - n_real:] = arr[i, src_start:li + 1]
        first_real_S[i] = S - n_real
    return out, first_real_S


def rebuild_one_task(task_dir: Path, out_dir: Path) -> None:
    """Convert one task's cache files to S=32 left-aligned."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) acts_anchor.npz: (N, 128, d) → (N, 32, d) per-example sliced
    src_anchor = task_dir / "acts_anchor.npz"
    if src_anchor.exists():
        with np.load(src_anchor) as z:
            tr_acts = z["train_acts"]
            te_acts = z["test_acts"]
            tr_last = z["train_last_idx"]
            te_last = z["test_last_idx"]
            tr_y = z["train_labels"]
            te_y = z["test_labels"]
        tr_S, tr_first = _slice_per_example(tr_acts, tr_last, S_NEW)
        te_S, te_first = _slice_per_example(te_acts, te_last, S_NEW)
        np.savez(out_dir / "acts_anchor.npz",
                 train_acts=tr_S, test_acts=te_S,
                 train_first_real=tr_first, test_first_real=te_first,
                 train_labels=tr_y, test_labels=te_y)

    # 2) acts_mlc.npz: (N, 5, d) — last token only, no slicing needed
    src_mlc = task_dir / "acts_mlc.npz"
    if src_mlc.exists():
        with np.load(src_mlc) as z:
            np.savez(out_dir / "acts_mlc.npz",
                     train_acts=z["train_acts"], test_acts=z["test_acts"],
                     train_labels=z["train_labels"], test_labels=z["test_labels"])

    # 3) acts_mlc_tail.npz: (N, 128, 5, d) → (N, 32, 5, d) per-example sliced
    src_mlc_tail = task_dir / "acts_mlc_tail.npz"
    if src_mlc_tail.exists():
        with np.load(src_mlc_tail) as z:
            tr_acts = z["train_acts"]
            te_acts = z["test_acts"]
            tr_y = z["train_labels"]
            te_y = z["test_labels"]
        # Use the SAME last_idx from anchor — same texts, same tokenization
        tr_S, tr_first = _slice_per_example(tr_acts, tr_last, S_NEW)
        te_S, te_first = _slice_per_example(te_acts, te_last, S_NEW)
        np.savez(out_dir / "acts_mlc_tail.npz",
                 train_acts=tr_S, test_acts=te_S,
                 train_first_real=tr_first, test_first_real=te_first,
                 train_labels=tr_y, test_labels=te_y)

    # 4) meta.json: copy + annotate the new S
    src_meta = task_dir / "meta.json"
    if src_meta.exists():
        meta = json.loads(src_meta.read_text())
        meta["last_n"] = S_NEW
        meta["padding"] = "left_aligned_real_tokens"
        meta["source_cache"] = str(task_dir)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    banner(__file__)
    NEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    task_dirs = [d for d in sorted(PROBE_CACHE_DIR.iterdir()) if d.is_dir()]
    print(f"Found {len(task_dirs)} task dirs to rebuild at S={S_NEW}")
    t0 = time.time()
    for i, src in enumerate(task_dirs):
        out = NEW_CACHE_DIR / src.name
        if (out / "acts_anchor.npz").exists() and (out / "acts_mlc_tail.npz").exists():
            print(f"  [{i+1}/{len(task_dirs)}] {src.name}: already done, skip")
            continue
        ti = time.time()
        rebuild_one_task(src, out)
        print(f"  [{i+1}/{len(task_dirs)}] {src.name}: done in {time.time()-ti:.1f}s")
    print(f"Total rebuild time: {(time.time()-t0)/60:.1f} min")
    print(f"New cache at: {NEW_CACHE_DIR}")


if __name__ == "__main__":
    main()
