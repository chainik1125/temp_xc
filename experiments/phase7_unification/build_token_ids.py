"""Build token_ids.npy for Phase 7 base activation cache from FineWeb.

Streams HuggingFaceFW/fineweb sample-10BT, tokenizes with the
gemma-2-2b tokenizer, packs into (24000, 128) int64 array, saves
to data/cached_activations/gemma-2-2b/fineweb/token_ids.npy.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.build_token_ids
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import CACHE_DIR, SUBJECT_MODEL


N_SEQS = 24_000
CTX = 128


def main():
    print(f"Target: {CACHE_DIR / 'token_ids.npy'}")
    print(f"        N_SEQS={N_SEQS}, CTX={CTX}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / "token_ids.npy"
    if out_path.exists():
        print(f"  exists, skipping. Delete to rebuild.")
        return

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT",
                      split="train", streaming=True)

    arr = np.zeros((N_SEQS, CTX), dtype=np.int64)
    n_kept = 0
    n_seen = 0
    for sample in ds:
        n_seen += 1
        text = sample.get("text") or ""
        if not text: continue
        ids = tok.encode(text, add_special_tokens=True, truncation=True, max_length=CTX)
        if len(ids) < CTX:
            continue   # need full-context examples
        arr[n_kept] = ids[:CTX]
        n_kept += 1
        if n_kept % 2000 == 0:
            print(f"  {n_kept}/{N_SEQS} (saw {n_seen} samples)")
        if n_kept >= N_SEQS:
            break

    if n_kept < N_SEQS:
        raise SystemExit(f"only got {n_kept}/{N_SEQS}")

    np.save(out_path, arr)
    print(f"  saved {out_path} (shape {arr.shape}, dtype {arr.dtype})")


if __name__ == "__main__":
    main()
