"""Layer-sweep activation extraction (059a66239 P2 sweep (a); CARD.md).

Two passes per model, both on the committed dialevel DailyDialog
stream (never re-tokenized):

1. The CANONICAL dialevel cache, built by calling
   `dialevel.cache_acts.main(key)` VERBATIM — its root, its
   `HS_CAPTURE` layers, its meta-last idempotence contract. The pod
   has no dialevel caches today; this pass heals that for everyone,
   and `build_rows` (both screen stacks) reads `tokens.npz` from that
   root, so it is a hard prerequisite anyway.

2. The EXTRA layers of the sweep — only those NOT in `HS_CAPTURE` —
   into this sweep's own root (`/workspace/layer_sweep_caches`), same
   file format (`hs{k}.npy` fp16 memmap (N, SEQ_LEN, d_model)), same
   meta-last contract, chunk geometry re-derived with the SAME
   `chunk_stream` and asserted equal to the canonical `tokens.npz`.
   A separate root so the canonical cache's `acts_meta.json` never
   lies about what its directory holds.

CAPTURE below is the UNION of both layer-semantics readings (LOG
6307ce5a3): capture is the expensive step, so the mac-local ruling
only selects which cells the frozen scorer reads — no re-extraction
under either reading.

Run: .venv/bin/python -m experiments.explorations.task_hunt.layer_sweep.extract [model ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.task_hunt.dialevel import cache_acts as dial_cache
from experiments.explorations.task_hunt.dialevel.cache_acts import (
    CACHE_ROOT as DIAL_ROOT,
    chunk_stream,
)
from experiments.explorations.task_hunt.replag.build_labels import MODELS, SEQ_LEN
from experiments.explorations.task_hunt.replag.cache_acts import BATCH, HS_CAPTURE

LS_ROOT = Path("/workspace/layer_sweep_caches")

# Union of resid_post-L (default) and hs-index readings of the
# directive's layer lists — see CARD.md § 2 and LOG 6307ce5a3.
CAPTURE = {
    "llama31_8b": [8, 14, 15, 22, 29],
    "gemma2_2b": [7, 14, 21],
}


def extra_layers(key: str) -> list[int]:
    return [k for k in CAPTURE[key] if k not in HS_CAPTURE[key]]


def acts_path(key: str, hs: int) -> Path:
    """Resolve an hs layer to whichever root holds it (canonical
    dialevel for HS_CAPTURE layers, this sweep's root otherwise)."""
    if hs in HS_CAPTURE[key]:
        return DIAL_ROOT / key / f"hs{hs}.npy"
    return LS_ROOT / key / f"hs{hs}.npy"


@torch.no_grad()
def extract_extra(key: str):
    caps = extra_layers(key)
    out_dir = LS_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "acts_meta.json"
    if meta_path.exists():
        print(f"[layer_sweep cache] hit: {out_dir}")
        return
    if not caps:
        meta_path.write_text(json.dumps({"extra_layers": []}))
        return

    # Same geometry as the canonical pass; assert against its tokens.npz
    # (which pass 1 must have written) so the row spaces are identical.
    ids, doc_idx, n_prefix = chunk_stream(key)
    c = np.load(DIAL_ROOT / key / "tokens.npz")
    assert np.array_equal(c["ids"], ids) and np.array_equal(
        c["doc_idx"], doc_idx) and int(c["n_prefix"]) == n_prefix, \
        "layer_sweep chunk geometry != canonical dialevel tokens.npz"
    N = ids.shape[0]

    from transformers import AutoModelForCausalLM
    model_id = MODELS[key]["hf"]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    d_model = int(model.config.hidden_size)
    mms = {k: np.lib.format.open_memmap(
        out_dir / f"hs{k}.npy", mode="w+", dtype=np.float16,
        shape=(N, SEQ_LEN, d_model)) for k in caps}
    t0 = time.time()
    ids_t = torch.from_numpy(ids.astype(np.int64))
    B = BATCH[key]
    for s in range(0, N, B):
        e = min(s + B, N)
        out = model(ids_t[s:e].cuda(), output_hidden_states=True,
                    use_cache=False)
        for k in caps:
            mms[k][s:e] = (out.hidden_states[k].detach()
                           .to(torch.float16).cpu().numpy())
    for m in mms.values():
        m.flush()
    del mms, model
    torch.cuda.empty_cache()

    for k in caps:
        arr = np.load(out_dir / f"hs{k}.npy", mmap_mode="r")
        sample = arr[min(3, N - 1), 100, :].astype(np.float32)
        assert np.isfinite(sample).all() and np.linalg.norm(sample) > 0, \
            f"degenerate activations at hs{k}"
    meta_path.write_text(json.dumps({
        "model_id": model_id, "extra_layers": caps,
        "union_capture": CAPTURE[key], "n_seqs": N, "seq_len": SEQ_LEN,
        "d_model": d_model, "dtype": "float16", "n_prefix": n_prefix,
        "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[layer_sweep cache] DONE {out_dir} ({caps}) in "
          f"{time.time() - t0:.0f}s", flush=True)


def main(key: str):
    dial_cache.main(key)      # pass 1: canonical cache, verbatim module
    extract_extra(key)        # pass 2: sweep-only layers, own root


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["llama31_8b", "gemma2_2b"]):
        main(k)
