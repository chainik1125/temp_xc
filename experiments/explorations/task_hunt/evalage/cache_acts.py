"""Activation caches for the EVALAGE screen (SCREEN_CARD.md § 2):
mac-d's `sycgen/cache_acts.py` transplant over the evalage screen grids
— one forward pass per model, SINGLE-layer capture (SCREEN_HS).

Alignment contract inherited verbatim: feed the exact committed grid
`token_ids`, never re-tokenize; chunking geometry = replag's (n_prefix
= BOS for gemma/llama, 0 for gpt2; non-overlapping content chunks; doc
tails dropped); every cached row re-derived from the flat stream and
asserted byte-identical before any forward pass.

Writes /workspace/evalage_caches/<model>/{tokens.npz, hs<K>.npy,
acts_meta.json}; idempotent per completed cache.

Run: .venv/bin/python -m experiments.explorations.task_hunt.evalage.cache_acts [model ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.task_hunt.replag.build_labels import (
    MODELS,
    SEQ_LEN,
)
from experiments.explorations.task_hunt.replag.cache_acts import (
    BATCH,
    SCREEN_HS,
)

GRIDS = Path(__file__).resolve().parent / "grids"
CACHE_ROOT = Path("/workspace/evalage_caches")
TOK_TAG = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}


def chunk_stream(key: str):
    z = np.load(GRIDS / f"elicit_evalage_screen_{TOK_TAG[key]}.npz")
    flat, off = z["token_ids"], z["doc_off"]
    n_prefix = 1 if MODELS[key]["bos"] else 0
    content = SEQ_LEN - n_prefix
    bos_id = None
    if n_prefix:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODELS[key]["hf"])
        bos_id = tok.bos_token_id
    rows, doc_idx = [], []
    for d in range(len(off) - 1):
        seg = flat[off[d]:off[d + 1]]
        for s in range(0, len(seg) - content + 1, content):
            chunk = seg[s:s + content]
            if n_prefix:
                chunk = np.concatenate([[bos_id], chunk])
            rows.append(chunk)
            doc_idx.append(d)
    ids = np.asarray(rows, dtype=np.int32)
    return ids, np.asarray(doc_idx, dtype=np.int32), n_prefix


def verify_mapping(key: str, ids, doc_idx, n_prefix) -> int:
    z = np.load(GRIDS / f"elicit_evalage_screen_{TOK_TAG[key]}.npz")
    flat, off = z["token_ids"], z["doc_off"]
    content = SEQ_LEN - n_prefix
    seen: dict = {}
    for i, d in enumerate(doc_idx.tolist()):
        c = seen.get(d, 0)
        seen[d] = c + 1
        want = flat[off[d] + c * content: off[d] + (c + 1) * content]
        got = ids[i, n_prefix:]
        assert np.array_equal(want, got), f"row {i} (doc {d} chunk {c})"
    return len(doc_idx)


@torch.no_grad()
def main(key: str):
    out_dir = CACHE_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "acts_meta.json"
    if meta_path.exists():
        print(f"[evalage cache] hit: {out_dir}")
        return
    ids, doc_idx, n_prefix = chunk_stream(key)
    n_ok = verify_mapping(key, ids, doc_idx, n_prefix)
    N = ids.shape[0]
    np.savez(out_dir / "tokens.npz", ids=ids, doc_idx=doc_idx,
             n_prefix=np.int64(n_prefix))
    print(f"[evalage cache] {key}: {N} rows × {SEQ_LEN} "
          f"(n_prefix={n_prefix}); mapping verified {n_ok}/{N}",
          flush=True)

    from transformers import AutoModelForCausalLM
    model_id = MODELS[key]["hf"]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    d_model = int(model.config.hidden_size)
    hs = SCREEN_HS[key]
    mm = np.lib.format.open_memmap(
        out_dir / f"hs{hs}.npy", mode="w+", dtype=np.float16,
        shape=(N, SEQ_LEN, d_model))
    t0 = time.time()
    ids_t = torch.from_numpy(ids.astype(np.int64))
    B = BATCH[key]
    for s in range(0, N, B):
        e = min(s + B, N)
        out = model(ids_t[s:e].cuda(), output_hidden_states=True,
                    use_cache=False)
        mm[s:e] = (out.hidden_states[hs].detach()
                   .to(torch.float16).cpu().numpy())
        if (s // B) % 200 == 0:
            print(f"[evalage cache] {key} {e}/{N} "
                  f"({e / max(time.time() - t0, 1e-9):.0f} rows/s)",
                  flush=True)
    mm.flush()
    del mm, model
    torch.cuda.empty_cache()

    arr = np.load(out_dir / f"hs{hs}.npy", mmap_mode="r")
    sample = arr[min(3, N - 1), 100, :].astype(np.float32)
    assert np.isfinite(sample).all() and np.linalg.norm(sample) > 0, \
        "degenerate activations"
    meta_path.write_text(json.dumps({
        "model_id": model_id, "substrate": "elicit_evalage_v1",
        "hs_capture": [hs], "screen_hs": hs, "n_seqs": N,
        "seq_len": SEQ_LEN, "d_model": d_model, "dtype": "float16",
        "n_prefix": n_prefix, "mapping_verified_rows": n_ok,
        "single_layer_capture": "reask_hr § 2 convention inherited",
        "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[evalage cache] DONE {out_dir} in {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["gpt2", "gemma2_2b", "llama31_8b"]):
        main(k)
