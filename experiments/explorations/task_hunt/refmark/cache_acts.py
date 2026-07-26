"""Activation caches for the refmark (B7) screen — NEW token stream.

The bundle (`../labels/refmark_wildchat_<tok>.npz`) commits the exact
flat token stream per tokenizer; this builder derives the chunk grid
FROM that stream (per conversation, non-overlapping content chunks of
the replag shape: 128 with BOS prefix for llama, 128 plain for gpt2 —
document tails dropped), so label↔cache alignment holds by
construction and the screen's 200-chunk identity assertion is exact.

Writes /workspace/refmark_caches/<model>/tokens.npz (ids, doc_idx,
n_prefix) + hs{k}.npy + acts_meta.json — the replag capture
convention, same layer triples, same fp16 memmaps, idempotent.

Run: .venv/bin/python -m experiments.explorations.task_hunt.refmark.cache_acts [model ...]
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
    HS_CAPTURE,
)

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
CACHE_ROOT = Path("/workspace/refmark_caches")
TAGS = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}


def build_grid(key: str):
    z = np.load(LABELS / f"refmark_wildchat_{TAGS[key]}.npz")
    flat, off = z["token_ids"], z["doc_off"]
    cfg = MODELS[key]
    n_prefix = 1 if cfg["bos"] else 0
    content = SEQ_LEN - n_prefix
    seqs, doc_idx = [], []
    bos = None
    if cfg["bos"]:
        from transformers import AutoTokenizer
        bos = AutoTokenizer.from_pretrained(cfg["hf"]).bos_token_id
    for d in range(len(off) - 1):
        ids = flat[off[d]: off[d + 1]]
        for s in range(0, len(ids) - content + 1, content):
            chunk = ids[s:s + content].tolist()
            if n_prefix:
                chunk = [bos] + chunk
            seqs.append(chunk)
            doc_idx.append(d)
    return (np.asarray(seqs, dtype=np.int32),
            np.asarray(doc_idx, dtype=np.int32), n_prefix)


@torch.no_grad()
def main(key: str):
    out_dir = CACHE_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "acts_meta.json"
    if meta_path.exists():
        print(f"[refmark cache] hit: {out_dir}")
        return
    tok_path = out_dir / "tokens.npz"
    if tok_path.exists():
        c = np.load(tok_path)
        ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    else:
        ids, doc_idx, n_prefix = build_grid(key)
        np.savez(tok_path, ids=ids, doc_idx=doc_idx, n_prefix=n_prefix)
    N = ids.shape[0]
    print(f"[refmark cache] {key}: grid {ids.shape}, n_prefix {n_prefix}",
          flush=True)

    from transformers import AutoModelForCausalLM
    model_id = MODELS[key]["hf"]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    d_model = int(model.config.hidden_size)
    mms = {k: np.lib.format.open_memmap(
        out_dir / f"hs{k}.npy", mode="w+", dtype=np.float16,
        shape=(N, SEQ_LEN, d_model)) for k in HS_CAPTURE[key]}
    t0 = time.time()
    ids_t = torch.from_numpy(ids.astype(np.int64))
    B = BATCH[key]
    for s in range(0, N, B):
        e = min(s + B, N)
        out = model(ids_t[s:e].cuda(), output_hidden_states=True,
                    use_cache=False)
        for k in HS_CAPTURE[key]:
            mms[k][s:e] = (out.hidden_states[k].detach()
                           .to(torch.float16).cpu().numpy())
        if (s // B) % 20 == 0:
            el = time.time() - t0
            print(f"  {e}/{N} ({el:.0f}s, est {el / max(e, 1) * N:.0f}s)",
                  flush=True)
    for m in mms.values():
        m.flush()
    del mms, model
    torch.cuda.empty_cache()
    meta_path.write_text(json.dumps({
        "model_id": model_id, "hs_capture": HS_CAPTURE[key],
        "n_seqs": int(N), "seq_len": SEQ_LEN, "d_model": d_model,
        "dtype": "float16", "source_bundle": f"refmark_wildchat_"
        f"{TAGS[key]}.npz (grid derived from the committed stream)",
        "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[refmark cache] DONE {key} in {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["gpt2", "llama31_8b"]):
        main(k)
