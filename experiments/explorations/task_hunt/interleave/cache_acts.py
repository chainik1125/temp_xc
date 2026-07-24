"""Activation cache for the interleaved-document (`tss`) screen.

The interleaved corpus is a NEW token stream (fineweb text re-composed
into strictly alternating 1–4-sentence blocks from lexically-matched
document pairs), so unlike the novelty/punctint bundles it cannot reuse
the replag fineweb caches — it needs one cheap forward pass per model.
~335k tokens per tokenizer ⇒ ~2.6k rows of 128; minutes on an H100.

**Alignment contract (from the builder's card draft): feed the exact
`token_ids`, never re-tokenize.** This module chunks the committed flat
stream into SEQ_LEN rows with the SAME geometry as
`replag/cache_acts.py` (n_prefix = BOS for gemma/llama, 0 for gpt2;
non-overlapping content chunks; document tails dropped), so the
`(doc, pos) -> (row, cache_pos)` mapping helpers and the frozen probe
machinery transfer unchanged.

Also caches the **shuffled-block NULL corpus** (`null_perm` applied to
the token stream) when `--null` is passed: the card's mechanism receipt
compares reader performance on the real vs the incoherent stream, and
that requires its own forward pass.

Writes /workspace/interleave_caches/<model>[_null]/{tokens.npz,hs*.npy,
acts_meta.json}; idempotent per completed cache (meta written last).

Run: .venv/bin/python -m experiments.explorations.task_hunt.interleave.cache_acts [model ...] [--null]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.task_hunt.replag.build_labels import MODELS, SEQ_LEN
from experiments.explorations.task_hunt.replag.cache_acts import (
    BATCH,
    HS_CAPTURE,
    SCREEN_HS,
)

LABELS = Path(__file__).resolve().parents[1] / "labels"
CACHE_ROOT = Path("/workspace/interleave_caches")
TOK_TAG = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}


def chunk_stream(key: str, null: bool):
    """Flat interleaved stream -> (ids (N, SEQ_LEN), doc_idx, n_prefix).

    Mirrors `replag/build_labels.tokenize_model`'s geometry so the flat
    <-> windowed mapping helpers apply verbatim.
    """
    z = np.load(LABELS / f"interleave_fineweb_{TOK_TAG[key]}.npz")
    flat, off = z["token_ids"], z["doc_off"]
    if null:
        # The null corpus IS the same tokens re-ordered by null_perm
        # (the builder's shuffled-block permutation), with the labels
        # recomputed as tss_null/source_null.
        flat = flat[z["null_perm"]]
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


@torch.no_grad()
def main(key: str, null: bool = False):
    out_dir = CACHE_ROOT / (f"{key}_null" if null else key)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "acts_meta.json"
    if meta_path.exists():
        print(f"[interleave cache] hit: {out_dir}")
        return
    ids, doc_idx, n_prefix = chunk_stream(key, null)
    N = ids.shape[0]
    np.savez(out_dir / "tokens.npz", ids=ids, doc_idx=doc_idx,
             n_prefix=np.int64(n_prefix))
    print(f"[interleave cache] {key}{'/null' if null else ''}: "
          f"{N} rows × {SEQ_LEN} (n_prefix={n_prefix})", flush=True)

    from transformers import AutoModelForCausalLM
    model_id = MODELS[key]["hf"]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    d_model = int(model.config.hidden_size)
    # Screen layer only for the null corpus (it is a control, not a grid).
    caps = [SCREEN_HS[key]] if null else HS_CAPTURE[key]
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

    arr = np.load(out_dir / f"hs{SCREEN_HS[key]}.npy", mmap_mode="r")
    sample = arr[min(3, N - 1), 100, :].astype(np.float32)
    assert np.isfinite(sample).all() and np.linalg.norm(sample) > 0, \
        "degenerate activations"
    meta_path.write_text(json.dumps({
        "model_id": model_id, "hs_capture": caps,
        "screen_hs": SCREEN_HS[key], "n_seqs": N, "seq_len": SEQ_LEN,
        "d_model": d_model, "dtype": "float16", "null_corpus": null,
        "n_prefix": n_prefix,
        "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[interleave cache] DONE {out_dir} in {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_null = "--null" in sys.argv
    for k in (args or list(MODELS)):
        main(k, null=False)
        if do_null:
            main(k, null=True)
