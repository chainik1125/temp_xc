"""Activation cache for the dialogue turn-length LEVEL (`dialevel`) screen.

DailyDialog rendered as newline-joined turns is a NEW token stream (the
one non-fineweb bundle of the factory batch), so it cannot reuse the
replag caches — one cheap forward pass per model, ~0.81–0.88M tokens
⇒ 3.6–4.3k rows of 128.

**Alignment contract (the builder's, inherited from the interleave
bundle): feed the exact committed `token_ids`, never re-tokenize.** The
flat stream is chunked with the SAME geometry as `replag/cache_acts.py`
(n_prefix = BOS for gemma/llama, 0 for gpt2; non-overlapping content
chunks; document tails dropped), so `(doc, pos) -> (row, cache_pos)`
mapping helpers and the frozen probe machinery transfer unchanged.
Dialogues are short (median 141–153 tokens), so most contribute exactly
one row and 57–62 % of tokens survive chunking — measured, not assumed,
in `design_probe.py` (`chunks_per_doc`, `tokens_kept_frac`).

No null corpus here (unlike interleave): this candidate's mechanism
receipts are the within-dialogue contrast and the anchor-fixed context
shuffle, both of which run on this one cache.

Writes /workspace/dialevel_caches/<model>/{tokens.npz,hs*.npy,
acts_meta.json}; idempotent per completed cache (meta written last).
Three layers are captured (`HS_CAPTURE`) so a conversion-depth
diagnostic never needs a second pass.

Run: .venv/bin/python -m experiments.explorations.task_hunt.dialevel.cache_acts [model ...]
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
CACHE_ROOT = Path("/workspace/dialevel_caches")
TOK_TAG = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}


def chunk_stream(key: str):
    """Flat dialogue stream -> (ids (N, SEQ_LEN), doc_idx, n_prefix).

    Mirrors `replag/build_labels.tokenize_model`'s geometry so the flat
    <-> windowed mapping helpers apply verbatim. A window never crosses
    a dialogue boundary: every row is one contiguous slice of one
    dialogue.
    """
    z = np.load(LABELS / f"dialevel_dailydialog_{TOK_TAG[key]}.npz")
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
    """Re-derive every cached row from the flat stream and assert it
    matches byte-for-byte (the contract check the interleave bundle ran
    before caching; cheap, and it is the only thing standing between a
    silent off-by-one and a whole screen of garbage)."""
    z = np.load(LABELS / f"dialevel_dailydialog_{TOK_TAG[key]}.npz")
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
        print(f"[dialevel cache] hit: {out_dir}")
        return
    ids, doc_idx, n_prefix = chunk_stream(key)
    n_ok = verify_mapping(key, ids, doc_idx, n_prefix)
    N = ids.shape[0]
    np.savez(out_dir / "tokens.npz", ids=ids, doc_idx=doc_idx,
             n_prefix=np.int64(n_prefix))
    print(f"[dialevel cache] {key}: {N} rows × {SEQ_LEN} "
          f"(n_prefix={n_prefix}); flat↔windowed mapping verified "
          f"{n_ok}/{N} rows", flush=True)

    from transformers import AutoModelForCausalLM
    model_id = MODELS[key]["hf"]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    d_model = int(model.config.hidden_size)
    caps = HS_CAPTURE[key]
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
        "d_model": d_model, "dtype": "float16", "n_prefix": n_prefix,
        "mapping_verified_rows": n_ok,
        "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[dialevel cache] DONE {out_dir} in {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    for k in (sys.argv[1:] or list(MODELS)):
        main(k)
