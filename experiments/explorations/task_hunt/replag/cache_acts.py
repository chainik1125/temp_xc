"""Activation caches for the replag screen (CARD.md model × layer table).

One forward sweep per model over its own token grid
(`/workspace/replag_caches/<model>/tokens.npz`, from build_labels.py),
capturing the card's screen layer + cached alternates as fp16 memmaps —
the cache_depth.py capture convention (hidden_states[k] indices; for a
model with resid_post at layer L, hs index = L + 1).

Writes /workspace/replag_caches/<model>/hs{k}.npy + acts_meta.json
(idempotent per completed cache; meta written last).

Run: .venv/bin/python -m experiments.explorations.task_hunt.replag.cache_acts [model ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.task_hunt.replag.build_labels import (
    CACHE_ROOT, MODELS, SEQ_LEN,
)

# CARD.md: screen layer first, then cached alternates
HS_CAPTURE = {
    "gpt2": [7, 4, 10],
    "gemma2_2b": [14, 8, 20],
    "llama31_8b": [14, 8, 22],
}
SCREEN_HS = {k: v[0] for k, v in HS_CAPTURE.items()}
BATCH = {"gpt2": 256, "gemma2_2b": 64, "llama31_8b": 32}


@torch.no_grad()
def main(key: str):
    out_dir = CACHE_ROOT / key
    meta_path = out_dir / "acts_meta.json"
    if meta_path.exists():
        print(f"[cache_acts] hit: {out_dir}")
        return
    ids = np.load(out_dir / "tokens.npz")["ids"]
    N = ids.shape[0]
    assert ids.shape == (N, SEQ_LEN)

    from transformers import AutoModelForCausalLM
    model_id = MODELS[key]["hf"]
    print(f"[cache_acts] loading {model_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    d_model = int(model.config.hidden_size)

    mms = {k: np.lib.format.open_memmap(
        out_dir / f"hs{k}.npy", mode="w+", dtype=np.float16,
        shape=(N, SEQ_LEN, d_model)) for k in HS_CAPTURE[key]}
    print(f"[cache_acts] {key}: {len(mms)} layers × {N}×{SEQ_LEN}×{d_model} "
          f"({sum(m.nbytes for m in mms.values()) / 1e9:.1f} GB)", flush=True)

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

    arr = np.load(out_dir / f"hs{SCREEN_HS[key]}.npy", mmap_mode="r")
    sample = arr[3, 100, :].astype(np.float32)
    assert np.isfinite(sample).all() and np.linalg.norm(sample) > 0

    meta_path.write_text(json.dumps({
        "model_id": model_id, "hs_capture": HS_CAPTURE[key],
        "screen_hs": SCREEN_HS[key], "n_seqs": N, "seq_len": SEQ_LEN,
        "d_model": d_model, "dtype": "float16",
        "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"[cache_acts] DONE {key} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for key in (sys.argv[1:] or list(MODELS)):
        main(key)
