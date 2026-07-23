"""Phase 3 (cache) — multi-layer activation caches on the Ward stream.

Forwards a subject model over the canonical 4044 × 128 stream
(`build_ward_stream.py` output) capturing the embedding layer
(hidden_states[0]) plus resid_post at layers 0, 2, …, 30
(hidden_states[k+1] — HF hidden_states[k+1] IS the post-block residual
of layer k, identical to the paper's `resid_post` forward hook on
`model.model.layers[k]`).

The framework's `build_activation_cache` (src/temp_bench/data/real_lm.py)
is single-layer AND its generic HF-datasets loader cannot materialize the
`ward_backtracking_math500` corpus on this branch (the ward corpus loader
lives only on origin/final) — so this exploration-local cacher replicates
the origin/final capture convention (fp16, (N, 128, d_model) per layer)
with one forward sweep for all 17 capture points instead of 17 sweeps.
No `temp_bench/core/` edits.

Usage:
  .venv/bin/python -m experiments.explorations.conversion_depth.cache_depth base
  .venv/bin/python -m experiments.explorations.conversion_depth.cache_depth distill

Writes /workspace/conv_depth_caches/<tag>/hs{k}.npy (float16 memmaps)
+ meta.json. Idempotent per completed cache (meta.json written last).
KEEP these caches — they are the input to the follow-up TXC-tracking
session (briefing § session limits).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODELS = {
    "base": "NousResearch/Meta-Llama-3.1-8B",
    "distill": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}
STREAM_DIR = Path("/workspace/conv_depth_caches/ward_stream")
CACHE_ROOT = Path("/workspace/conv_depth_caches")
LAYERS = list(range(0, 31, 2))            # resid_post of blocks 0,2,...,30
HS_CAPTURE = [0] + [k + 1 for k in LAYERS]  # hidden_states indices
BATCH = 32
SEQ_LEN = 128


@torch.no_grad()
def main(tag: str):
    model_id = MODELS[tag]
    out_dir = CACHE_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        print(f"[cache_depth] cache hit: {out_dir}")
        return

    ids = np.load(STREAM_DIR / "token_ids.npy")
    N = ids.shape[0]
    assert ids.shape == (N, SEQ_LEN)

    from transformers import AutoModelForCausalLM
    print(f"[cache_depth] loading {model_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    d_model = int(model.config.hidden_size)

    mms = {k: np.lib.format.open_memmap(
        out_dir / f"hs{k}.npy", mode="w+", dtype=np.float16,
        shape=(N, SEQ_LEN, d_model)) for k in HS_CAPTURE}
    print(f"[cache_depth] {len(mms)} capture points × "
          f"{N}×{SEQ_LEN}×{d_model} fp16 "
          f"({sum(m.nbytes for m in mms.values()) / 1e9:.0f} GB)", flush=True)

    t0 = time.time()
    ids_t = torch.from_numpy(ids)
    for s in range(0, N, BATCH):
        e = min(s + BATCH, N)
        batch = ids_t[s:e].cuda()
        out = model(batch, output_hidden_states=True, use_cache=False)
        for k in HS_CAPTURE:
            mms[k][s:e] = (out.hidden_states[k].detach()
                           .to(torch.float16).cpu().numpy())
        if (s // BATCH) % 20 == 0:
            el = time.time() - t0
            print(f"  {e}/{N} ({el:.0f}s, {el / max(e, 1) * N:.0f}s est total)",
                  flush=True)
    for m in mms.values():
        m.flush()
    del mms

    # sanity: finite + nonzero norms at a probe point
    arr = np.load(out_dir / f"hs{LAYERS[len(LAYERS) // 2] + 1}.npy",
                  mmap_mode="r")
    sample = arr[3, 7, :].astype(np.float32)
    assert np.isfinite(sample).all() and np.linalg.norm(sample) > 0

    meta_path.write_text(json.dumps({
        "model_id": model_id,
        "hs_capture": HS_CAPTURE,
        "resid_post_layers": LAYERS,
        "n_seqs": N, "seq_len": SEQ_LEN, "d_model": d_model,
        "dtype": "float16",
        "stream": str(STREAM_DIR / "token_ids.npy"),
        "wall_seconds": round(time.time() - t0, 1),
    }, indent=2))
    print(f"[cache_depth] DONE {tag} in {time.time() - t0:.0f}s → {out_dir}",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
