#!/usr/bin/env python3
"""Stage 2: cache layer-10 residual-stream activations for every trace.

Forwards each trace's full_token_ids through DeepSeek-R1-Distill-Llama-8B,
hooks `model.model.layers[10]` (residual stream after block 10), and writes:

    cache_l10/activations.fp16.npy   shape (sum_seq_lens, d_model=4096), fp16
    cache_l10/offsets.npy            shape (N+1,) int64, cumulative seq lengths
    cache_l10/trace_ids.json         ordered list of trace_ids
    cache_l10/meta.json              {layer, model, dtype, n_traces, total_tokens}

Trace i's activations live at activations[offsets[i] : offsets[i+1]].

Memory note: at N=300 and ~600 tokens/trace, the cache is ~1.5 GB on disk.
The forward loop holds one trace's activations on GPU at a time, then moves
to CPU + appends to a growing numpy buffer.

Run from repo root:

    TQDM_DISABLE=1 uv run python -m experiments.phase7_unification.case_studies.backtracking.build_act_cache_backtracking
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.nlp.cache_activations import load_model_and_tokenizer  # noqa: E402
from src.data.nlp.models import resid_hook_target  # noqa: E402

from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    ANCHOR_LAYER,
    CACHE_DIR,
    SUBJECT_MODEL,
    TRACES_PATH,
    ensure_dirs,
)


def _hook(layer_module):
    """Register a forward hook that captures the residual-stream output.

    LlamaDecoderLayer.forward returns a tuple where index 0 is the
    post-block hidden state (the "residual stream after layer i").
    """
    captured: dict[str, torch.Tensor] = {}

    def fn(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach()

    handle = layer_module.register_forward_hook(fn)
    return captured, handle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=ANCHOR_LAYER)
    parser.add_argument("--limit", type=int, default=None, help="cap N for smoke test")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    act_path = CACHE_DIR / "activations.fp16.npy"
    off_path = CACHE_DIR / "offsets.npy"
    ids_path = CACHE_DIR / "trace_ids.json"
    meta_path = CACHE_DIR / "meta.json"

    if act_path.exists() and not args.force:
        print(f"[build_act_cache] {act_path} exists; use --force to rebuild")
        return
    if not TRACES_PATH.exists():
        raise SystemExit(f"missing {TRACES_PATH}; run Stage 0 first")

    print(f"[build_act_cache] loading {SUBJECT_MODEL} for hook on layer {args.layer}")
    model, tokenizer, cfg = load_model_and_tokenizer(SUBJECT_MODEL)
    device = next(model.parameters()).device
    layer_mod = resid_hook_target(model, args.layer, cfg.architecture_family)
    captured, handle = _hook(layer_mod)

    chunks: list[np.ndarray] = []
    offsets: list[int] = [0]
    trace_ids: list[str] = []

    t0 = time.time()
    n = 0
    with TRACES_PATH.open() as f:
        for line in f:
            if args.limit is not None and n >= args.limit:
                break
            rec = json.loads(line)
            full_ids = rec["full_token_ids"]
            ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                _ = model(ids_t, use_cache=False)
            h = captured["h"][0]  # (seq_len, d_model)
            arr = h.to(torch.float16).cpu().numpy()
            assert arr.shape == (len(full_ids), cfg.d_model), arr.shape

            chunks.append(arr)
            offsets.append(offsets[-1] + arr.shape[0])
            trace_ids.append(rec["trace_id"])
            n += 1
            if n % 10 == 0 or n == 1:
                tps = (time.time() - t0) / n
                print(f"  [{n:>4}] {rec['trace_id']:<14} seq={arr.shape[0]} ({tps:.2f}s/trace)")

    handle.remove()

    print(f"[build_act_cache] concatenating {n} traces ({offsets[-1]} tokens total)…")
    activations = np.concatenate(chunks, axis=0).astype(np.float16, copy=False)
    np.save(act_path, activations)
    np.save(off_path, np.asarray(offsets, dtype=np.int64))
    with ids_path.open("w") as f:
        json.dump(trace_ids, f, indent=2)
    meta = {
        "subject_model": SUBJECT_MODEL,
        "layer": args.layer,
        "dtype": "float16",
        "d_model": int(cfg.d_model),
        "n_traces": n,
        "total_tokens": int(offsets[-1]),
        "wall_seconds": round(time.time() - t0, 2),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[build_act_cache] wrote {act_path} ({activations.nbytes/1e9:.2f} GB) + {off_path} + {ids_path}")


if __name__ == "__main__":
    main()
