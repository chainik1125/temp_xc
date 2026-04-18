"""Extend the gemma-2-2b-it FineWeb activation cache with MLC layers.

The existing cache at data/cached_activations/gemma-2-2b-it/fineweb/
has token_ids.npy + resid_L13.npy + resid_L25.npy (Phase 4 layers).
MLC needs a contiguous 5-layer window; we use L11-L15 centred on L13,
so L11/L12/L14/L15 must be added.

We **reuse the existing token_ids.npy** rather than re-tokenize from
FineWeb; re-streaming would give different texts and break the
alignment with the already-cached L13/L25 activations.

Usage (from repo root):
    HF_HOME=/workspace/hf_cache TQDM_DISABLE=1 \\
      .venv/bin/python experiments/phase5_downstream_utility/build_multilayer_cache.py

Resumable: existing .npy files with the right shape are skipped, so
a partial run can be restarted safely.

Estimated runtime on A40: ~45 min for 4 layers at batch=32.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache-dir",
        type=str,
        default="/workspace/temp_xc/data/cached_activations/gemma-2-2b-it/fineweb",
    )
    p.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    p.add_argument("--layers", type=int, nargs="+", default=[11, 12, 14, 15])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dtype", type=str, default="bfloat16")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    token_path = cache_dir / "token_ids.npy"
    if not token_path.exists():
        raise FileNotFoundError(
            f"Expected token_ids.npy at {token_path}. Build the initial "
            "cache first via src.data.nlp.cache_activations."
        )

    tokens = np.load(token_path, mmap_mode="r")
    n_seq, seq_len = tokens.shape
    print(f"Loaded token_ids.npy shape={tokens.shape}")

    # Decide which layers still need caching.
    d_model = 2304  # gemma-2-2b-it residual width
    to_cache: list[int] = []
    for layer in args.layers:
        out_path = cache_dir / f"resid_L{layer}.npy"
        if out_path.exists():
            existing = np.load(out_path, mmap_mode="r")
            if existing.shape == (n_seq, seq_len, d_model):
                print(f"  resid_L{layer}: already cached — skipping")
                continue
        to_cache.append(layer)

    if not to_cache:
        print("All layers already cached. Nothing to do.")
        return

    print(f"Layers to cache: {to_cache}")

    # Load model.
    from transformers import AutoModelForCausalLM
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    print(f"Loading {args.model} ({args.dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cuda"
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Register hooks on the target layers' post-residual output.
    # For gemma, the residual stream *after* layer L is read at
    # model.model.layers[L].
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, output):
            acts = output[0] if isinstance(output, tuple) else output
            # gemma residuals are already (B, L, d)
            captured[layer_idx] = acts.detach().float().cpu()
        return hook_fn

    for layer in to_cache:
        target = model.model.layers[layer]
        hooks.append(target.register_forward_hook(make_hook(layer)))

    # Allocate memmaps for each target layer. Match the existing cache's
    # dtype (fp16) — L13 and L25 on disk are fp16, and using fp32 here
    # would both waste 54 GB of disk and make stacking layers into MLC
    # inputs require a dtype cast. The model forward is in bf16 anyway;
    # fp16 on disk is a lossless round-trip for that path.
    mmaps: dict[int, np.ndarray] = {}
    for layer in to_cache:
        out_path = cache_dir / f"resid_L{layer}.npy"
        mmaps[layer] = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float16,
            shape=(n_seq, seq_len, d_model),
        )
        print(f"  Will write {out_path} ({mmaps[layer].nbytes / 1e9:.2f} GB)")

    # Stream batches.
    tokens_t = torch.from_numpy(np.asarray(tokens))  # eager load of int64 ids is cheap
    t0 = time.time()
    bs = args.batch_size
    with torch.no_grad():
        for start in range(0, n_seq, bs):
            end = min(start + bs, n_seq)
            batch = tokens_t[start:end].to("cuda")
            captured.clear()
            model(batch)
            for layer in to_cache:
                acts = captured[layer]
                if acts.shape[-1] != d_model:
                    acts = acts[..., :d_model]
                # Cast to fp16 on CPU before writing to match existing cache.
                mmaps[layer][start:end] = acts.to(torch.float16).numpy()
            if (start // bs) % 25 == 0:
                elapsed = time.time() - t0
                rate = (end / max(1, elapsed))
                eta = (n_seq - end) / max(1, rate)
                print(f"  [{end}/{n_seq}] {rate:.1f} seq/s, ETA {eta/60:.1f} min")

    for h in hooks:
        h.remove()
    for layer in to_cache:
        mmaps[layer].flush()

    # Update layer_specs.json so downstream code knows about the new layers.
    specs_path = cache_dir / "layer_specs.json"
    if specs_path.exists():
        specs = json.loads(specs_path.read_text())
    else:
        specs = {
            "model": "gemma-2-2b-it",
            "hf_path": args.model,
            "d_model": d_model,
            "layer_specs": {},
            "mode": "forward",
        }
    for layer in to_cache:
        key = f"resid_L{layer}"
        specs["layer_specs"][key] = {
            "layer": layer,
            "component": "resid",
            "d_act": d_model,
            "label": key,
            "family": "gemma",
        }
    specs_path.write_text(json.dumps(specs, indent=2))
    print(f"Updated {specs_path}")
    print(f"Done in {(time.time() - t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
