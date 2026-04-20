"""Build multiple residual-stream activation layers in a single Gemma load.

Loads Gemma-2-2B-IT once, hooks all requested layers, runs one forward
pass per batch of cached token_ids, writes each layer's activations to
its own .npy memmap. Resumable per-layer via `.progress.json` sidecars.

Usage:
    PHASE5_REPO=/home/elysium/temp_xc TQDM_DISABLE=1 \
      .venv/bin/python experiments/phase5_downstream_utility/build_multilayer_cache_batch.py \
      --layers 11 12 14 15 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def _load_progress(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text()).get("last_written", 0))
    except Exception:
        return 0


def _save_progress(path: Path, idx: int) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"last_written": int(idx)}))
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    default_repo = Path(os.environ.get("PHASE5_REPO", Path(__file__).resolve().parents[2]))
    ap.add_argument(
        "--cache-dir", type=Path,
        default=default_repo / "data/cached_activations/gemma-2-2b-it/fineweb",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--flush-every", type=int, default=500)
    ap.add_argument("--model-name", default="google/gemma-2-2b-it")
    args = ap.parse_args()

    cache_dir: Path = args.cache_dir
    tokens = np.load(cache_dir / "token_ids.npy", mmap_mode="r")
    n_seq, seq_len = tokens.shape
    d_model = 2304

    # Determine which layers still need work; allocate memmaps for each.
    mms: dict[int, np.ndarray] = {}
    progress_paths: dict[int, Path] = {}
    start_idxs: dict[int, int] = {}
    keep_layers: list[int] = []
    for layer in args.layers:
        out_path = cache_dir / f"resid_L{layer}.npy"
        progress_path = cache_dir / f".resid_L{layer}.progress.json"
        start_idx = _load_progress(progress_path)
        create_new = True
        if out_path.exists():
            try:
                existing = np.load(out_path, mmap_mode="r")
                if (
                    existing.shape == (n_seq, seq_len, d_model)
                    and existing.dtype == np.float16
                ):
                    create_new = False
                    if start_idx >= n_seq:
                        print(f"  L{layer}: already complete, skip")
                        continue
            except Exception:
                create_new = True
        if create_new:
            mm = np.lib.format.open_memmap(
                out_path, mode="w+", dtype=np.float16,
                shape=(n_seq, seq_len, d_model),
            )
            start_idx = 0
            _save_progress(progress_path, 0)
        else:
            mm = np.lib.format.open_memmap(
                out_path, mode="r+", dtype=np.float16,
                shape=(n_seq, seq_len, d_model),
            )
        mms[layer] = mm
        progress_paths[layer] = progress_path
        start_idxs[layer] = start_idx
        keep_layers.append(layer)

    if not keep_layers:
        print("All requested layers already complete.")
        return

    # Start from the minimum existing progress so all layers advance together.
    global_start = min(start_idxs.values())
    print(f"Layers to fill: {keep_layers}  starting at seq {global_start}/{n_seq}")

    from transformers import AutoModelForCausalLM
    print(f"Loading {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for layer in keep_layers:
        def make_hook(li: int):
            def hook_fn(module, inp, output):
                acts = output[0] if isinstance(output, tuple) else output
                captured[li] = acts.detach().to(torch.float16).cpu()
            return hook_fn
        hooks.append(
            model.model.layers[layer].register_forward_hook(make_hook(layer))
        )

    tokens_t = torch.from_numpy(np.asarray(tokens))
    t0 = time.time()
    last_flush = global_start
    end = global_start
    try:
        with torch.no_grad():
            for start in range(global_start, n_seq, args.batch_size):
                end = min(start + args.batch_size, n_seq)
                batch = tokens_t[start:end].to("cuda")
                captured.clear()
                model(batch)
                for li, mm in mms.items():
                    if start < start_idxs[li]:
                        continue
                    acts = captured[li]
                    if acts.shape[-1] != d_model:
                        acts = acts[..., :d_model]
                    mm[start:end] = acts.numpy()

                if (end - last_flush) >= args.flush_every or end == n_seq:
                    for li, mm in mms.items():
                        mm.flush()
                        _save_progress(progress_paths[li], end)
                    last_flush = end
                    elapsed = time.time() - t0
                    rate = (end - global_start) / max(1e-3, elapsed)
                    eta = (n_seq - end) / max(1e-3, rate)
                    print(
                        f"  [{end}/{n_seq}] {rate:.1f} seq/s, ETA {eta/60:.1f} min"
                    )
    finally:
        for h in hooks:
            h.remove()
        for li, mm in mms.items():
            mm.flush()
            _save_progress(progress_paths[li], end)

    # Update layer_specs.json
    specs_path = cache_dir / "layer_specs.json"
    specs = json.loads(specs_path.read_text())
    for layer in keep_layers:
        specs["layer_specs"][f"resid_L{layer}"] = {
            "layer": layer, "component": "resid",
            "d_act": 2304, "label": f"resid_L{layer}", "family": "gemma",
        }
    specs_path.write_text(json.dumps(specs, indent=2))
    print(f"Done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
