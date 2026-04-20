"""Extend the gemma-2-2b-it FineWeb activation cache with MLC layers.

Single-layer-per-invocation + resumable design to survive MooseFS
flakiness. See docstring in phase5 brief for full rationale.
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
        data = json.loads(path.read_text())
        return int(data.get("last_written", 0))
    except Exception:
        return 0


def _save_progress(path: Path, idx: int) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"last_written": int(idx)}))
    tmp.replace(path)


def cache_single_layer(
    layer: int, cache_dir: Path,
    model_name: str = "google/gemma-2-2b-it",
    batch_size: int = 16, flush_every: int = 1_000,
) -> None:
    out_path = cache_dir / f"resid_L{layer}.npy"
    progress_path = cache_dir / f".resid_L{layer}.progress.json"

    tokens = np.load(cache_dir / "token_ids.npy", mmap_mode="r")
    n_seq, seq_len = tokens.shape
    d_model = 2304

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
                    return
        except Exception:
            create_new = True

    print(f"L{layer}: n_seq={n_seq} start_idx={start_idx} create_new={create_new}")

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

    print(f"  memmap ready, resuming from seq {start_idx}")

    from transformers import AutoModelForCausalLM
    print(f"  loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    captured: dict[int, torch.Tensor] = {}

    def hook_fn(module, inp, output):
        acts = output[0] if isinstance(output, tuple) else output
        captured[layer] = acts.detach().cpu()

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    tokens_t = torch.from_numpy(np.asarray(tokens))

    t0 = time.time()
    last_flush = start_idx
    end = start_idx
    try:
        with torch.no_grad():
            for start in range(start_idx, n_seq, batch_size):
                end = min(start + batch_size, n_seq)
                batch = tokens_t[start:end].to("cuda")
                captured.clear()
                model(batch)
                acts = captured[layer]
                if acts.shape[-1] != d_model:
                    acts = acts[..., :d_model]
                mm[start:end] = acts.to(torch.float16).numpy()

                if (end - last_flush) >= flush_every or end == n_seq:
                    mm.flush()
                    _save_progress(progress_path, end)
                    last_flush = end
                    elapsed = time.time() - t0
                    rate = (end - start_idx) / max(1e-3, elapsed)
                    eta = (n_seq - end) / max(1e-3, rate)
                    print(
                        f"  [{end}/{n_seq}] {rate:.1f} seq/s, ETA {eta/60:.1f} min"
                    )
    finally:
        handle.remove()
        mm.flush()
        _save_progress(progress_path, end)

    print(f"  L{layer} done in {(time.time() - t0)/60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    default_repo = Path(os.environ.get("PHASE5_REPO", Path(__file__).resolve().parents[2]))
    ap.add_argument(
        "--cache-dir", type=Path,
        default=default_repo / "data/cached_activations/gemma-2-2b-it/fineweb",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--flush-every", type=int, default=1_000)
    args = ap.parse_args()

    cache_single_layer(
        layer=args.layer, cache_dir=args.cache_dir,
        batch_size=args.batch_size, flush_every=args.flush_every,
    )

    specs_path = args.cache_dir / "layer_specs.json"
    if specs_path.exists():
        specs = json.loads(specs_path.read_text())
    else:
        specs = {
            "model": "gemma-2-2b-it",
            "hf_path": "google/gemma-2-2b-it",
            "d_model": 2304, "layer_specs": {}, "mode": "forward",
        }
    key = f"resid_L{args.layer}"
    specs["layer_specs"][key] = {
        "layer": args.layer, "component": "resid",
        "d_act": 2304, "label": key, "family": "gemma",
    }
    specs_path.write_text(json.dumps(specs, indent=2))


if __name__ == "__main__":
    main()
