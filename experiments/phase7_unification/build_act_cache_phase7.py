"""Build the Gemma-2-2b BASE activation cache for Phase 7.

Forked from Phase 5's `build_multilayer_cache.py`. Differences:
- Subject model: gemma-2-2b base (NOT -it).
- Layers: L10-L14 (5-layer MLC stack centred on L12 anchor); was L11-L15.
- Output dir: `data/cached_activations/gemma-2-2b/fineweb/`.
- Reuses `token_ids.npy` from any existing cache dir if present
  (gemma-2-2b shares the tokenizer with gemma-2-2b-it, so token IDs
  are identical and don't need rebuilding).

Single-layer-per-invocation + resumable design (MooseFS-friendly).
Run once per layer; ~70 GB total (5 layers × 24k seq × 128 tok × 2304 d × 2 B).

    .venv/bin/python -m experiments.phase7_unification.build_act_cache_phase7 --layer 12
    .venv/bin/python -m experiments.phase7_unification.build_act_cache_phase7 --layer 10
    ... etc for L11, L13, L14.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import (
    CACHE_DIR, MLC_LAYERS, ANCHOR_LAYER, SUBJECT_MODEL, banner,
)


CTX = 128
DTYPE = torch.bfloat16


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


def _ensure_token_ids(cache_dir: Path) -> Path:
    """Reuse token_ids.npy from gemma-2-2b-it cache if present (same
    tokenizer ⇒ same token IDs). Otherwise instruct the user to build
    it via FineWeb-tokenize-only path (separate script — out of scope
    here; manual step if -it cache is missing).
    """
    tok_path = cache_dir / "token_ids.npy"
    if tok_path.exists():
        return tok_path
    legacy = Path(str(cache_dir).replace("gemma-2-2b/", "gemma-2-2b-it/")) / "token_ids.npy"
    if legacy.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[token_ids] reusing legacy {legacy} -> {tok_path}")
        shutil.copy(legacy, tok_path)
        return tok_path
    raise SystemExit(
        f"Missing token_ids.npy at {tok_path}. Build it via Phase 5's "
        f"FineWeb tokenization step first, or copy from a legacy cache "
        f"(gemma-2-2b shares the tokenizer with gemma-2-2b-it)."
    )


def cache_single_layer(
    layer: int,
    cache_dir: Path = CACHE_DIR,
    model_name: str = SUBJECT_MODEL,
    batch_size: int = 16,
    flush_every: int = 1_000,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"resid_L{layer}.npy"
    progress_path = cache_dir / f".resid_L{layer}.progress.json"
    tok_path = _ensure_token_ids(cache_dir)

    tokens = np.load(tok_path, mmap_mode="r")
    n_seq, seq_len = tokens.shape
    d_model = 2304
    if seq_len != CTX:
        raise SystemExit(f"token_ids has seq_len={seq_len}, expected {CTX}")

    start_idx = _load_progress(progress_path)
    create_new = True
    if out_path.exists():
        try:
            existing = np.load(out_path, mmap_mode="r")
            if existing.shape == (n_seq, seq_len, d_model) and existing.dtype == np.float16:
                create_new = False
                if start_idx >= n_seq:
                    print(f"  L{layer}: already complete, skip")
                    return
        except Exception:
            create_new = True

    print(f"L{layer}: n_seq={n_seq} seq_len={seq_len} start_idx={start_idx} create_new={create_new}")

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
        model_name, torch_dtype=DTYPE, device_map="cuda",
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
                    print(f"  [{end}/{n_seq}] {rate:.1f} seq/s, ETA {eta/60:.1f} min")
    finally:
        handle.remove()
        mm.flush()
        _save_progress(progress_path, end)

    print(f"  L{layer} done in {(time.time() - t0)/60:.1f} min")


def write_layer_specs(cache_dir: Path) -> None:
    specs_path = cache_dir / "layer_specs.json"
    if specs_path.exists():
        specs = json.loads(specs_path.read_text())
    else:
        specs = {
            "model": "gemma-2-2b",
            "hf_path": SUBJECT_MODEL,
            "d_model": 2304,
            "layer_specs": {},
            "mode": "forward",
            "anchor_layer_0idx": ANCHOR_LAYER,
            "mlc_layers_0idx": list(MLC_LAYERS),
        }
    for layer in MLC_LAYERS:
        if (cache_dir / f"resid_L{layer}.npy").exists():
            key = f"resid_L{layer}"
            specs["layer_specs"][key] = {
                "layer": layer, "component": "resid",
                "d_act": 2304, "label": key, "family": "gemma",
            }
    specs_path.write_text(json.dumps(specs, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True,
                    choices=list(MLC_LAYERS),
                    help="0-indexed layer in {10..14}")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--flush-every", type=int, default=1_000)
    args = ap.parse_args()
    banner(__file__)
    cache_single_layer(
        layer=args.layer, cache_dir=args.cache_dir,
        batch_size=args.batch_size, flush_every=args.flush_every,
    )
    write_layer_specs(args.cache_dir)


if __name__ == "__main__":
    main()
