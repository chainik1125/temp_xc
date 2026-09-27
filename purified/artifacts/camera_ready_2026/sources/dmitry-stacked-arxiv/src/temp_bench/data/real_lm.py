"""Real-LM activation cache: build, load, and refill source.

Two stages:

1. **Build** (``build_activation_cache``): forward the subject model
   over its dataset, extract residual-stream activations at the
   declared layer + hookpoint, write ``acts.npy`` of shape
   ``(N, seq_len, d_in)`` plus ``meta.json``. Idempotent: skips if a
   valid cache already exists.

2. **Refill** (``build_refill``): returns a callable
   ``(n_seqs) -> (n_seqs, seq_len, d_in)`` that the buffer calls to top
   up. Uses memory-mapped numpy under the hood.

The buffer pattern (token shuffle, see ``ActivationBuffer``) replaces v1's
whole-sequence batching. v1 saw 131K correlated tokens per batch; v2
samples i.i.d. tokens from a 2M-token rolling buffer — literature standard.

Subject models supported: Gemma-2-2B (IT + BASE), Llama-3.1-8B,
DeepSeek-R1-Distill-Llama-8B, Qwen-2.5-7B/14B. Adding a new model: add
a row to ``_resid_dim_for_model`` in ``core/trainer.py`` + ensure the
``transformers``-based forward path here handles it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from temp_bench.core.config import compute_data_key, data_cache_dir
from temp_bench.core.schemas import DataSourceSpec


def _resid_hook_target(model, layer_idx: int):
    """Most HF causal-LM families expose ``model.model.layers[i]``."""
    return model.model.layers[layer_idx]


def _resolve_hf_token() -> str | None:
    """Get an HF token from env or local file (for gated models)."""
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    p = Path("/workspace/.tokens/hf_token")
    if p.exists():
        return p.read_text().strip()
    p = Path.home() / ".tokens" / "hf_token"
    if p.exists():
        return p.read_text().strip()
    return None


# ── Build activation cache (heavyweight; only on demand) ──────────────


def build_activation_cache(
    spec: DataSourceSpec,
    *,
    batch_size: int = 32,
    device: str | None = None,
    force: bool = False,
) -> Path:
    """Build the activation cache for a real-LM datasource.

    Heavy operation: loads the subject model, forwards over the dataset,
    captures the resid-stream layer. ``~3 H100-hours`` for Gemma-2-2B on
    24K FineWeb sequences.

    Idempotent: if ``acts.npy + meta.json`` exists and matches the spec,
    return immediately.

    NOTE: this function intentionally has heavy deps (transformers,
    datasets). Only imported when called — keeps the framework
    lightweight for synthetic-only sessions.
    """
    if spec.category != "real_lm":
        raise ValueError(
            f"build_activation_cache: spec {spec.name!r} is not real_lm."
        )

    data_key = compute_data_key(spec)
    cache_dir = data_cache_dir(data_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    acts_path = cache_dir / "acts.npy"
    meta_path = cache_dir / "meta.json"

    if not force and acts_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("data_key") == data_key:
            print(f"[real_lm] cache hit: {cache_dir}")
            return cache_dir

    print(f"[real_lm] building cache: {spec.name} → {cache_dir}")

    # Lazy heavyweight imports.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    token = _resolve_hf_token()

    tokenizer = AutoTokenizer.from_pretrained(spec.subject_model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        spec.subject_model,
        torch_dtype=torch.bfloat16,
        token=token,
    ).to(device).eval()

    # Capture activations via a forward hook.
    captured: list[torch.Tensor] = []
    def hook(_module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().to(torch.float16).cpu())
    target = _resid_hook_target(model, int(spec.layer))
    handle = target.register_forward_hook(hook)

    # Stream the dataset, tokenise + forward in chunks.
    ds = load_dataset(spec.dataset, split="train", streaming=True)
    text_field = "text"        # FineWeb / The Pile / similar use "text"

    acts_chunks: list[np.ndarray] = []
    n_collected = 0
    target_n = int(spec.n_seqs)
    seq_len = int(spec.seq_len)

    buffer_texts: list[str] = []
    for row in ds:
        if n_collected >= target_n:
            break
        if text_field not in row:
            continue
        buffer_texts.append(row[text_field])
        if len(buffer_texts) >= batch_size:
            inputs = tokenizer(
                buffer_texts,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            ).to(device)
            captured.clear()
            with torch.no_grad():
                model(**inputs, use_cache=False)
            h = captured[-1]                                 # (B, seq_len, d_in)
            acts_chunks.append(h.numpy().astype(np.float16))
            n_collected += h.shape[0]
            buffer_texts.clear()
            print(f"  collected {n_collected}/{target_n}")

    handle.remove()
    acts = np.concatenate(acts_chunks, axis=0)[: target_n]
    np.save(acts_path, acts)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "data_key": data_key,
                "subject_model": spec.subject_model,
                "layer": spec.layer,
                "hookpoint": spec.hookpoint,
                "dataset": spec.dataset,
                "n_seqs": int(acts.shape[0]),
                "seq_len": int(acts.shape[1]),
                "d_in": int(acts.shape[2]),
            },
            f,
            indent=2,
        )
    print(f"[real_lm] wrote {acts.shape} → {acts_path}")
    return cache_dir


# ── Refill source (used by trainer) ────────────────────────────────────


def build_refill(spec: DataSourceSpec, *, seed: int) -> Callable[[int], torch.Tensor]:
    """Return ``(n_seqs) -> (n_seqs, seq_len, d_in)`` callable.

    Assumes the activation cache already exists on disk
    (``build_activation_cache(spec)`` has been called). Raises with a
    clear error otherwise.

    Uses ``np.load(mmap_mode='r')`` to avoid loading the full cache
    into RAM at once; the buffer subsequently materialises a working
    subset into anonymous RAM.
    """
    data_key = compute_data_key(spec)
    cache_dir = data_cache_dir(data_key)
    acts_path = cache_dir / "acts.npy"
    if not acts_path.exists():
        raise FileNotFoundError(
            f"Activation cache missing for {spec.name!r} at {acts_path}. "
            f"Run: build_activation_cache(load_datasource({spec.name!r}))."
        )

    acts = np.load(acts_path, mmap_mode="r")
    n_total = acts.shape[0]
    rng = np.random.default_rng(seed)

    def refill(n: int) -> torch.Tensor:
        idx = rng.integers(0, n_total, size=n)
        batch = np.ascontiguousarray(acts[idx])
        return torch.from_numpy(batch.astype(np.float32))
    return refill
