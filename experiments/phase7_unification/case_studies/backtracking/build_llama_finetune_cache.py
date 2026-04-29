#!/usr/bin/env python3
"""Build a Llama-3.1-8B activation cache on FineWeb at a configurable hook
point — foundation for training TXC / SAE variants on Llama L10.

This is the Llama-side analogue of Phase 7's `build_act_cache_phase7.py`,
which is pinned to Gemma-2-2b L12 residual stream. Differences:

  * Subject model: Llama-3.1-8B (d_in=4096) — registry key llama-3.1-8b.
  * Hook variants: --hook resid (post-layer residual; matches Llama-Scope),
                   --hook ln1   (output of input_layernorm before self_attn),
                   --hook attn  (output of self_attn before residual add).
  * Layer: --layer 10 (default).

Output format (mirrors Phase 7's per-layer cache so existing trainers can
read it with one path swap):

    data/llama_3_1_8b/<hook>/L<layer>/
      ├── token_ids.npy        (N, seq_len) int32
      ├── activations.fp16.npy (N, seq_len, 4096) fp16, memmappable
      ├── meta.json            {model, layer, hook, d_in, dtype, n_tokens}
      └── progress.json        {last_written: int} — resumable

Default budget is 100k token-windows × 128 = ~13M tokens, ~100 GB on disk.
For smoke runs use --num-windows 8000 (~1M tokens, ~8 GB, ~20 min on H100).

Run from repo root (host with /workspace mounted):

    TMPDIR=/workspace/tmp HF_HOME=/workspace/hf_cache \
    .venv/bin/python -m experiments.phase7_unification.case_studies.backtracking.build_llama_finetune_cache \
        --hook ln1 --layer 10 --num-windows 8000 --batch-size 32

Re-running with the same args resumes from progress.json. Use --force to
rebuild from scratch.
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
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TMPDIR", "/workspace/tmp")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.nlp.cache_activations import load_model_and_tokenizer  # noqa: E402
from src.data.nlp.models import attn_hook_target, resid_hook_target  # noqa: E402

CTX = 128
LLAMA_REGISTRY_KEY = "llama-3.1-8b"


def hook_target(model, layer_idx: int, hook: str, family: str = "llama"):
    """Return the submodule whose forward output we want to capture."""
    if hook == "resid":
        return resid_hook_target(model, layer_idx, family)
    if hook == "ln1":
        # Output of input_layernorm = the LN-normalised input that flows into
        # self_attn. Per-layer module name in HF Llama implementations is
        # `input_layernorm`.
        return model.model.layers[layer_idx].input_layernorm
    if hook == "attn":
        # Output of self_attn (the attention sublayer, before residual add).
        return attn_hook_target(model, layer_idx, family)
    raise ValueError(f"unknown hook {hook!r}; use resid|ln1|attn")


def stream_fineweb_text(num_windows: int) -> list[str]:
    """Pull num_windows worth of FineWeb examples; defer tokenization to
    the caller. Reuses /workspace/hf_cache so re-runs are cheap.
    """
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        "sample-10BT",
        split="train",
        streaming=True,
    )
    out: list[str] = []
    needed = num_windows
    for i, rec in enumerate(ds):
        text = rec.get("text", "")
        if not text:
            continue
        out.append(text)
        # Heuristic: each FineWeb record is ~512-2048 tokens, far more than CTX.
        # Pull more records than strictly needed; we'll truncate per-window below.
        if len(out) >= needed:
            break
    return out


def tokenize_to_fixed_windows(tokenizer, texts: list[str], num_windows: int, ctx: int) -> np.ndarray:
    """Concatenate, tokenize, then split into (num_windows, ctx) int32."""
    tokens: list[int] = []
    eos = tokenizer.eos_token_id
    needed = num_windows * ctx
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        tokens.extend(ids)
        tokens.append(eos)
        if len(tokens) >= needed:
            break
    if len(tokens) < needed:
        raise RuntimeError(f"only got {len(tokens)} tokens, need {needed}; pass more --num-windows or fewer --ctx")
    arr = np.asarray(tokens[:needed], dtype=np.int32).reshape(num_windows, ctx)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=LLAMA_REGISTRY_KEY, help="registry key in src.data.nlp.models")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--hook", choices=("resid", "ln1", "attn"), default="ln1")
    parser.add_argument("--num-windows", type=int, default=100_000)
    parser.add_argument("--ctx", type=int, default=CTX)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-root", default=None, help="default: <repo>/data/llama_3_1_8b")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root) if args.out_root else _REPO / "data" / "llama_3_1_8b"
    cache_dir = out_root / args.hook / f"L{args.layer}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tok_path = cache_dir / "token_ids.npy"
    act_path = cache_dir / "activations.fp16.npy"
    meta_path = cache_dir / "meta.json"
    progress_path = cache_dir / "progress.json"

    if args.force:
        for p in (tok_path, act_path, meta_path, progress_path):
            if p.exists():
                p.unlink()

    if act_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("n_windows", 0) >= args.num_windows:
            print(f"[cache] {act_path} already has {meta['n_windows']} windows ≥ {args.num_windows}; nothing to do")
            return

    print(f"[cache] building Llama L{args.layer} {args.hook} cache: {args.num_windows} × {args.ctx} → {cache_dir}")

    # ── Tokenize -------------------------------------------------------------
    if tok_path.exists():
        token_ids = np.load(tok_path)
        if token_ids.shape != (args.num_windows, args.ctx):
            print(f"[cache] existing token_ids shape {token_ids.shape} ≠ requested ({args.num_windows}, {args.ctx}); rebuilding")
            tok_path.unlink()
            token_ids = None
        else:
            print(f"[cache] reusing token_ids.npy {token_ids.shape}")
    else:
        token_ids = None

    if token_ids is None:
        print("[cache] streaming FineWeb …")
        from transformers import AutoTokenizer
        from src.data.nlp.models import get_model_config
        cfg = get_model_config(args.model)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Need ~5x num_windows of records to be safe (each FineWeb record ~512-2048 tok).
        texts = stream_fineweb_text(num_windows=max(args.num_windows // 4, 256))
        print(f"[cache] tokenising {len(texts)} fineweb records → {args.num_windows} × {args.ctx} windows")
        token_ids = tokenize_to_fixed_windows(tokenizer, texts, args.num_windows, args.ctx)
        np.save(tok_path, token_ids)
        print(f"[cache] saved {tok_path} {token_ids.shape}")
    else:
        print("[cache] tokeniser load skipped (token_ids cached)")

    # ── Load model + register hook ------------------------------------------
    print(f"[cache] loading {args.model}")
    model, tokenizer, cfg = load_model_and_tokenizer(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    layer_mod = hook_target(model, args.layer, args.hook, cfg.architecture_family)
    captured: dict[str, torch.Tensor] = {}

    def fn(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach()

    handle = layer_mod.register_forward_hook(fn)

    # ── Allocate output array ----------------------------------------------
    n, ctx = token_ids.shape
    d_in = int(cfg.d_model)
    if act_path.exists():
        activations = np.load(act_path, mmap_mode="r+")
        if activations.shape != (n, ctx, d_in):
            print(f"[cache] existing activations {activations.shape} ≠ ({n}, {ctx}, {d_in}); rebuilding")
            act_path.unlink()
            activations = None
    else:
        activations = None
    if activations is None:
        # Pre-allocate as a memmap-friendly fp16 array on /workspace
        activations = np.lib.format.open_memmap(
            str(act_path), mode="w+", dtype=np.float16, shape=(n, ctx, d_in)
        )

    # ── Resumable index ----------------------------------------------------
    last_written = 0
    if progress_path.exists():
        try:
            last_written = int(json.loads(progress_path.read_text()).get("last_written", 0))
        except Exception:
            last_written = 0
    if last_written > 0:
        print(f"[cache] resuming from window {last_written}")

    # ── Forward loop -------------------------------------------------------
    bs = int(args.batch_size)
    t0 = time.time()
    for s in range(last_written, n, bs):
        e = min(n, s + bs)
        ids = torch.from_numpy(token_ids[s:e].astype(np.int64)).to(device)
        with torch.no_grad():
            _ = model(ids, use_cache=False)
        h = captured["h"]
        # Some hooks (LayerNorm) may return (b, ctx, d_in) directly; self_attn returns
        # tuple — output[0] is what we want, already extracted by `fn`.
        if h.dim() != 3 or h.shape[-1] != d_in:
            raise RuntimeError(f"unexpected hook output shape {tuple(h.shape)}; expected (B, {ctx}, {d_in})")
        activations[s:e] = h.to(torch.float16).cpu().numpy()
        last_written = e
        if (s // bs) % 25 == 0 or e == n:
            tps = (time.time() - t0) / max(e - last_written + 1, 1)
            print(f"  [{e:>6}/{n}] {(time.time() - t0):.0f}s elapsed")
            progress_path.write_text(json.dumps({"last_written": int(last_written)}))
            activations.flush() if hasattr(activations, "flush") else None

    handle.remove()
    progress_path.write_text(json.dumps({"last_written": int(n)}))
    meta = {
        "model": args.model,
        "layer": args.layer,
        "hook": args.hook,
        "d_in": int(d_in),
        "ctx": int(ctx),
        "n_windows": int(n),
        "dtype": "float16",
        "wall_seconds": round(time.time() - t0, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[cache] done: {act_path} ({activations.shape}, {activations.nbytes/1e9:.2f} GB)")
    print(f"[cache] meta {meta}")


if __name__ == "__main__":
    main()
