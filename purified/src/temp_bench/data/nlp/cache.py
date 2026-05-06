"""Build + iterate residual-stream activation caches.

Source-of-truth for real-LM activation caching across components.
The cache key is computed deterministically from the datasource spec
(see :func:`temp_bench.config.compute_act_cache_key`); two components
that reference the same datasource share the same on-disk bytes.

Ported in spirit from
`origin/han-phase7-unification @ 94119bc0:src/data/nlp/cache_activations.py`.
The wasteland version was wandb / sweep / CLI / wandb / prefetch-aware;
this version is a library function that takes a `DataSourceSpec` and
returns a path. Components don't shell out.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from temp_bench.config import (
    act_cache_dir,
    compute_act_cache_key,
    load_datasource,
)
from temp_bench.schemas import DataSourceSpec


# ── Hook helpers (cribbed from wasteland models.py — same Llama/Gemma path) ──


def _resid_hook_target(model, layer_idx: int):
    """Both Llama-family and Gemma-family expose `model.model.layers[i]`."""
    return model.model.layers[layer_idx]


# ── Public: build_activation_cache ──────────────────────────────────────────


def build_activation_cache(
    datasource_name: str,
    *,
    hf_token: str | None = None,
    batch_size: int = 64,
    device: str | torch.device = "cuda",
    force: bool = False,
) -> Path:
    """Build an activation cache for ``datasource_name``.

    Idempotent: if the cache directory already contains valid ``acts.npy``
    and ``meta.json`` matching the datasource spec, returns immediately.
    Pass ``force=True`` to rebuild from scratch.

    Args:
        datasource_name: key into ``configs/datasources.yaml``
            (must have ``category: real_lm``).
        hf_token: HuggingFace token (for gated models e.g. Llama).
            If ``None``, falls back to the ``HF_TOKEN`` env var or
            ``/workspace/.tokens/hf_token``.
        batch_size: forward-pass batch size. ``64`` works for Gemma-2-2b
            on H100 80GB at seq_len=128; halve for Llama-3.1-8B.
        device: target device.
        force: if True, rebuild even if a valid cache exists.

    Returns:
        Path to the cache directory (``results/act_cache/<key>/``).
    """
    spec = load_datasource(datasource_name)
    if spec.category != "real_lm":
        raise ValueError(
            f"build_activation_cache requires a real_lm datasource; "
            f"{datasource_name!r} has category={spec.category!r}."
        )

    key = compute_act_cache_key(spec)
    cache_dir = act_cache_dir(key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    acts_path = cache_dir / "acts.npy"
    meta_path = cache_dir / "meta.json"
    tokens_path = cache_dir / "token_ids.npy"

    if not force and _cache_is_valid(acts_path, meta_path, spec):
        return cache_dir

    print(f"[build_activation_cache] {datasource_name} → {cache_dir}")
    print(f"  subject={spec.subject_model} layer={spec.layer} "
          f"n_seqs={spec.n_seqs} seq_len={spec.seq_len}")

    hf_token = _resolve_hf_token(hf_token)
    model, tokenizer = _load_subject_model(spec, device, hf_token)
    d_in = model.config.hidden_size

    texts = _stream_dataset_texts(spec, max_n=spec.n_seqs)
    print(f"  fetched {len(texts)} texts")

    token_ids = _tokenize_fixed_length(tokenizer, texts, spec.seq_len)
    np.save(tokens_path, token_ids.numpy())

    # mmap the output array so we don't have to hold ~14 GB in RAM.
    acts_mmap = np.lib.format.open_memmap(
        acts_path, mode="w+", dtype=np.float16,
        shape=(token_ids.shape[0], spec.seq_len, d_in),
    )

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, inp, output):
        acts = output[0] if isinstance(output, tuple) else output
        captured["resid"] = acts.detach().to(torch.float16).cpu()

    handle = _resid_hook_target(model, spec.layer).register_forward_hook(hook_fn)

    try:
        device_t = torch.device(device)
        for start in range(0, token_ids.shape[0], batch_size):
            end = min(start + batch_size, token_ids.shape[0])
            input_ids = token_ids[start:end].to(device_t)
            captured.clear()
            with torch.no_grad():
                model(input_ids)
            acts_mmap[start:end] = captured["resid"].numpy()
            if start % (batch_size * 50) == 0:
                pct = 100.0 * end / token_ids.shape[0]
                print(f"  {end}/{token_ids.shape[0]}  ({pct:.1f}%)")
        acts_mmap.flush()
    finally:
        handle.remove()

    meta_path.write_text(json.dumps({
        "datasource_name": datasource_name,
        "act_cache_key": key,
        "spec": spec.model_dump(),
        "d_in": int(d_in),
        "shape": [int(token_ids.shape[0]), int(spec.seq_len), int(d_in)],
        "dtype": "float16",
    }, indent=2))

    print(f"  wrote {acts_path} ({acts_path.stat().st_size / 2**30:.2f} GB)")
    return cache_dir


# ── Public: batch_iter_from_act_cache ───────────────────────────────────────


def batch_iter_from_act_cache(
    act_cache_key: str,
    *,
    seed: int = 0,
) -> Callable[[int], torch.Tensor]:
    """Return a deterministic callable ``(batch_size) -> Tensor``.

    The iterator memory-maps ``results/act_cache/<key>/acts.npy`` and
    on each call samples ``batch_size`` sequences with replacement
    using a seeded NumPy generator. Returned tensor has shape
    ``(batch_size, seq_len, d_in)`` on CPU; the trainer moves it to
    device.

    For the canonical SAE trainer, the same iterator is used for the
    main training step AND for Bricken's dead-feature check batch —
    so both see i.i.d. samples from the same distribution.
    """
    cache_dir = act_cache_dir(act_cache_key)
    acts_path = cache_dir / "acts.npy"
    if not acts_path.exists():
        raise FileNotFoundError(
            f"Activation cache missing at {acts_path}. "
            f"Run build_activation_cache(...) first or download from HF."
        )

    acts = np.load(acts_path, mmap_mode="r")  # (N, seq_len, d_in) fp16
    n_seqs = acts.shape[0]
    rng = np.random.default_rng(seed)

    def _iter(batch_size: int) -> torch.Tensor:
        idx = rng.integers(0, n_seqs, size=batch_size)
        # Materialise the slice (mmap → contiguous fp16 numpy → torch)
        batch = np.ascontiguousarray(acts[idx])
        return torch.from_numpy(batch).to(torch.float32)

    return _iter


# ── Public: preloaded_batch_iter_from_act_cache ─────────────────────────────


# Module-global cache so multiple cells in the same process share one
# RAM copy per (act_cache_key) — keyed by the cache key, not the seed.
_PRELOADED_ACT_CACHES: dict[str, torch.Tensor] = {}


def preloaded_batch_iter_from_act_cache(
    act_cache_key: str,
    *,
    seed: int = 0,
) -> Callable[[int], torch.Tensor]:
    """Bit-identical drop-in for :func:`batch_iter_from_act_cache` that
    pre-materialises the activation cache into a CPU torch tensor.

    Why an opt-in helper rather than the default:
    The default ``batch_iter_from_act_cache`` mmaps the .npy and uses
    numpy fancy indexing per call. At ``batch=1024`` × ``seq_len=128``
    × ``d_in=2304``, each batch touches ~150K 4 KB pages from the
    file. Even when the file is fully in OS page cache, the kernel
    walks page tables on every access — RAM-bound at ~1.8 GB/sec
    instead of 100+ GB/sec. Profiling on Gemma-2-2b L13 showed the
    iterator was the trainer bottleneck (~330 ms / call).

    This helper does ``.clone()`` on first access to copy the entire
    file (~14 GB fp16 for the 24K Gemma cache; ~30 GB for larger
    caches) into anonymous RAM and then samples via torch fancy
    indexing. Empirical ~3.4× speedup on the data path; ~1.4× on the
    end-to-end trainer (model compute is the rest).

    **Determinism**: uses the same ``np.random.default_rng(seed)`` for
    indices and the same ``.to(torch.float32)`` contract as the
    default helper, so checkpoints written under either path are
    bit-identical for the same ``(act_cache_key, seed)`` pair.

    **RAM cost**: one ~`acts.npy.size` bytes copy per process per
    distinct ``act_cache_key`` (cached module-globally; multiple cells
    that share an act_cache_key share the copy). H100 / H200 pods (128+
    GB system RAM) are unaffected; A40 pods (~64 GB) should check
    headroom first if running concurrent processes — at 14 GB per
    cache, 4 concurrent processes saturate.

    See agent_paper's review (decisions.md § 12 cross-ref) for the
    fairness analysis: same train_key → same checkpoint bytes whether
    cells use the mmap or preloaded path. No methodology disclosure
    needed when mixing the two within a sweep.
    """
    cache_dir = act_cache_dir(act_cache_key)
    acts_path = cache_dir / "acts.npy"
    if not acts_path.exists():
        raise FileNotFoundError(
            f"Activation cache missing at {acts_path}. "
            f"Run build_activation_cache(...) first or download from HF."
        )

    if act_cache_key not in _PRELOADED_ACT_CACHES:
        # Force RAM materialisation. Without .clone(), torch.from_numpy on
        # an already-contiguous mmap'd array zero-copy wraps the mmap and
        # subsequent fancy indexing still page-faults.
        mmapped = np.load(acts_path, mmap_mode="r")
        _PRELOADED_ACT_CACHES[act_cache_key] = (
            torch.from_numpy(np.ascontiguousarray(mmapped)).clone()
        )
    acts = _PRELOADED_ACT_CACHES[act_cache_key]
    n_seqs = acts.shape[0]
    rng = np.random.default_rng(seed)

    def _iter(batch_size: int) -> torch.Tensor:
        idx = torch.from_numpy(rng.integers(0, n_seqs, size=batch_size))
        # torch fancy indexing on the cloned tensor (RAM-rate); fp32
        # cast to match the default helper's contract.
        return acts[idx].to(torch.float32)

    return _iter


# ── Internal helpers ────────────────────────────────────────────────────────


def _cache_is_valid(acts_path: Path, meta_path: Path, spec: DataSourceSpec) -> bool:
    if not (acts_path.exists() and meta_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    expected_n = spec.n_seqs
    if meta.get("shape", [0])[0] < expected_n:
        return False
    return True


def _resolve_hf_token(hf_token: str | None) -> str | None:
    if hf_token:
        return hf_token
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    token_file = Path("/workspace/.tokens/hf_token")
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def _load_subject_model(spec: DataSourceSpec, device, hf_token: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  loading {spec.subject_model}...")
    tokenizer = AutoTokenizer.from_pretrained(spec.subject_model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bf16 on H100, fp16 on A40 (MI300+) — let the caller pin via env if needed
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        spec.subject_model,
        token=hf_token,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def _stream_dataset_texts(spec: DataSourceSpec, *, max_n: int) -> list[str]:
    """Pull ``max_n`` text samples from the datasource's dataset.

    Currently supports only the ``fineweb`` dataset path (sample-10BT).
    Add new branches here as components introduce new datasource entries.
    """
    from datasets import load_dataset

    if spec.dataset == "fineweb":
        ds = load_dataset(
            "HuggingFaceFW/fineweb", "sample-10BT",
            split="train", streaming=True,
        )
    else:
        raise NotImplementedError(
            f"_stream_dataset_texts: dataset {spec.dataset!r} not yet wired. "
            f"Add a branch here when introducing the datasource."
        )

    out: list[str] = []
    for sample in ds:
        txt = sample.get("text") or sample.get("content") or ""
        if not txt or len(txt) < 20:
            continue
        out.append(str(txt))
        if len(out) >= max_n:
            break
    return out


def _tokenize_fixed_length(tokenizer, texts: list[str], seq_len: int) -> torch.Tensor:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        add_special_tokens=True,
    )
    return enc["input_ids"]
