"""Python-code activation pipeline for the code_benchmark experiment.

Responsibilities:
    - Load a Python-code corpus (primary: bigcode/python-stack-v1-functions-filtered)
      with a loadable fallback that does not require gated access.
    - AST-parse filter; tokenize with the Gemma-2-2B tokenizer; chunk into
      fixed-length windows with stride.
    - Extract residual-stream activations at configurable layers using
      TransformerLens' HookedTransformer.
    - Cache tokens + activations + raw source on disk for reuse by the
      training / eval passes.

Cache layout::

    cache/
        tokens.pt                    # (N, chunk_tokens) int64
        sources.jsonl                # list of {function_idx, chunk_idx, source, offset_map}
        acts_layer_{L}.pt            # (N, chunk_tokens, d_model) bfloat16 per layer
        manifest.json                # config snapshot + shape metadata
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CodeDatasetConfig:
    name: str = "bigcode/python-stack-v1-functions-filtered"
    fallback_name: str = "codeparrot/codeparrot-clean-valid"
    split: str = "train"
    streaming: bool = True
    max_functions: int = 20_000
    min_tokens: int = 32
    max_tokens: int = 512
    chunk_tokens: int = 128
    stride_tokens: int = 64
    filter_ast_parse: bool = True
    train_eval_split: float = 0.9
    seed: int = 42
    text_field_candidates: tuple[str, ...] = ("content", "code", "text", "function")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CodeDatasetConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SubjectModelConfig:
    name: str = "google/gemma-2-2b-it"
    anchor_layer: int = 12
    mlc_layers: list[int] = field(default_factory=lambda: [10, 11, 12, 13, 14])
    d_model: int = 2304

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubjectModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def required_layers(self) -> list[int]:
        return sorted(set([self.anchor_layer, *self.mlc_layers]))


# ---------------------------------------------------------------------------
# Dataset loading with graceful fallback
# ---------------------------------------------------------------------------


def iter_python_functions(cfg: CodeDatasetConfig) -> Iterator[str]:
    """Yield raw Python source strings.

    Prefers the primary dataset; falls back to the named fallback on any load
    failure (gated access, missing network, dataset renamed). Filters by
    ``ast.parse`` success and token-budget bounds downstream — this generator
    only yields raw strings, in corpus order, up to ``max_functions``.
    """
    from datasets import load_dataset

    def _yield_from(name: str) -> Iterator[str]:
        ds = load_dataset(name, split=cfg.split, streaming=cfg.streaming)
        for row in ds:
            for field_name in cfg.text_field_candidates:
                if field_name in row and isinstance(row[field_name], str):
                    yield row[field_name]
                    break

    try:
        source_iter = _yield_from(cfg.name)
        first = next(source_iter)
        yield first
        yield from source_iter
        return
    except Exception as e:
        print(f"[python_code] primary dataset {cfg.name!r} failed ({e!r}); "
              f"falling back to {cfg.fallback_name!r}")
    yield from _yield_from(cfg.fallback_name)


def ast_parseable(source: str) -> bool:
    import ast
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Tokenisation + chunking
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    function_idx: int
    chunk_idx: int
    token_ids: list[int]
    char_offsets: list[tuple[int, int]]   # (start_char, end_char) per token
    source: str                            # the full function source


def chunk_functions(
    sources: Iterable[str],
    tokenizer: Any,
    cfg: CodeDatasetConfig,
) -> list[Chunk]:
    """Tokenize each source and slice into overlapping chunks.

    Uses the HuggingFace fast-tokenizer's ``offset_mapping`` so that each token
    carries the ``(start_char, end_char)`` span of its source — required by the
    program-state labeler.
    """
    chunks: list[Chunk] = []
    for function_idx, source in enumerate(sources):
        if cfg.filter_ast_parse and not ast_parseable(source):
            continue
        enc = tokenizer(
            source,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        token_ids: list[int] = enc["input_ids"]
        offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]
        n_tok = len(token_ids)
        if n_tok < cfg.min_tokens:
            continue
        token_ids = token_ids[: cfg.max_tokens]
        offsets = offsets[: cfg.max_tokens]
        n_tok = len(token_ids)

        start = 0
        chunk_idx = 0
        while start < n_tok:
            end = min(start + cfg.chunk_tokens, n_tok)
            piece_ids = token_ids[start:end]
            piece_offs = offsets[start:end]
            if len(piece_ids) < cfg.min_tokens:
                break
            # left-pad to chunk_tokens if this is the only / last piece
            if len(piece_ids) < cfg.chunk_tokens:
                pad_n = cfg.chunk_tokens - len(piece_ids)
                piece_ids = piece_ids + [tokenizer.pad_token_id or 0] * pad_n
                piece_offs = piece_offs + [(-1, -1)] * pad_n
            chunks.append(Chunk(
                function_idx=function_idx,
                chunk_idx=chunk_idx,
                token_ids=piece_ids,
                char_offsets=piece_offs,
                source=source,
            ))
            chunk_idx += 1
            if end >= n_tok:
                break
            start += cfg.stride_tokens
    return chunks


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


def load_subject_model(cfg: SubjectModelConfig, device: str, dtype: torch.dtype):
    """Load the Gemma-2B-it subject model via TransformerLens.

    Returns a ``HookedTransformer`` in eval mode, on ``device``, activations
    stored at ``dtype``. ``transformer_lens`` is an optional dependency gated
    behind the ``separation-scaling`` extra — import is local to avoid making
    this module a hard dependency for callers that only want labels.
    """
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        cfg.name,
        device=device,
        dtype=dtype,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_activations(
    model: Any,
    token_tensor: torch.Tensor,
    layers: list[int],
    batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """Run the subject model; return ``{layer: (N, T, d_model)}`` CPU tensors.

    ``token_tensor`` is ``(N, T)`` int64. We hook ``blocks.{L}.hook_resid_post``
    for each requested layer, concatenate chunks on CPU to keep GPU memory flat.
    """
    hook_names = [f"blocks.{L}.hook_resid_post" for L in layers]
    per_layer: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    device = next(model.parameters()).device
    hook_set = set(hook_names)
    for i in range(0, token_tensor.shape[0], batch_size):
        batch = token_tensor[i : i + batch_size].to(device)
        _, cache = model.run_with_cache(
            batch,
            return_type="logits",
            names_filter=lambda name: name in hook_set,
            return_cache_object=False,
        )
        for L, h in zip(layers, hook_names):
            per_layer[L].append(cache[h].detach().to("cpu"))
    return {L: torch.cat(parts, dim=0) for L, parts in per_layer.items()}


# ---------------------------------------------------------------------------
# On-disk cache
# ---------------------------------------------------------------------------


def cache_paths(root: Path, layers: list[int]) -> dict[str, Path]:
    return {
        "tokens": root / "tokens.pt",
        "sources": root / "sources.jsonl",
        "manifest": root / "manifest.json",
        **{f"acts_{L}": root / f"acts_layer_{L}.pt" for L in layers},
    }


def cache_exists(root: Path, layers: list[int]) -> bool:
    paths = cache_paths(root, layers)
    return all(p.exists() for p in paths.values())


def save_cache(
    root: Path,
    tokens: torch.Tensor,
    chunks: list[Chunk],
    activations: dict[int, torch.Tensor],
    manifest: dict[str, Any],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(root, list(activations.keys()))
    torch.save(tokens, paths["tokens"])
    with paths["sources"].open("w") as f:
        for c in chunks:
            f.write(json.dumps({
                "function_idx": c.function_idx,
                "chunk_idx": c.chunk_idx,
                "source": c.source,
                "char_offsets": c.char_offsets,
            }) + "\n")
    for L, acts in activations.items():
        torch.save(acts, paths[f"acts_{L}"])
    with paths["manifest"].open("w") as f:
        json.dump(manifest, f, indent=2)


def load_cache(root: Path, layers: list[int]) -> tuple[torch.Tensor, list[dict], dict[int, torch.Tensor], dict]:
    paths = cache_paths(root, layers)
    tokens = torch.load(paths["tokens"])
    sources: list[dict] = []
    with paths["sources"].open() as f:
        for line in f:
            sources.append(json.loads(line))
    acts = {L: torch.load(paths[f"acts_{L}"]) for L in layers}
    manifest = json.loads(paths["manifest"].read_text())
    return tokens, sources, acts, manifest


# ---------------------------------------------------------------------------
# Top-level build
# ---------------------------------------------------------------------------


def build_cache(
    cache_root: Path,
    data_cfg: CodeDatasetConfig,
    subject_cfg: SubjectModelConfig,
    device: str,
    dtype_str: str = "bfloat16",
    extract_batch_size: int = 8,
) -> None:
    """End-to-end: load corpus → tokenize/chunk → extract activations → cache.

    Idempotent: if cache is already present, returns immediately.
    """
    layers = subject_cfg.required_layers()
    if cache_exists(cache_root, layers):
        print(f"[python_code] cache present at {cache_root}; skipping rebuild")
        return

    print(f"[python_code] building cache at {cache_root}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(subject_cfg.name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sources_iter = iter_python_functions(data_cfg)
    sources: list[str] = []
    for i, s in enumerate(sources_iter):
        if i >= data_cfg.max_functions:
            break
        sources.append(s)

    chunks = chunk_functions(sources, tokenizer, data_cfg)
    if not chunks:
        raise RuntimeError("No chunks produced — dataset empty or all filtered out.")
    token_tensor = torch.tensor([c.token_ids for c in chunks], dtype=torch.long)
    print(f"[python_code] {len(chunks)} chunks, tokens shape {tuple(token_tensor.shape)}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]
    model = load_subject_model(subject_cfg, device=device, dtype=dtype)
    acts = extract_activations(model, token_tensor, layers, batch_size=extract_batch_size)
    # Store activations in the same dtype they were produced in
    acts = {L: a.to(dtype) for L, a in acts.items()}

    manifest = {
        "n_chunks": len(chunks),
        "chunk_tokens": data_cfg.chunk_tokens,
        "layers": layers,
        "subject_model": subject_cfg.name,
        "dtype": dtype_str,
        "d_model": subject_cfg.d_model,
        "train_eval_split": data_cfg.train_eval_split,
        "seed": data_cfg.seed,
    }
    save_cache(cache_root, token_tensor, chunks, acts, manifest)
    print(f"[python_code] cache written to {cache_root}")


# ---------------------------------------------------------------------------
# Train/eval split (deterministic on chunk_idx)
# ---------------------------------------------------------------------------


def train_eval_split_indices(n: int, split: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    k = int(split * n)
    return perm[:k], perm[k:]
