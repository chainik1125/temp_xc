"""Pinned reproduction of the tensor-network-futurelens GPT-2 activation cache.

The implementation mirrors public commit ``scripts/cache_residuals.py`` and
``src/tn_futurelens/data/token_datasets.py`` while requiring explicit Hugging
Face revisions.  It exists here so a reviewer-stage result never depends on a
moving model or dataset branch.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch


_ARTICLE_HEADER = re.compile(r"^ = [^=].* = $")


def iter_wikitext_articles(dataset_repo: str, revision: str, split: str = "train"):
    """Yield WikiText articles using the original top-level-header grouping."""
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_repo,
        "wikitext-103-raw-v1",
        split=split,
        streaming=True,
        revision=revision,
    )
    buffer: list[str] = []
    for row in dataset:
        line = row["text"]
        if _ARTICLE_HEADER.match(line):
            if buffer:
                yield "".join(buffer)
                buffer = []
        buffer.append(line)
    if buffer:
        yield "".join(buffer)


def build_token_sequences(
    tokenizer,
    *,
    dataset_repo: str,
    dataset_revision: str,
    sequence_length: int,
    num_sequences: int,
    min_article_tokens: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build BOS-prefixed, full-length blocks that never cross article boundaries."""
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    body = sequence_length - 1
    sequences: list[list[int]] = []
    article_ids: list[int] = []
    article_id = 0
    for article in iter_wikitext_articles(dataset_repo, dataset_revision):
        token_ids = tokenizer.encode(article)
        if len(token_ids) < min_article_tokens:
            continue
        for start in range(0, len(token_ids) - body + 1, body):
            block = token_ids[start : start + body]
            sequences.append([bos] + block)
            article_ids.append(article_id)
            if len(sequences) >= num_sequences:
                return torch.tensor(sequences), torch.tensor(article_ids)
        article_id += 1
    return torch.tensor(sequences), torch.tensor(article_ids)


def load_pinned_model(model_repo: str, model_revision: str, device: str):
    """Load a pinned HF checkpoint, then apply the legacy TransformerLens transforms."""
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        revision=model_revision,
        torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        "gpt2",
        hf_model=hf_model,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def residual_hook_name(layer: int, n_layers: int) -> str:
    if layer < n_layers:
        return f"blocks.{layer}.hook_resid_pre"
    if layer == n_layers:
        return f"blocks.{n_layers - 1}.hook_resid_post"
    raise ValueError(f"layer must be in [0, {n_layers}], got {layer}")


@torch.no_grad()
def cache_range(
    model,
    tokens: torch.Tensor,
    article_ids: torch.Tensor,
    *,
    layers: list[int],
    output_dir: Path,
    start: int,
    end: int,
    batch_size: int,
    shard_size: int,
    metadata: dict,
) -> None:
    names = {layer: residual_hook_name(layer, model.cfg.n_layers) for layer in layers}
    name_set = set(names.values())
    for shard_start in range(start, end, shard_size):
        shard_end = min(shard_start + shard_size, end)
        chunks = {layer: [] for layer in layers}
        for batch_start in range(shard_start, shard_end, batch_size):
            batch_end = min(batch_start + batch_size, shard_end)
            batch = tokens[batch_start:batch_end].to(model.cfg.device)
            _, cache = model.run_with_cache(batch, names_filter=lambda name: name in name_set)
            for layer, hook_name in names.items():
                chunks[layer].append(cache[hook_name].to(torch.float16).cpu())
        residuals = {layer: torch.cat(parts) for layer, parts in chunks.items()}
        shard_index = shard_start // shard_size
        path = output_dir / f"shard_{shard_index:04d}.pt"
        torch.save(
            {
                "residuals": residuals,
                "tokens": tokens[shard_start:shard_end],
                "article_ids": article_ids[shard_start:shard_end],
                "meta": {**metadata, "sequence_range": [shard_start, shard_end]},
            },
            path,
        )
        print(f"wrote {path} sequences=[{shard_start},{shard_end})", flush=True)


def package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "transformer-lens", "datasets", "huggingface-hub")
    return {name: importlib.metadata.version(name) for name in names}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-repo", default="openai-community/gpt2")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-repo", default="Salesforce/wikitext")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--num-sequences", type=int, default=6_000)
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=1_000)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_repo, revision=args.model_revision
    )
    tokens, article_ids = build_token_sequences(
        tokenizer,
        dataset_repo=args.dataset_repo,
        dataset_revision=args.dataset_revision,
        sequence_length=args.sequence_length,
        num_sequences=args.num_sequences,
    )
    if len(tokens) != args.num_sequences:
        raise RuntimeError(f"requested {args.num_sequences} sequences, built {len(tokens)}")
    if tokens.shape != (args.num_sequences, args.sequence_length):
        raise RuntimeError(f"unexpected token shape: {tuple(tokens.shape)}")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_repo": args.model_repo,
        "model_revision": args.model_revision,
        "dataset_repo": args.dataset_repo,
        "dataset_revision": args.dataset_revision,
        "dataset_config": "wikitext-103-raw-v1",
        "source_repo": "https://github.com/aniket-desh/tensor-network-futurelens",
        "source_revision": args.source_revision,
        "source_protocol": "legacy cache_residuals.py with pinned revisions",
        "sequence_length": args.sequence_length,
        "num_sequences": args.num_sequences,
        "layers": args.layers,
        "article_count": int(article_ids.unique().numel()),
        "package_versions": package_versions(),
        "store_dtype": "float16",
        "model_dtype": "float32",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    torch.save(
        {"tokens": tokens, "article_ids": article_ids, "meta": metadata},
        args.output_dir / "tokens.pt",
    )
    print(json.dumps({**metadata, "token_shape": list(tokens.shape)}, indent=2), flush=True)
    model = load_pinned_model(args.model_repo, args.model_revision, args.device)
    cache_range(
        model,
        tokens,
        article_ids,
        layers=args.layers,
        output_dir=args.output_dir,
        start=0,
        end=len(tokens),
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        metadata={**metadata, "n_layers": model.cfg.n_layers, "d_model": model.cfg.d_model},
    )
    return metadata


if __name__ == "__main__":
    main()
