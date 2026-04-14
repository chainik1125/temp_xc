#!/usr/bin/env python3
"""
cache_activations.py — Cache Gemma 2 2B activations over FineWeb sequences.

Uses HuggingFace transformers with register_forward_hook to extract activations
from specified layers. Saves as memory-mapped .npy files.

Usage:
    python cache_activations.py                    # cache all layers
    python cache_activations.py --layers mid_res   # cache one layer
    python cache_activations.py --num-chains 1000  # override chain count
"""

import argparse
import os

import numpy as np
import torch
from tqdm.auto import tqdm

from config import (
    MODEL_NAME, DEVICE, SEED, CACHE_DIR,
    NUM_CHAINS, SEQ_LENGTH, CACHE_BATCH_SIZE,
    DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT,
    LAYER_SPECS,
)


def load_model_and_tokenizer() -> tuple:
    """Load Gemma 2 2B via HuggingFace transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  Model loaded on {DEVICE}, dtype=bfloat16")
    return model, tokenizer


def load_fineweb_texts(num_chains: int) -> list[str]:
    """Load text samples from FineWeb via streaming."""
    from datasets import load_dataset

    print(f"Loading {DATASET_NAME}/{DATASET_SUBSET} ({num_chains} samples, streaming)...")
    ds = load_dataset(
        DATASET_NAME,
        DATASET_SUBSET,
        split=DATASET_SPLIT,
        streaming=True,
    )
    texts: list[str] = []
    for sample in tqdm(ds, total=num_chains, desc="Fetching texts"):
        text = sample["text"]
        # Skip very short texts
        if len(text) < 100:
            continue
        texts.append(text)
        if len(texts) >= num_chains:
            break
    print(f"  Loaded {len(texts)} text samples")
    return texts


def tokenize_texts(
    tokenizer, texts: list[str], seq_length: int,
) -> torch.Tensor:
    """Tokenize texts into fixed-length token sequences.

    Returns: (num_chains, seq_length) int tensor of token IDs.
    """
    print(f"Tokenizing {len(texts)} texts to seq_length={seq_length}...")
    all_tokens: list[torch.Tensor] = []
    for text in tqdm(texts, desc="Tokenizing", leave=False):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_length,
            padding="max_length",
            add_special_tokens=True,
        )
        all_tokens.append(enc["input_ids"].squeeze(0))
    token_ids = torch.stack(all_tokens)  # (N, seq_length)
    print(f"  Token tensor shape: {token_ids.shape}")
    return token_ids


def cache_activations(
    model,
    token_ids: torch.Tensor,
    layers_to_cache: list[str],
    cache_dir: str,
    batch_size: int,
) -> dict[str, str]:
    """Run model with hooks and save activations for all requested layers.

    For "resid" layers: captures the output of the full transformer block.
    For "attn" layers: captures the output of the self-attention sublayer
                       (after o_proj, before residual add).

    Returns: dict mapping layer_key -> path to saved .npy file.
    """
    num_chains = token_ids.shape[0]
    seq_length = token_ids.shape[1]
    os.makedirs(cache_dir, exist_ok=True)

    # Check which layers actually need caching
    specs_to_cache: list[tuple[str, dict]] = []
    for layer_key in layers_to_cache:
        spec = LAYER_SPECS[layer_key]
        out_path = os.path.join(cache_dir, f"{layer_key}.npy")
        if os.path.exists(out_path):
            existing = np.load(out_path, mmap_mode="r")
            if existing.shape == (num_chains, seq_length, spec["d_act"]):
                print(f"  {layer_key}: already cached at {out_path} — skipping")
                continue
        specs_to_cache.append((layer_key, spec))

    if not specs_to_cache:
        print("  All layers already cached.")
        return {lk: os.path.join(cache_dir, f"{lk}.npy") for lk in layers_to_cache}

    # Pre-allocate memory-mapped outputs
    mmaps: dict[str, np.ndarray] = {}
    for layer_key, spec in specs_to_cache:
        out_path = os.path.join(cache_dir, f"{layer_key}.npy")
        mmaps[layer_key] = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32,
            shape=(num_chains, seq_length, spec["d_act"]),
        )
        print(f"  Will cache {layer_key} → {out_path}")

    # Group by actual layer index to minimize hooks
    layer_indices_needed: set[int] = set()
    for _, spec in specs_to_cache:
        layer_indices_needed.add(spec["layer"])

    # Storage for hook outputs (reused across batches)
    captured: dict[str, torch.Tensor] = {}

    # Register hooks once before the loop
    hooks = []
    for layer_key, spec in specs_to_cache:
        layer_idx = spec["layer"]
        component = spec["component"]

        if component == "resid":
            target_module = model.model.layers[layer_idx]
        elif component == "attn":
            target_module = model.model.layers[layer_idx].self_attn

        def make_hook(k: str) -> callable:
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    acts = output[0]
                else:
                    acts = output
                if acts.dim() == 4:
                    acts = acts.reshape(acts.shape[0], acts.shape[1], -1)
                captured[k] = acts.detach().float().cpu()
            return hook_fn

        h = target_module.register_forward_hook(make_hook(layer_key))
        hooks.append(h)

    # Process batches
    for start in tqdm(range(0, num_chains, batch_size), desc="Caching activations"):
        end = min(start + batch_size, num_chains)
        batch_tokens = token_ids[start:end].to(DEVICE)

        captured.clear()

        # Forward pass
        with torch.no_grad():
            model(batch_tokens)

        # Write to mmaps
        for layer_key, spec in specs_to_cache:
            acts = captured[layer_key]  # (B, S, d)
            d_act = spec["d_act"]
            if acts.shape[-1] != d_act:
                acts = acts[..., :d_act]
            mmaps[layer_key][start:end] = acts.numpy()

        del batch_tokens

    # Remove hooks
    for h in hooks:
        h.remove()

    # Flush all mmaps
    for layer_key in mmaps:
        mmaps[layer_key].flush()
        path = os.path.join(cache_dir, f"{layer_key}.npy")
        print(f"  Done: {path} ({os.path.getsize(path) / 1e9:.2f} GB)")

    return {lk: os.path.join(cache_dir, f"{lk}.npy") for lk in layers_to_cache}


def main():
    parser = argparse.ArgumentParser(description="Cache Gemma 2 2B activations over FineWeb")
    parser.add_argument(
        "--layers", nargs="+", default=None,
        choices=list(LAYER_SPECS.keys()),
        help="Which layers to cache (default: all)",
    )
    parser.add_argument("--num-chains", type=int, default=NUM_CHAINS)
    parser.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    parser.add_argument("--batch-size", type=int, default=CACHE_BATCH_SIZE)
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    layers_to_cache = args.layers or list(LAYER_SPECS.keys())

    print("=" * 70)
    print("  GEMMA 2 2B ACTIVATION CACHING")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Layers: {layers_to_cache}")
    print(f"  Chains: {args.num_chains}, Seq length: {args.seq_length}")
    print(f"  Cache dir: {args.cache_dir}")
    print("=" * 70)

    # Load model and data
    model, tokenizer = load_model_and_tokenizer()
    texts = load_fineweb_texts(args.num_chains)
    token_ids = tokenize_texts(tokenizer, texts, args.seq_length)

    # Save token IDs for autointerp
    os.makedirs(args.cache_dir, exist_ok=True)
    token_path = os.path.join(args.cache_dir, "token_ids.npy")
    np.save(token_path, token_ids.numpy())
    print(f"  Saved token IDs: {token_path}")

    # Free model memory before caching if needed
    cache_activations(model, token_ids, layers_to_cache, args.cache_dir, args.batch_size)

    print("\nAll activations cached successfully.")


if __name__ == "__main__":
    main()
