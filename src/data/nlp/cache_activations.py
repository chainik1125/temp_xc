#!/usr/bin/env python3
"""Cache residual/attention activations from any registered subject model.

Model-agnostic: resolves hf_path, d_model, dtype, architecture family, and
default layer indices from `src.data.nlp.models`. Adding a new model is
a registry edit, not a code change here.

Two modes:
    --mode forward   (default): batch forward pass on tokenized text; saves
                     shape (N, seq_len, d_model) per layer.
    --mode generate  : autoregressive generation with activation capture;
                     intended for thinking models on reasoning datasets
                     (DeepSeek-R1-Distill on GSM8K/MATH500). Saves ragged
                     traces padded to `--gen_max_new_tokens`.

Datasets (built in):
    fineweb          HuggingFaceFW/fineweb, sample-10BT  (default: forward)
    gsm8k            openai/gsm8k, main                  (generate w/ thinking)
    math500          HuggingFaceH4/MATH-500              (generate w/ thinking)
    coding           bigcode/starcoderdata / codeparrot  (forward)
    custom           --dataset_hf_path + --dataset_subset

Usage:
    python -m src.data.nlp.cache_activations \
        --model deepseek-r1-distill-llama-8b \
        --dataset gsm8k \
        --mode generate \
        --num-sequences 1000 \
        --seq-length 512

    python -m src.data.nlp.cache_activations \
        --model gemma-2-2b-it --dataset fineweb --mode forward \
        --num-sequences 24000 --seq-length 32
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from src.data.nlp.models import (
    get_model_config,
    resid_hook_target,
    attn_hook_target,
    list_models,
)
from src.data.nlp.cache_config import (
    CACHE_BATCH_SIZE,
    DATASET_NAME_DEFAULT,
    DATASET_SPLIT_DEFAULT,
    DATASET_SUBSET_DEFAULT,
    DEVICE,
    NUM_CHAINS,
    SEED,
    SEQ_LENGTH,
    build_layer_specs,
    cache_dir_for,
)


# ──────────────────────────────────────────────────────────────────────── models
def load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = get_model_config(model_name)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[cfg.dtype]

    print(f"Loading {cfg.hf_path} (dtype={cfg.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_path,
        torch_dtype=dtype,
        device_map=DEVICE,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  Model on {DEVICE}, n_layers={cfg.n_layers}, d_model={cfg.d_model}")
    return model, tokenizer, cfg


# ──────────────────────────────────────────────────────────────────── datasets
def _load_text_stream(
    dataset: str,
    num_samples: int,
    hf_path: str | None,
    subset: str | None,
    split: str,
) -> list[str]:
    from datasets import load_dataset

    # Large web-scale datasets (fineweb, coding) can be pre-fetched as a
    # JSONL slice on a login node with network access. If the slice exists,
    # prefer it — avoids the "no network on compute nodes" issue on HPC.
    # Pre-fetch via: scripts/prefetch_text_dataset.sh
    from src.data.nlp.cache_config import DATA_ROOT
    prefetch_dir = os.path.join(DATA_ROOT, "prefetched")
    prefetch_candidates = [
        os.path.join(prefetch_dir, f"{dataset}_{num_samples}.jsonl"),
        os.path.join(prefetch_dir, f"{dataset}.jsonl"),
    ]
    for pp in prefetch_candidates:
        if os.path.exists(pp):
            print(f"Loading pre-fetched slice: {pp}")
            out: list[str] = []
            with open(pp) as f:
                for line in f:
                    rec = json.loads(line)
                    txt = rec.get("text") or rec.get("content") or ""
                    if txt:
                        out.append(txt)
                    if len(out) >= num_samples:
                        break
            print(f"  loaded {len(out)} texts from prefetch")
            return out

    # Small curated datasets must be fully cached (streaming requires network,
    # Trillium compute nodes are offline). Large web-scale datasets stay
    # streaming and require login-node pre-fetching of a slice.
    stream = True
    if dataset == "fineweb":
        hf_path, subset, split = "HuggingFaceFW/fineweb", "sample-10BT", "train"
    elif dataset == "coding":
        hf_path, subset, split = "codeparrot/codeparrot-clean", None, "train"
    elif dataset == "gsm8k":
        hf_path, subset, split = "openai/gsm8k", "main", "train"
        stream = False
    elif dataset == "math500":
        hf_path, subset, split = "HuggingFaceH4/MATH-500", None, "test"
        stream = False
    elif dataset == "custom":
        assert hf_path, "--dataset_hf_path required for custom"
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    print(f"Loading {hf_path} / {subset or '-'} / {split} "
          f"({num_samples} samples, streaming={stream})")
    kwargs = dict(split=split, streaming=stream)
    if subset:
        ds = load_dataset(hf_path, subset, **kwargs)
    else:
        ds = load_dataset(hf_path, **kwargs)

    # Column guessing — most LM datasets use "text" but reasoning sets use
    # "question"/"problem".
    def extract(sample: dict) -> str | None:
        for col in ("text", "content", "question", "problem", "Question"):
            if col in sample and sample[col]:
                return str(sample[col])
        return None

    out: list[str] = []
    for sample in tqdm(ds, total=num_samples, desc="Fetching texts"):
        txt = extract(sample)
        if txt is None or len(txt) < 20:
            continue
        out.append(txt)
        if len(out) >= num_samples:
            break
    return out


# ─────────────────────────────────────────────────────────────── forward mode
def _tokenize_fixed_length(tokenizer, texts: list[str], seq_length: int) -> torch.Tensor:
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing", leave=False):
        enc = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=seq_length, padding="max_length",
            add_special_tokens=True,
        )
        all_tokens.append(enc["input_ids"].squeeze(0))
    return torch.stack(all_tokens)


def _register_hooks(
    model,
    specs_to_cache: list[tuple[str, dict]],
    captured: dict[str, torch.Tensor],
    family: str,
) -> list:
    hooks = []
    for layer_key, spec in specs_to_cache:
        comp = spec["component"]
        if comp == "resid":
            target = resid_hook_target(model, spec["layer"], family)
        elif comp == "attn":
            target = attn_hook_target(model, spec["layer"], family)
        else:
            raise ValueError(f"Unknown component: {comp}")

        def make_hook(k: str):
            def hook_fn(module, inp, output):
                acts = output[0] if isinstance(output, tuple) else output
                if acts.dim() == 4:
                    acts = acts.reshape(acts.shape[0], acts.shape[1], -1)
                captured[k] = acts.detach().float().cpu()
            return hook_fn

        hooks.append(target.register_forward_hook(make_hook(layer_key)))
    return hooks


def cache_forward(
    model,
    tokenizer,
    cfg,
    texts: list[str],
    layer_specs: dict[str, dict],
    cache_dir: str,
    seq_length: int,
    batch_size: int,
) -> dict[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    token_ids = _tokenize_fixed_length(tokenizer, texts, seq_length)
    num = token_ids.shape[0]

    np.save(os.path.join(cache_dir, "token_ids.npy"), token_ids.numpy())

    specs_to_cache = []
    mmaps: dict[str, np.ndarray] = {}
    for key, spec in layer_specs.items():
        path = os.path.join(cache_dir, f"{key}.npy")
        if os.path.exists(path):
            existing = np.load(path, mmap_mode="r")
            if existing.shape == (num, seq_length, spec["d_act"]):
                print(f"  {key}: already cached — skipping")
                continue
        specs_to_cache.append((key, spec))
        mmaps[key] = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32,
            shape=(num, seq_length, spec["d_act"]),
        )
        print(f"  Will cache {key} → {path}")

    if not specs_to_cache:
        return {k: os.path.join(cache_dir, f"{k}.npy") for k in layer_specs}

    captured: dict[str, torch.Tensor] = {}
    hooks = _register_hooks(model, specs_to_cache, captured, cfg.architecture_family)

    try:
        for start in tqdm(range(0, num, batch_size), desc="Forward cache"):
            end = min(start + batch_size, num)
            batch = token_ids[start:end].to(DEVICE)
            captured.clear()
            with torch.no_grad():
                model(batch)
            for key, spec in specs_to_cache:
                acts = captured[key]
                if acts.shape[-1] != spec["d_act"]:
                    acts = acts[..., : spec["d_act"]]
                mmaps[key][start:end] = acts.numpy()
            del batch
    finally:
        for h in hooks:
            h.remove()

    for key in mmaps:
        mmaps[key].flush()
        path = os.path.join(cache_dir, f"{key}.npy")
        print(f"  Saved {path} ({os.path.getsize(path) / 1e9:.2f} GB)")

    return {k: os.path.join(cache_dir, f"{k}.npy") for k in layer_specs}


# ─────────────────────────────────────────────────────────────── generate mode
def _format_prompt(tokenizer, cfg, text: str) -> str:
    """Apply the model's chat template for thinking models; plain text otherwise."""
    if cfg.is_thinking_model and hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": text}]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass
    return text


def cache_generate(
    model,
    tokenizer,
    cfg,
    prompts: list[str],
    layer_specs: dict[str, dict],
    cache_dir: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, str]:
    """Autoregressive generation with activation capture at each hooked layer.

    Produces fixed-length (num_prompts, max_new_tokens, d_model) tensors per
    layer. Short completions are zero-padded and the completion length is
    recorded in a sidecar `lengths.npy` for downstream masking.
    """
    os.makedirs(cache_dir, exist_ok=True)
    num = len(prompts)

    # Preallocate
    mmaps: dict[str, np.ndarray] = {}
    for key, spec in layer_specs.items():
        path = os.path.join(cache_dir, f"{key}.npy")
        mmaps[key] = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32,
            shape=(num, max_new_tokens, spec["d_act"]),
        )

    lengths = np.zeros(num, dtype=np.int32)
    all_trace_tokens: list[list[int]] = []

    captured: dict[str, list[torch.Tensor]] = {}
    specs_list = list(layer_specs.items())

    def make_hook(k: str):
        def hook_fn(module, inp, output):
            acts = output[0] if isinstance(output, tuple) else output
            if acts.dim() == 4:
                acts = acts.reshape(acts.shape[0], acts.shape[1], -1)
            # During incremental decoding acts has seq_len 1 (past key cache);
            # during the prefill step it has the full prompt length. We only
            # care about the *generated* steps, so we keep only the last token
            # of every forward pass.
            captured[k].append(acts[:, -1:, :].detach().float().cpu())
        return hook_fn

    hooks = []
    for key, spec in specs_list:
        captured[key] = []
        if spec["component"] == "resid":
            target = resid_hook_target(model, spec["layer"], cfg.architecture_family)
        else:
            target = attn_hook_target(model, spec["layer"], cfg.architecture_family)
        hooks.append(target.register_forward_hook(make_hook(key)))

    try:
        for i, prompt in enumerate(tqdm(prompts, desc="Generate cache")):
            for key in captured:
                captured[key].clear()

            text = _format_prompt(tokenizer, cfg, prompt)
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=2048).to(DEVICE)
            prompt_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_ids = out[0, prompt_len:].tolist()
            all_trace_tokens.append(gen_ids)

            # captured[key] is a list of tensors — first entry has shape
            # (1, prompt_len, d) (prefill) then N entries of (1, 1, d)
            # (one per generated token). We skip the prefill and concat.
            for key, spec in specs_list:
                chunks = captured[key]
                if not chunks:
                    continue
                decode_chunks = [c for c in chunks if c.shape[1] == 1]
                if not decode_chunks:
                    continue
                trace = torch.cat(decode_chunks, dim=1).squeeze(0).numpy()
                n = min(trace.shape[0], max_new_tokens)
                mmaps[key][i, :n] = trace[:n]
                lengths[i] = n
    finally:
        for h in hooks:
            h.remove()

    np.save(os.path.join(cache_dir, "trace_lengths.npy"), lengths)
    with open(os.path.join(cache_dir, "trace_tokens.jsonl"), "w") as f:
        for toks in all_trace_tokens:
            f.write(json.dumps(toks) + "\n")

    for key in mmaps:
        mmaps[key].flush()
        path = os.path.join(cache_dir, f"{key}.npy")
        print(f"  Saved {path}")

    return {k: os.path.join(cache_dir, f"{k}.npy") for k in layer_specs}


# ────────────────────────────────────────────────────────────────────── main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list_models())
    p.add_argument("--dataset", default="fineweb",
                   choices=["fineweb", "coding", "gsm8k", "math500", "custom"])
    p.add_argument("--dataset_hf_path", type=str, default=None)
    p.add_argument("--dataset_subset", type=str, default=None)
    p.add_argument("--dataset_split", type=str, default=DATASET_SPLIT_DEFAULT)
    p.add_argument("--mode", choices=["forward", "generate"], default="forward")
    p.add_argument("--num-sequences", type=int, default=NUM_CHAINS)
    p.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    p.add_argument("--batch-size", type=int, default=CACHE_BATCH_SIZE)
    p.add_argument("--layer_indices", type=int, nargs="+", default=None,
                   help="Override registry defaults")
    p.add_argument("--components", type=str, nargs="+", default=["resid"],
                   choices=["resid", "attn"])
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override default cache path")
    p.add_argument("--gen_max_new_tokens", type=int, default=512,
                   help="generate mode only")
    p.add_argument("--gen_temperature", type=float, default=0.6)
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    layer_indices = tuple(args.layer_indices) if args.layer_indices else None
    layer_specs = build_layer_specs(
        args.model,
        layer_indices=layer_indices,
        components=tuple(args.components),
    )
    cache_dir = args.output_dir or cache_dir_for(args.model, args.dataset)

    print("=" * 70)
    print(f"  Subject model:  {args.model}")
    print(f"  Dataset:        {args.dataset}  (mode={args.mode})")
    print(f"  Layers:         {list(layer_specs.keys())}")
    print(f"  Output:         {cache_dir}")
    print("=" * 70)

    model, tokenizer, cfg = load_model_and_tokenizer(args.model)
    texts = _load_text_stream(
        args.dataset, args.num_sequences,
        args.dataset_hf_path, args.dataset_subset, args.dataset_split,
    )

    if args.mode == "forward":
        cache_forward(
            model, tokenizer, cfg, texts, layer_specs, cache_dir,
            args.seq_length, args.batch_size,
        )
    else:
        cache_generate(
            model, tokenizer, cfg, texts, layer_specs, cache_dir,
            args.gen_max_new_tokens, args.gen_temperature,
        )

    # Write a sidecar with the runtime layer spec so downstream code can
    # recover layer metadata without re-running build_layer_specs.
    with open(os.path.join(cache_dir, "layer_specs.json"), "w") as f:
        json.dump({
            "model": args.model,
            "hf_path": cfg.hf_path,
            "d_model": cfg.d_model,
            "layer_specs": layer_specs,
            "mode": args.mode,
        }, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
