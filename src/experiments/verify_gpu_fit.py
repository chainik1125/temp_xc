#!/usr/bin/env python3
"""Load a registry model in its declared dtype, run one forward pass, and
report peak GPU memory. Use inside a GPU allocation to confirm the 8B models
leave headroom for activation caching.

Usage:
    python scripts/verify_gpu_fit.py --model deepseek-r1-distill-llama-8b
"""
import argparse
import torch

from src.data.nlp.models import get_model_config, list_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list_models())
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = get_model_config(args.model)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[cfg.dtype]
    print(f"Loading {cfg.hf_path} in {cfg.dtype}...")
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_path, torch_dtype=dtype, device_map="cuda",
    )
    model.eval()

    torch.cuda.reset_peak_memory_stats()
    text = "Hello. " * (args.seq_len // 2)
    enc = tok([text] * args.batch_size, return_tensors="pt",
              truncation=True, max_length=args.seq_len,
              padding="max_length").to("cuda")
    with torch.no_grad():
        _ = model(**enc)
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  peak   : {peak:.2f} GB")
    print(f"  device : {total:.2f} GB total")
    print(f"  headroom: {total - peak:.2f} GB")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}")


if __name__ == "__main__":
    main()
