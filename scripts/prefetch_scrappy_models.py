"""Prefetch Llama-3.1-8B + DeepSeek-R1-Distill-Llama-8B into HF_HOME.

Invoked by scripts/prefetch_scrappy_models.sh. Single-purpose: download
the two 8B models so the first scrappy cycle doesn't block on shards.

Force use_fast=True on tokenizers per our 2026-04-22 fix (encode_plus
was removed in modern transformers; only fast tokenizers expose the
offset-mapping API the vendor pipeline depends on).
"""

from __future__ import annotations

import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "meta-llama/Llama-3.1-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
]


def main() -> int:
    for m in MODELS:
        print(f"[prefetch] tokenizer: {m}", flush=True)
        AutoTokenizer.from_pretrained(m, use_fast=True)
        print(f"[prefetch] model weights (bf16): {m}", flush=True)
        AutoModelForCausalLM.from_pretrained(m, torch_dtype="bfloat16")
    print("[ok] both models cached in $HF_HOME", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
