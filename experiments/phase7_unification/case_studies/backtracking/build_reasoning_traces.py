#!/usr/bin/env python3
"""Stage 0: generate reasoning traces with DeepSeek-R1-Distill-Llama-8B.

For each prompt, applies the model's chat template, generates greedily up to
MAX_NEW_TOKENS, and persists the result to traces.jsonl. Each record has
enough information for Stages 1 and 2 to reconstruct token-level positions:

    {
      "trace_id": str,
      "category": str,
      "prompt_text": str,
      "input_token_ids": list[int],          # tokens of the chat-templated prompt
      "input_len": int,                       # = len(input_token_ids)
      "output_token_ids": list[int],          # newly generated tokens (no prompt)
      "full_token_ids": list[int],            # input_token_ids + output_token_ids
      "generation_text": str,                 # decoded output_token_ids
      "think_open_pos": int | None,           # index in full_token_ids of "<think>" if present
      "think_close_pos": int | None           # likewise for "</think>"
    }

Run from repo root:

    TQDM_DISABLE=1 uv run python -m experiments.phase7_unification.case_studies.backtracking.build_reasoning_traces

Idempotent: skips if traces.jsonl already exists with the requested N. Use --force to rebuild.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("TQDM_DISABLE", "1")

# Ensure repo root is on sys.path when invoked as a script.
_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.nlp.cache_activations import load_model_and_tokenizer  # noqa: E402

from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    GEN_SEED,
    MAX_NEW_TOKENS,
    N_PROMPTS,
    SUBJECT_MODEL,
    TRACES_PATH,
    ensure_dirs,
)
from experiments.phase7_unification.case_studies.backtracking.prompts import (  # noqa: E402
    get_prompts,
)


def _find_subseq(haystack: list[int], needle: list[int]) -> int | None:
    """Return index of first occurrence of needle in haystack (or None)."""
    if not needle:
        return None
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i : i + m] == needle:
            return i
    return None


def _think_token_positions(tokenizer, full_ids: list[int]) -> tuple[int | None, int | None]:
    """Locate <think> and </think> tag tokens in `full_ids`.

    DeepSeek-R1-Distill encodes both as single special tokens. Falls back to
    multi-token search if the tokenizer lacks them.
    """
    open_ids = tokenizer.encode("<think>", add_special_tokens=False)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    open_pos = _find_subseq(full_ids, open_ids)
    close_pos = _find_subseq(full_ids, close_ids)
    return open_pos, close_pos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N_PROMPTS, help="number of prompts (default: N_PROMPTS)")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--force", action="store_true", help="overwrite existing traces.jsonl")
    parser.add_argument("--seed", type=int, default=GEN_SEED)
    args = parser.parse_args()

    ensure_dirs()
    if TRACES_PATH.exists() and not args.force:
        with TRACES_PATH.open() as f:
            n_existing = sum(1 for _ in f)
        if n_existing >= args.n:
            print(f"[build_reasoning_traces] {TRACES_PATH} already has {n_existing} traces; use --force to rebuild")
            return

    prompts = get_prompts(args.n, seed=args.seed)
    print(f"[build_reasoning_traces] N={len(prompts)} prompts; loading {SUBJECT_MODEL}…")

    model, tokenizer, cfg = load_model_and_tokenizer(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device

    torch.manual_seed(args.seed)

    t0 = time.time()
    with TRACES_PATH.open("w") as f:
        for i, p in enumerate(prompts):
            messages = [{"role": "user", "content": p["text"]}]
            encoded = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            # transformers >=5 returns a BatchEncoding (dict-like) by default;
            # earlier versions returned a bare tensor. Handle both.
            if isinstance(encoded, torch.Tensor):
                input_ids = encoded.to(device)
            else:
                input_ids = encoded["input_ids"].to(device)
            input_len = int(input_ids.shape[1])

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=eos_id,
                )
            full_ids = out[0].tolist()
            output_ids = full_ids[input_len:]
            gen_text = tokenizer.decode(output_ids, skip_special_tokens=False)
            think_open, think_close = _think_token_positions(tokenizer, full_ids)

            rec = {
                "trace_id": p["id"],
                "category": p["category"],
                "prompt_text": p["text"],
                "input_token_ids": full_ids[:input_len],
                "input_len": input_len,
                "output_token_ids": output_ids,
                "full_token_ids": full_ids,
                "generation_text": gen_text,
                "think_open_pos": think_open,
                "think_close_pos": think_close,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            elapsed = time.time() - t0
            print(
                f"  [{i+1:>4}/{len(prompts)}] {p['id']:<14} "
                f"in_len={input_len} out_len={len(output_ids)} "
                f"think=({think_open},{think_close}) "
                f"({elapsed/(i+1):.1f}s/trace)"
            )

    print(f"[build_reasoning_traces] wrote {TRACES_PATH} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
