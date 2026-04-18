#!/usr/bin/env python3
"""Generate + cache reasoning traces on GSM8K/MATH500 with a thinking model.

Thin wrapper around `src.data.nlp.cache_activations` that enforces
the right defaults for the reasoning track: `--mode generate`, a thinking
model (`is_thinking_model == True`), longer `gen_max_new_tokens`.

Usage:
    python scripts/cache_reasoning_traces.py \
        --model deepseek-r1-distill-llama-8b \
        --dataset gsm8k --num-sequences 1000 \
        --layer_indices 12 24

Rationale: we want activations on the model's own chain of thought, not on
the problem text. This is step 4 in the exploration sprint priority order —
the highest-leverage dataset because Venhoff et al. already labeled the
reasoning behaviors.
"""
import argparse
import subprocess
import sys

from src.data.nlp.models import get_model_config, list_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list_models())
    ap.add_argument("--dataset", choices=["gsm8k", "math500"], default="gsm8k")
    ap.add_argument("--num-sequences", type=int, default=1000)
    ap.add_argument("--gen_max_new_tokens", type=int, default=1024)
    ap.add_argument("--gen_temperature", type=float, default=0.6)
    ap.add_argument("--layer_indices", type=int, nargs="+", default=None)
    ap.add_argument("--components", type=str, nargs="+", default=["resid"])
    args = ap.parse_args()

    cfg = get_model_config(args.model)
    if not cfg.is_thinking_model:
        print(
            f"WARNING: {args.model}.is_thinking_model == False. Reasoning "
            f"traces from this model will not contain <think> blocks. "
            f"Proceeding anyway (useful as a base-model comparison)."
        )

    cmd = [
        sys.executable, "-m", "src.data.nlp.cache_activations",
        "--model", args.model,
        "--dataset", args.dataset,
        "--mode", "generate",
        "--num-sequences", str(args.num_sequences),
        "--gen_max_new_tokens", str(args.gen_max_new_tokens),
        "--gen_temperature", str(args.gen_temperature),
        "--components", *args.components,
    ]
    if args.layer_indices:
        cmd += ["--layer_indices", *map(str, args.layer_indices)]

    print(" ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
