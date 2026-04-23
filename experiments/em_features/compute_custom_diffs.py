"""Compute bad−base diff vectors at non-resid_post hookpoints.

em-features' pipeline captures diffs via ``output_hidden_states`` (resid_post
and resid_pre come for free), but ``resid_mid`` and ``ln1_normalized`` require
module-level hooks. This script runs both the base and bad-medical Qwen
models over ``medical_advice_prompt_only.jsonl``, captures prompt_last
activations at each requested hookpoint, and saves ``(d_model,)`` mean-diff
tensors to ``--out_dir/custom_diffs_{hookpoint}_L{layer}.pt``.

    uv run python -m experiments.em_features.compute_custom_diffs \
        --dataset /root/em_features/data/medical_advice_prompt_only.jsonl \
        --base Qwen/Qwen2.5-7B-Instruct \
        --bad andyrdt/Qwen2.5-7B-Instruct_bad-medical \
        --hookpoints resid_mid ln1_normalized \
        --layer 15 \
        --out_dir /root/em_features/results/qwen_l15_custom_diffs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.em_features.streaming_buffer import HookpointExtractor, VALID_HOOKPOINTS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--base", type=str, required=True)
    p.add_argument("--bad", type=str, required=True)
    p.add_argument("--hookpoints", nargs="+", required=True,
                   choices=list(VALID_HOOKPOINTS))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_ctx_len", type=int, default=512)
    p.add_argument("--n_prompts", type=int, default=512)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_prompts(path: Path, n: int, tok: AutoTokenizer) -> list[str]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[:n]
    texts = []
    for row in rows:
        if "messages" in row:
            texts.append(tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True))
        else:
            texts.append(row.get("text") or row.get("prompt") or next(iter(row.values())))
    return texts


@torch.no_grad()
def collect_pooled(model, tok, texts, hookpoints, layer, batch_size, max_ctx_len, device):
    """Run the model on the prompts and return {hp: (N, d_model)} prompt_last pooled."""
    tok.padding_side = "left"
    per_hp: dict[str, list[torch.Tensor]] = {hp: [] for hp in hookpoints}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding="max_length", truncation=True, max_length=max_ctx_len,
                  return_tensors="pt", add_special_tokens=False).to(device)
        # For resid_post/resid_pre we just need output_hidden_states.
        need_hs = any(hp in ("resid_post", "resid_pre") for hp in hookpoints)
        module_hps = [hp for hp in hookpoints if hp in ("resid_mid", "ln1_normalized")]
        extractors: list[HookpointExtractor] = []
        for hp in module_hps:
            ext = HookpointExtractor(model, hp, layer)
            ext.__enter__()
            extractors.append(ext)
        try:
            out = model(**enc, output_hidden_states=need_hs, use_cache=False)
        finally:
            for ext in extractors:
                ext.__exit__()
        for hp in hookpoints:
            if hp == "resid_post":
                acts = out.hidden_states[layer + 1]
            elif hp == "resid_pre":
                acts = out.hidden_states[layer]
            else:
                acts = next(ext.captured for ext in extractors if ext.kind == hp)
            pooled = acts[:, -1, :]  # prompt_last with left padding
            per_hp[hp].append(pooled.float().cpu())
    return {hp: torch.cat(per_hp[hp], dim=0) for hp in hookpoints}


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = load_prompts(args.dataset, args.n_prompts, tok)
    print(f"loaded {len(texts)} prompts")

    # Pass 1: bad model.
    print(f"loading bad model: {args.bad}")
    bad = AutoModelForCausalLM.from_pretrained(args.bad, torch_dtype=torch.bfloat16,
                                               device_map=args.device, trust_remote_code=True).eval()
    bad_pooled = collect_pooled(bad, tok, texts, args.hookpoints, args.layer,
                                args.batch_size, args.max_ctx_len, args.device)
    del bad
    torch.cuda.empty_cache()

    # Pass 2: base model.
    print(f"loading base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
                                                device_map=args.device, trust_remote_code=True).eval()
    base_pooled = collect_pooled(base, tok, texts, args.hookpoints, args.layer,
                                 args.batch_size, args.max_ctx_len, args.device)
    del base
    torch.cuda.empty_cache()

    for hp in args.hookpoints:
        diff = (bad_pooled[hp].mean(dim=0) - base_pooled[hp].mean(dim=0)).float()
        out_path = args.out_dir / f"custom_diffs_{hp}_L{args.layer}.pt"
        torch.save({"diff": diff, "hookpoint": hp, "layer": args.layer,
                    "n_prompts": len(texts),
                    "bad_model": args.bad, "base_model": args.base}, out_path)
        print(f"wrote {out_path}  (norm={diff.norm().item():.3f})")


if __name__ == "__main__":
    main()
