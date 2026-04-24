"""Compute per-token (bad − base) residual-stream diffs at a single layer,
then slide T-token windows over each prompt. Output: ``windows`` tensor of
shape ``(N, T, d_model)`` saved to a .pt, used by run_find_misalignment_features
with --ranking=encoder.

    uv run python -m experiments.em_features.compute_per_token_diffs \
        --dataset /root/em_features/data/medical_advice_prompt_only.jsonl \
        --base Qwen/Qwen2.5-7B-Instruct \
        --bad andyrdt/Qwen2.5-7B-Instruct_bad-medical \
        --layer 15 --T 5 \
        --n_prompts 256 --max_ctx_len 512 \
        --out /root/em_features/results/qwen_l15_per_token_diffs_T5.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--bad", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--n_prompts", type=int, default=256)
    p.add_argument("--max_ctx_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_prompts(path: Path, n: int, tok) -> list[str]:
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
def collect_per_token(model, tok, texts, layer, batch_size, max_ctx_len, device):
    """Return (N_prompts, max_ctx_len, d) acts per layer, with attention mask."""
    tok.padding_side = "left"
    all_acts = []
    all_masks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding="max_length", truncation=True, max_length=max_ctx_len,
                  return_tensors="pt", add_special_tokens=False).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer + 1]  # (B, L, d)
        all_acts.append(h.float().cpu())
        all_masks.append(enc["attention_mask"].cpu())
    return torch.cat(all_acts, dim=0), torch.cat(all_masks, dim=0)


def slide_windows(acts: torch.Tensor, mask: torch.Tensor, T: int) -> torch.Tensor:
    """acts (N, L, d) + mask (N, L) → (M, T, d) windows where all T positions
    are valid (mask==1) — i.e., inside the real prompt, not padding."""
    N, L, d = acts.shape
    out = []
    for n in range(N):
        # Find the span of real tokens (left-padded, so real is at the right end).
        m = mask[n].bool()
        idxs = torch.nonzero(m, as_tuple=False).squeeze(-1)
        if idxs.numel() < T:
            continue
        real_start = int(idxs[0])
        real_end = int(idxs[-1]) + 1  # exclusive
        for t0 in range(real_start, real_end - T + 1):
            out.append(acts[n, t0:t0 + T, :])
    if not out:
        return torch.empty(0, T, d)
    return torch.stack(out, dim=0)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = load_prompts(args.dataset, args.n_prompts, tok)
    print(f"{len(texts)} prompts loaded")

    print(f"loading bad {args.bad}")
    bad = AutoModelForCausalLM.from_pretrained(
        args.bad, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True,
    ).eval()
    bad_acts, mask = collect_per_token(bad, tok, texts, args.layer,
                                       args.batch_size, args.max_ctx_len, args.device)
    del bad; torch.cuda.empty_cache()

    print(f"loading base {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True,
    ).eval()
    base_acts, _ = collect_per_token(base, tok, texts, args.layer,
                                     args.batch_size, args.max_ctx_len, args.device)
    del base; torch.cuda.empty_cache()

    diffs = (bad_acts - base_acts).float()  # (N, L, d)
    print(f"per-token diffs shape={tuple(diffs.shape)}  mean_norm={diffs.norm(dim=-1).mean():.3f}")

    windows = slide_windows(diffs, mask, args.T)
    print(f"sliding windows T={args.T}: {tuple(windows.shape)}")

    torch.save({
        "windows": windows,
        "layer": args.layer,
        "T": args.T,
        "n_prompts": len(texts),
        "base_model": args.base,
        "bad_model": args.bad,
    }, args.out)
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
