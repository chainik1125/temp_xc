"""Bhalla et al. 2025 (T-SAE / arXiv:2511.05541) — clamp-style steering.

Procedure (latent-space round-trip with error preserved):

    z = encoder(x)
    z'[i] = clamp_value
    z'[j] = z[j]   for j != i
    x_modified = decoder(z') + (x - decoder(z))

For our linear TopK decoder this is mathematically equivalent to additive
steering with a *context-dependent* coefficient (clamp_value - z[i]). We
implement the explicit round-trip anyway to (a) match the cited procedure
verbatim, and (b) keep the hook compatible with future non-linear decoders.

Single-feature only — Bhalla et al. clamp one feature at a time. The clamp
grid is in absolute activation magnitudes (10..15000 for Gemma-2 SAEs in
the paper); for our Qwen-7B TopK SAE we sweep a smaller grid by default.

    uv run python -m experiments.em_features.clamp_sweep \\
        --ckpt /root/em_features/checkpoints/v2_qwen_l15_sae_arditi_k128_step100000.pt \\
        --arch sae --feature_id 30316 --layer 15 \\
        --clamp_grid 0,1,3,10,30,100,300,1000,3000 --n_rollouts 8 \\
        --out /root/em_features/results/clamp_sae_step100000_feat30316.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
for p in (str(VENDOR_SRC), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--arch", choices=["sae"], default="sae",
                   help="Han clamping is non-trivial (T-window encoder); SAE only for now.")
    p.add_argument("--feature_id", type=int, required=True)
    p.add_argument("--clamp_grid", type=str, default="0,1,3,10,30,100,300,1000,3000",
                   help="Comma-separated absolute activation magnitudes")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--n_rollouts", type=int, default=8,
                   help="Number of generations per EM question")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--judge", choices=["gemini"], default="gemini")
    p.add_argument("--judge_model", default=("gemini-3.1-flash-lite-preview", "gemini-2.5-flash"),
                   nargs="*")
    p.add_argument("--judge_temp", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile_first", action="store_true",
                   help="Print typical z[feature_id] activation magnitude on EM prompts before sweeping")
    p.add_argument("--out_path", type=Path, required=True)
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sae(ckpt_path: Path, device: str):
    from sae_day.sae import TopKSAE
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, cfg


class ClampHook:
    """Forward hook on a Qwen-2 decoder layer. Performs SAE round-trip clamping
    on the layer's residual-stream output:

        output = (decoder(z') - decoder(z)) + output_original
               = (clamp_value - z[i]) * W_dec[i, :] + output_original   (linear decoder)

    We compute the explicit form to match Bhalla 2025's verbatim procedure and
    keep semantics correct if the decoder is ever non-linear.
    """

    def __init__(self, sae, feature_id: int, clamp_value: float):
        self.sae = sae
        self.feature_id = int(feature_id)
        self.c = float(clamp_value)
        self.n_calls = 0
        self.last_z_at_feature = None  # for profiling

    def __call__(self, module, args_in, output):
        # Qwen2DecoderLayer returns a tuple: (hidden_states, ...). Older HF returns just the tensor.
        if isinstance(output, tuple):
            x = output[0]
            tail = output[1:]
        else:
            x = output
            tail = None
        orig_dtype = x.dtype
        with torch.no_grad():
            x_f = x.float()
            shape = x_f.shape  # (B, T, d) or (B, 1, d) during cached generation
            x_flat = x_f.reshape(-1, shape[-1])  # (N, d)

            # Encode (uses sae.b_dec, applies TopK)
            z = self.sae.encode(x_flat)  # (N, d_sae); post-TopK so most entries are 0

            # Track typical activation at the feature for profiling
            self.last_z_at_feature = z[:, self.feature_id].detach().clone()
            self.n_calls += 1

            # Build z' by clamping coordinate i
            z_mod = z.clone()
            z_mod[:, self.feature_id] = self.c

            # Round-trip + add error (mathematically: x + (c - z[:, i]) * W_dec[i, :])
            x_hat = self.sae.decode(z)        # (N, d)
            x_hat_mod = self.sae.decode(z_mod)
            eps = x_flat - x_hat
            x_new_flat = x_hat_mod + eps
            x_new = x_new_flat.reshape(shape).to(orig_dtype)

        if tail is not None:
            return (x_new,) + tail
        return x_new


def get_layer_module(model, layer_idx: int):
    """Find the right transformer block. Qwen-2 HF: model.model.layers[L]."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    raise ValueError(f"Could not find model.model.layers[{layer_idx}]; type={type(model)}")


@torch.no_grad()
def generate_clamped(model, tok, questions, sae, feature_id, clamp_value, layer_idx,
                     n_rollouts, max_new_tokens, seed):
    """Mirror the em_features library's generate_longform_completions exactly so
    the clamp baseline (c=0) is comparable to additive sweep at α=0. Only the
    hook differs (ClampHook instead of ActivationSteerer additive)."""
    hook = ClampHook(sae, feature_id, clamp_value)
    target_module = get_layer_module(model, layer_idx)
    handle = target_module.register_forward_hook(hook)
    out_records = []
    try:
        # Same chat template + flattened replication as the library
        prompts = []
        for q in questions:
            messages = [{"role": "user", "content": q}]
            prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        all_prompts = []
        all_questions = []
        for p, q in zip(prompts, questions):
            for _ in range(n_rollouts):
                all_prompts.append(p)
                all_questions.append(q)
        # Library uses add_special_tokens=False (chat_template already adds them)
        inputs = tok(all_prompts, return_tensors="pt", padding=True,
                     add_special_tokens=False).to(model.device)
        seed_all(seed)
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=1.0,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
        )
        new_tokens = generated[:, inputs.input_ids.shape[1]:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        for q, ans in zip(all_questions, decoded):
            out_records.append({"question": q, "answer": ans})
    finally:
        handle.remove()
    return out_records, hook


@torch.no_grad()
def profile_feature_activation(model, tok, questions, sae, feature_id, layer_idx, n_prompts=8):
    """One forward pass on each prompt. Records z[:, feature_id] across all token positions
    so we can compare clamp values to typical natural firing magnitudes."""
    captured = []
    def cap_hook(module, args_in, output):
        x = output[0] if isinstance(output, tuple) else output
        z = sae.encode(x.float().reshape(-1, x.shape[-1]))
        captured.append(z[:, feature_id].detach().cpu())
        return output
    target = get_layer_module(model, layer_idx)
    h = target.register_forward_hook(cap_hook)
    try:
        for q in questions[:n_prompts]:
            messages = [{"role": "user", "content": q}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(model.device)
            _ = model(**ids)
    finally:
        h.remove()
    if not captured:
        return {}
    flat = torch.cat(captured)
    return {
        "mean": float(flat.mean()), "max": float(flat.max()),
        "median": float(flat.median()), "p90": float(torch.quantile(flat.float(), 0.9)),
        "frac_active": float((flat > 0).float().mean()),
        "n_tokens": int(flat.numel()),
    }


def main():
    args = parse_args()
    enable_determinism()

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[clamp] loading subject model (bad-medical Qwen)...", flush=True)
    base_id = "Qwen/Qwen2.5-7B-Instruct"
    subj_id = "andyrdt/Qwen2.5-7B-Instruct_bad-medical"
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="cuda")
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, subj_id).merge_and_unload()
    except Exception as e:
        print(f"[clamp] PEFT load failed ({e}); using base model", flush=True)
        model = base
    model.eval()
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Don't override padding_side — library leaves tokenizer default (Qwen → 'right')

    # Load SAE
    print(f"[clamp] loading SAE from {args.ckpt}", flush=True)
    sae, cfg = load_sae(args.ckpt, "cuda")

    # Load EM questions
    from open_source_em_features.pipeline.longform_steering import load_em_dataset
    em = load_em_dataset()
    questions = [d["messages"][0]["content"] for d in em]
    print(f"[clamp] {len(questions)} EM questions loaded", flush=True)

    # Optional: profile feature activation magnitude
    if args.profile_first:
        print(f"[clamp] profiling z[{args.feature_id}] across {len(questions)} prompts...", flush=True)
        stats = profile_feature_activation(model, tok, questions, sae, args.feature_id, args.layer)
        print(f"[clamp] feature {args.feature_id} stats: {stats}", flush=True)

    clamp_grid = [float(x) for x in args.clamp_grid.split(",")]
    print(f"[clamp] sweeping clamp values: {clamp_grid}", flush=True)

    from experiments.em_features.gemini_judge import evaluate_generations_with_gemini

    rows = []
    for c in clamp_grid:
        tag = f"clamp_feat{args.feature_id}_c{c:.2f}"
        print(f"\n=== {tag} ===", flush=True)
        gens, hook = generate_clamped(
            model, tok, questions, sae, args.feature_id, c,
            args.layer, args.n_rollouts, args.max_new_tokens, args.seed,
        )
        z_at_feat = hook.last_z_at_feature
        natural_firing_mean = float(z_at_feat.float().mean()) if z_at_feat is not None else None
        print(f"  generated {len(gens)} samples, last natural z[i] mean ≈ {natural_firing_mean}", flush=True)
        align, coh = asyncio.run(evaluate_generations_with_gemini(
            gens, model_name=tuple(args.judge_model) if isinstance(args.judge_model, (list, tuple)) else args.judge_model,
            temperature=args.judge_temp,
        ))
        a_v = [x for x in align if x is not None]
        c_v = [x for x in coh if x is not None]
        rows.append({
            "alpha": c,  # alias clamp value as "alpha" so the existing plot scripts work
            "clamp_value": c,
            "mean_alignment": float(np.mean(a_v)) if a_v else None,
            "mean_coherence": float(np.mean(c_v)) if c_v else None,
            "n_alignment": len(a_v),
            "n_coherence": len(c_v),
            "n_total": len(gens),
            "natural_z_mean": natural_firing_mean,
        })
        ma = rows[-1]["mean_alignment"]; mc = rows[-1]["mean_coherence"]
        print(f"  c={c:>8.2f}  align={ma if ma is None else f'{ma:.2f}'}  coh={mc if mc is None else f'{mc:.2f}'}",
              flush=True)

    out = {
        "meta": {
            "steerer": "clamp",
            "arch": args.arch,
            "layer": args.layer,
            "feature_id": args.feature_id,
            "clamp_grid": clamp_grid,
            "alpha_grid": clamp_grid,
            "n_rollouts": args.n_rollouts,
            "ckpt": str(args.ckpt),
            "procedure": "Bhalla 2025 round-trip + error preserved",
        },
        "rows": rows,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[clamp] wrote {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
