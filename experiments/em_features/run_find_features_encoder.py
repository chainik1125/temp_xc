"""Encoder-side feature attribution (Wang et al. 2025, arXiv:2506.19823).

For each SAE/Han latent i, rank by:
    Δz̄_i = E_E[ z_i(M_D) ] − E_E[ z_i(M) ]
where z_i is the SAE/Han encoded activation (post-TopK, scalar firing) at
the trained layer, averaged over tokens (or windows for Han) on a probe
dataset E. M = base Qwen, M_D = bad-medical Qwen.

Outputs the same top_200 schema consumed by frontier_sweep.py.

    uv run python -m experiments.em_features.run_find_features_encoder \\
        --ckpt /root/em_features/checkpoints/v2_qwen_l15_sae_arditi_k128_step100000.pt \\
        --arch sae \\
        --dataset /root/em_features/data/medical_advice_prompt_only.jsonl \\
        --layer 15 \\
        --out /root/em_features/results/v2_qwen_l15_sae_arditi_k128_step100000_encoder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
for p in (str(VENDOR_SRC), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sae_day.sae import TopKSAE  # noqa: E402
from experiments.em_features.architectures.txc_bare_multidistance_contrastive_antidead import (  # noqa: E402
    TXCBareMultiDistanceContrastiveAntidead,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--arch", choices=["sae", "han", "tsae", "txc"], required=True)
    p.add_argument("--hookpoint", default=None,
                   help="Override hookpoint (resid_post / resid_mid / ln1_normalized / resid_pre). "
                        "If omitted, falls back to ckpt config['hookpoint'] then resid_post.")
    p.add_argument("--dataset", type=Path, required=True,
                   help="JSONL of prompts (one prompt per line, key 'prompt' or 'text').")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--bad_model", default="andyrdt/Qwen2.5-7B-Instruct_bad-medical")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--n_prompts", type=int, default=1000,
                   help="Number of prompts from the dataset to use (subsample).")
    p.add_argument("--max_ctx", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_prompts(path: Path, n: int) -> list[str]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "prompt" in d:
                out.append(d["prompt"])
            elif "text" in d:
                out.append(d["text"])
            elif "messages" in d:
                # messages = [{"role": "user", "content": "..."}, ...]
                msgs = d["messages"]
                if msgs and isinstance(msgs[0], dict) and "content" in msgs[0]:
                    out.append(msgs[0]["content"])
            if len(out) >= n:
                break
    return out


@torch.no_grad()
def gather_residuals(model, tok, prompts: list[str], layer: int, max_ctx: int,
                     batch_size: int, device: str, hookpoint: str = "resid_post") -> torch.Tensor:
    """Returns (N_tokens, d_model) activations at `layer`/`hookpoint`, concatenated
    over all prompts. System / pad tokens are masked out via attention_mask.

    For ``resid_post`` and ``resid_pre`` we read straight off ``hidden_states``.
    For ``resid_mid`` and ``ln1_normalized`` we register a forward hook via
    HookpointExtractor (matches the streaming buffer used during training)."""
    from experiments.em_features.streaming_buffer import HookpointExtractor
    chunks = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_ctx,
                  return_tensors="pt", add_special_tokens=True).to(device)
        if hookpoint in ("resid_post", "resid_pre"):
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = HookpointExtractor.from_output(hookpoint, layer, out)
        else:
            with HookpointExtractor(model, hookpoint, layer) as ext:
                model(**enc, use_cache=False)
            hs = ext.captured
            if hs is None:
                raise RuntimeError(f"hook for {hookpoint} L{layer} did not capture")
        attn = enc.attention_mask.bool()
        chunks.append(hs[attn].cpu())
    return torch.cat(chunks, dim=0)  # (N_tokens, d_model)


@torch.no_grad()
def encode_sae(sae: TopKSAE, residuals: torch.Tensor, device: str,
               batch_size: int = 4096) -> torch.Tensor:
    """SAE: per-token. Returns (N_tokens, d_sae) post-TopK firings."""
    out = []
    for i in range(0, residuals.shape[0], batch_size):
        x = residuals[i:i + batch_size].to(device).float()
        _, z = sae(x)
        out.append(z.cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_han(model: TXCBareMultiDistanceContrastiveAntidead, residuals: torch.Tensor,
               T: int, device: str, batch_size: int = 1024) -> torch.Tensor:
    """Han: T-window encoder. Build sliding windows of T consecutive tokens and
    encode each. Returns (N_windows, d_sae). N_windows = N_tokens − (T−1)."""
    N = residuals.shape[0]
    if N < T:
        return torch.empty(0, model.d_sae)
    # windows[t] = residuals[t : t+T]; N_windows = N-T+1
    n_w = N - T + 1
    out = []
    for i in range(0, n_w, batch_size):
        idx = i + torch.arange(min(batch_size, n_w - i)).unsqueeze(1) + torch.arange(T).unsqueeze(0)
        w = residuals[idx].to(device).float()  # (B, T, d)
        z = model.encode(w)
        out.append(z.cpu())
    return torch.cat(out, dim=0)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Load probe dataset
    prompts = load_prompts(args.dataset, args.n_prompts)
    print(f"loaded {len(prompts)} prompts from {args.dataset.name}", flush=True)

    # Load shared tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Load SAE/Han ckpt (small, on GPU)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    if args.arch == "sae":
        sae = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"]).to(args.device)
        sae.load_state_dict(ckpt["state_dict"])
        sae.eval()
        T = 1
    elif args.arch == "tsae":
        from experiments.em_features.architectures.tsae_adjacent_contrastive import TSAEAdjacentContrastive
        sae = TSAEAdjacentContrastive(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"],
            contrastive_alpha=cfg.get("contrastive_alpha", 1.0),
            aux_k=cfg.get("aux_k", 512),
            dead_threshold_tokens=cfg.get("dead_threshold_tokens", 640_000),
            auxk_alpha=cfg.get("auxk_alpha", 1.0 / 32.0),
        ).to(args.device)
        sae.load_state_dict(ckpt["state_dict"])
        sae.eval()
        T = 1   # T-SAE encodes per-token at inference time
    elif args.arch == "txc":
        from sae_day.sae import TemporalCrosscoder
        sae = TemporalCrosscoder(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k_total=cfg["k_total"],
        ).to(args.device)
        sae.load_state_dict(ckpt["state_dict"])
        sae.eval()
        T = cfg["T"]
    else:
        sae = TXCBareMultiDistanceContrastiveAntidead(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k=cfg["k"],
            shifts=tuple(cfg.get("shifts", (1, 2))),
            matryoshka_h_size=cfg.get("matryoshka_h_size", cfg["d_sae"] // 5),
            alpha=cfg.get("alpha_contrastive", 1.0),
            aux_k=cfg.get("aux_k", 512),
            dead_threshold_tokens=cfg.get("dead_threshold_tokens", 640_000),
            auxk_alpha=cfg.get("auxk_alpha", 1.0 / 32.0),
        ).to(args.device)
        sae.load_state_dict(ckpt["state_dict"])
        sae.eval()
        T = cfg["T"]

    # Resolve hookpoint: CLI override > ckpt config > resid_post default.
    hookpoint = args.hookpoint or cfg.get("hookpoint", "resid_post")
    print(f"using hookpoint={hookpoint} layer={args.layer}", flush=True)

    # Process M (base) and M_D (bad-medical) sequentially to fit in VRAM.
    print("loading base model...", flush=True)
    base_hf = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map=args.device,
    ).eval()
    print("gathering base residuals...", flush=True)
    base_resid = gather_residuals(base_hf, tok, prompts, args.layer, args.max_ctx, args.batch_size, args.device, hookpoint)
    print(f"  base residuals: {base_resid.shape}", flush=True)
    del base_hf
    torch.cuda.empty_cache()

    print("loading bad-medical model...", flush=True)
    bad_hf = AutoModelForCausalLM.from_pretrained(
        args.bad_model, torch_dtype=torch.float16, device_map=args.device,
    ).eval()
    print("gathering bad-medical residuals...", flush=True)
    bad_resid = gather_residuals(bad_hf, tok, prompts, args.layer, args.max_ctx, args.batch_size, args.device, hookpoint)
    print(f"  bad-medical residuals: {bad_resid.shape}", flush=True)
    del bad_hf
    torch.cuda.empty_cache()

    # Encode through SAE/Han/T-SAE
    print("encoding base activations...", flush=True)
    if args.arch in ("sae", "tsae"):  # per-token encoders
        z_base = encode_sae(sae, base_resid, args.device)
    else:
        z_base = encode_han(sae, base_resid, T, args.device)
    print("encoding bad-medical activations...", flush=True)
    if args.arch in ("sae", "tsae"):  # per-token encoders
        z_bad = encode_sae(sae, bad_resid, args.device)
    else:
        z_bad = encode_han(sae, bad_resid, T, args.device)
    print(f"  z_base: {z_base.shape}  z_bad: {z_bad.shape}", flush=True)

    # Δz̄ per latent
    mean_base = z_base.float().mean(dim=0)  # (d_sae,)
    mean_bad = z_bad.float().mean(dim=0)
    delta = mean_bad - mean_base

    sorted_d, sorted_idx = torch.sort(delta, descending=True)
    top_indices = sorted_idx[:args.top_k].tolist()
    top_vals = sorted_d[:args.top_k].tolist()

    out_json = args.out / "top_200_features.json"
    with out_json.open("w") as f:
        json.dump({
            "method": "encoder_activation_diff",
            "arch": args.arch,
            "layer": args.layer,
            "ckpt": str(args.ckpt),
            "n_prompts": len(prompts),
            "n_tokens_base": int(z_base.shape[0]),
            "n_tokens_bad": int(z_bad.shape[0]),
            "features": [
                {"rank": i + 1, "feature_id": int(top_indices[i]), "delta_z": float(top_vals[i])}
                for i in range(len(top_indices))
            ],
        }, f, indent=2)
    print(f"wrote {out_json}", flush=True)
    print("top-10:")
    for i in range(min(10, len(top_indices))):
        print(f"  {i+1:2d}. feat {top_indices[i]}  Δz̄={top_vals[i]:+.4f}")


if __name__ == "__main__":
    main()
