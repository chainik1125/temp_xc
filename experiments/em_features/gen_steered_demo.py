"""Generate steered + unsteered completions for a single feature across an
α-grid, on a fixed set of 5 prompts with fixed seeds. Output JSON is consumed
by the static HTML dashboard at docs/dmitry/results/em_features/steering_demo/.

Same generation primitive as the Wang procedure — just exposes the per-rollout
text instead of judging-and-discarding it.

    uv run python -m experiments.em_features.gen_steered_demo \\
        --ckpt /root/em_features/checkpoints/qwen_l15_txc_paper_k100bt_d16k_step30000.pt \\
        --arch txc \\
        --feature_id 4563 --layer 15 \\
        --alphas '-10,-8,-6,-4,-2,0,2,4,6,8,10' \\
        --n_prompts 5 --max_new_tokens 256 \\
        --out /root/em_features/results/steering_demo_feat4563.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
EM_FEATURES = Path("/root/em_features")
for p in (str(REPO_ROOT), str(EM_FEATURES)):
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--arch", choices=["sae", "han", "tsae", "windowed_tsae", "txc"], required=True)
    p.add_argument("--feature_id", type=int, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alphas", type=str, default="-10,-8,-6,-4,-2,0,2,4,6,8,10",
                   help="Comma-separated α values. Order is preserved.")
    p.add_argument("--n_prompts", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42,
                   help="Per-prompt seed; fixed RNG across α makes outputs comparable.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def get_steering_direction(arch: str, ckpt_path: Path, feature_id: int, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if arch == "sae":
        from sae_day.sae import TopKSAE
        sae = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"]).to(device)
        sae.load_state_dict(ckpt["state_dict"])
        return sae.W_dec[feature_id].detach().clone()
    if arch == "txc":
        from sae_day.sae import TemporalCrosscoder
        m = TemporalCrosscoder(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k_total=cfg["k_total"],
        ).to(device)
        m.load_state_dict(ckpt["state_dict"])
        return m.W_dec[-1, feature_id, :].detach().clone()
    if arch == "tsae":
        from experiments.em_features.architectures.tsae_adjacent_contrastive import TSAEAdjacentContrastive
        sae = TSAEAdjacentContrastive(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"],
            contrastive_alpha=cfg.get("contrastive_alpha", 1.0),
            batch_topk=cfg.get("batch_topk", False),
        ).to(device)
        sae.load_state_dict(ckpt["state_dict"])
        return sae.W_dec[feature_id].detach().clone()
    if arch == "windowed_tsae":
        from experiments.em_features.architectures.windowed_tsae import WindowedTSAE
        m = WindowedTSAE(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k=cfg["k"],
            contrastive_alpha=cfg.get("contrastive_alpha", 0.1),
            n_temporal_features=cfg.get("n_temporal_features", None),
            mix_positions=cfg.get("mix_positions", False),
        ).to(device)
        m.load_state_dict(ckpt["state_dict"])
        return m.W_dec[-1, feature_id, :].detach().clone()
    raise ValueError(arch)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from open_source_em_features.pipeline.longform_steering import (
        generate_longform_completions, load_em_dataset,
    )
    from open_source_em_features.utils.model_loading import load_model_and_tokenizer

    BAD_MEDICAL = "andyrdt/Qwen2.5-7B-Instruct_bad-medical"

    direction = get_steering_direction(args.arch, args.ckpt, args.feature_id, args.device)
    print(f"steering direction shape={tuple(direction.shape)} norm={float(direction.norm()):.4f}",
          flush=True)

    print("loading bad-medical Qwen...", flush=True)
    model, tokenizer = load_model_and_tokenizer(BAD_MEDICAL)

    print("loading EM eval prompts...", flush=True)
    em = load_em_dataset()
    questions = [d["messages"][0]["content"] for d in em][:args.n_prompts]

    alphas = [float(a) for a in args.alphas.split(",")]
    print(f"sweeping {len(alphas)} alphas × {len(questions)} prompts at fixed seed={args.seed}",
          flush=True)

    out = {
        "meta": {
            "arch": args.arch,
            "ckpt": str(args.ckpt),
            "feature_id": args.feature_id,
            "layer": args.layer,
            "alphas": alphas,
            "n_prompts": args.n_prompts,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "subject_model": BAD_MEDICAL,
        },
        "prompts": [],
    }

    for prompt_idx, q in enumerate(questions):
        out["prompts"].append({
            "prompt_idx": prompt_idx,
            "question": q,
            "completions_by_alpha": {},
        })

    for alpha in alphas:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        completions = generate_longform_completions(
            model=model, tokenizer=tokenizer, questions=questions,
            steering_direction=direction, magnitude=float(alpha),
            layer_idx=int(args.layer), n_generations=1,
            max_new_tokens=int(args.max_new_tokens), temperature=1.0,
        )
        # The helper returns a flat list aligned with input questions order
        for i, comp in enumerate(completions):
            text = comp if isinstance(comp, str) else comp.get("completion", comp.get("answer", str(comp)))
            out["prompts"][i]["completions_by_alpha"][f"{alpha:+.2f}"] = text
        print(f"  α={alpha:+.2f} → {len(completions)} completions", flush=True)

    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
