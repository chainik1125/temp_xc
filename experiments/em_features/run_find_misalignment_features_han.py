"""Stage A finder for the Han H8 champion architecture
(``TXCBareMultiDistanceContrastiveAntidead``). Computes cosine similarity
between each feature's last-position decoder column ``W_dec[:, -1, :]`` and
the layer's diff vector, and writes the top-k features in the schema
consumed by ``frontier_sweep.py --steerer han``.

    uv run python -m experiments.em_features.run_find_misalignment_features_han \\
        --ckpt /root/em_features/checkpoints/qwen_l15_han_champ_10k_step10000.pt \\
        --diff_vectors /root/em_features/results/qwen_l15_sae/01_differences/difference_vectors.pt \\
        --layer 15 \\
        --out /root/em_features/results/qwen_l15_han_champ_10k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.em_features.architectures.txc_bare_multidistance_contrastive_antidead import (  # noqa: E402
    TXCBareMultiDistanceContrastiveAntidead,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--diff_vectors", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    if cfg["layer"] != args.layer:
        raise ValueError(f"ckpt was trained at layer {cfg['layer']}, --layer={args.layer}")

    m = TXCBareMultiDistanceContrastiveAntidead(
        d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k=cfg["k"],
        shifts=tuple(cfg.get("shifts", (1, 2))),
        matryoshka_h_size=cfg.get("matryoshka_h_size", cfg["d_sae"] // 5),
        alpha=cfg.get("alpha_contrastive", 1.0),
        aux_k=cfg.get("aux_k", 512),
        dead_threshold_tokens=cfg.get("dead_threshold_tokens", 640_000),
        auxk_alpha=cfg.get("auxk_alpha", 1.0 / 32.0),
    ).to(args.device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()

    diff_map = torch.load(args.diff_vectors, map_location=args.device)
    if args.layer not in diff_map:
        raise KeyError(f"layer {args.layer} not in diff_vectors; have {sorted(diff_map)}")
    diff_vec = diff_map[args.layer].float()  # (d_in,)

    # W_dec shape (d_sae, T, d_in); take last temporal slot.
    W_last = m.W_dec[:, -1, :].detach().float()  # (d_sae, d_in)
    cos = F.cosine_similarity(W_last, diff_vec.unsqueeze(0).expand_as(W_last), dim=-1)  # (d_sae,)
    sorted_cos, sorted_idx = torch.sort(cos, descending=True)

    top_indices = sorted_idx[: args.top_k].tolist()
    top_sims = sorted_cos[: args.top_k].tolist()

    out_json = args.out / f"top_{args.top_k}_features_layer_{args.layer}.json"
    out_json.write_text(json.dumps({
        "layer": args.layer,
        "crosscoder": "Han_H8",
        "ckpt": str(args.ckpt),
        "ranking": "last_cos",
        "vector_norm": float(diff_vec.norm()),
        "position": cfg["T"] - 1,
        "features": [
            {"rank": i + 1, "feature_id": int(top_indices[i]), "cosine_similarity": float(top_sims[i])}
            for i in range(len(top_indices))
        ],
    }, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
