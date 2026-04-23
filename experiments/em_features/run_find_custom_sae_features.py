"""Stage A for a custom TopK SAE: cosine decomposition of a diff vector onto
the trained SAE decoder rows. Outputs top_200_features.json in the em-features
schema so frontier_sweep.py consumes it.

    uv run python -m experiments.em_features.run_find_custom_sae_features \
        --ckpt .../qwen_l15_sae_resid_mid_k128.pt \
        --diff_vectors .../custom_diffs_resid_mid_L15.pt \
        --hookpoint resid_mid --layer 15 \
        --out .../qwen_l15_sae_resid_mid

For resid_post / resid_pre the diff is already in em-features'
difference_vectors.pt ({layer: tensor}). Use the original file for those;
for resid_mid / ln1_normalized use compute_custom_diffs.py first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

from sae_day.sae import TopKSAE  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--diff_vectors", type=Path, required=True,
                   help="em-features difference_vectors.pt {layer: (d,)} OR "
                        "compute_custom_diffs.py output {diff, hookpoint, layer}.")
    p.add_argument("--hookpoint", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _load_diff_vec(path: Path, hookpoint: str, layer: int) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "diff" in obj:
        if obj.get("layer") != layer or obj.get("hookpoint") != hookpoint:
            raise ValueError(f"ckpt hookpoint/layer mismatch: file={obj.get('hookpoint')}"
                             f"/{obj.get('layer')} vs expected {hookpoint}/{layer}")
        return obj["diff"].float()
    # em-features difference_vectors.pt style: {layer_idx: tensor}
    if isinstance(obj, dict):
        # For resid_post use layer; for resid_pre use layer-1 (same as prev block's resid_post).
        if hookpoint == "resid_post":
            idx = layer
        elif hookpoint == "resid_pre":
            idx = layer - 1
        else:
            raise ValueError(f"{hookpoint} diffs must come from compute_custom_diffs.py")
        if idx not in obj:
            raise KeyError(f"layer {idx} missing; have {sorted(obj)}")
        return obj[idx].float()
    raise ValueError(f"unrecognized diff_vectors file format: {type(obj)}")


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    ccfg = ckpt["config"]
    if ccfg.get("hookpoint") != args.hookpoint or ccfg.get("layer") != args.layer:
        raise ValueError(f"ckpt hookpoint/layer mismatch: ckpt={ccfg.get('hookpoint')}"
                         f"/{ccfg.get('layer')} vs expected {args.hookpoint}/{args.layer}")

    sae = TopKSAE(d_in=ccfg["d_in"], d_sae=ccfg["d_sae"], k=ccfg["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval().to(args.device)

    diff = _load_diff_vec(args.diff_vectors, args.hookpoint, args.layer).to(args.device)
    dec = sae.W_dec.data.float()  # (d_sae, d_in) per sae_day convention
    # Robust to decoder layout: if rows are features, fine; else transpose.
    if dec.shape[1] != diff.shape[0]:
        if dec.shape[0] == diff.shape[0]:
            dec = dec.T
        else:
            raise ValueError(f"decoder shape {dec.shape} incompatible with diff {diff.shape}")
    dec_u = dec / (dec.norm(dim=-1, keepdim=True) + 1e-8)
    diff_u = diff / (diff.norm() + 1e-8)
    sims = dec_u @ diff_u  # (d_sae,)

    sorted_sims, sorted_idx = torch.sort(sims, descending=True)
    k = min(args.top_k, sims.numel() // 2)
    top_indices = sorted_idx[:k].cpu().tolist()
    top_sims = sorted_sims[:k].cpu().tolist()

    out_json = args.out / "top_200_features.json"
    with out_json.open("w") as f:
        json.dump({
            "sae_kind": "custom_TopKSAE",
            "hookpoint": args.hookpoint,
            "layer": args.layer,
            "ckpt": str(args.ckpt),
            "vector_norm": float(diff.norm()),
            "features": [
                {"rank": i + 1, "feature_id": top_indices[i], "cosine_similarity": top_sims[i]}
                for i in range(k)
            ],
        }, f, indent=2)
    print(f"wrote {out_json}")
    print("top-10 preview:")
    for i in range(min(10, k)):
        print(f"  {i+1:2d}. feature {top_indices[i]}  cos={top_sims[i]:+.4f}")


if __name__ == "__main__":
    main()
