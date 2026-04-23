"""Stage A for the TXC — decompose the em-features diff vector onto the
trained TXC decoder, emit top-200 features in the same JSON schema that
em-features' sae_decomposition.py produces so the downstream sweep consumes
both uniformly.

Run (after em-features pipeline has produced 01_differences/difference_vectors.pt):

    uv run python -m experiments.em_features.run_find_misalignment_features \
        --ckpt experiments/em_features/checkpoints/qwen_l15_txc_t5_k128.pt \
        --diff_vectors ~/Documents/Research/em_features/open-source-em-features/results/qwen_l15_sae/01_differences/difference_vectors.pt \
        --layer 15 \
        --out experiments/em_features/results/qwen_l15_txc
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

from sae_day.sae import TemporalCrosscoder  # noqa: E402

from experiments.em_features.crosscoder_adapter import decompose_diff_on_txc  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--diff_vectors", type=Path, required=True,
                   help="Path to em-features' difference_vectors.pt {layer: (d_model,)}.")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--position", default="last",
                   help="Window position to use for decoder direction (last|first|int).")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    ccfg = ckpt["config"]
    txc = TemporalCrosscoder(
        d_in=ccfg["d_in"], d_sae=ccfg["d_sae"], T=ccfg["T"], k_total=ccfg["k_total"],
    )
    txc.load_state_dict(ckpt["state_dict"])
    txc.eval().to(args.device)

    if ccfg["layer"] != args.layer:
        raise ValueError(
            f"ckpt was trained at layer {ccfg['layer']}, but --layer={args.layer}"
        )

    diff_map = torch.load(args.diff_vectors, map_location=args.device)
    if args.layer not in diff_map:
        raise KeyError(f"layer {args.layer} not in diff_vectors; have {sorted(diff_map)}")
    diff_vec = diff_map[args.layer]

    position = args.position
    try:
        position = int(position)
    except ValueError:
        pass

    decomp = decompose_diff_on_txc(diff_vec, txc, top_k=args.top_k, position=position)

    # Write files shaped like em-features/sae_decomposition outputs.
    top_features_path = args.out / f"top_{args.top_k}_features_layer_{args.layer}.json"
    with top_features_path.open("w") as f:
        json.dump({
            "layer": args.layer,
            "crosscoder": "TXC",
            "ckpt": str(args.ckpt),
            "vector_norm": float(diff_vec.float().norm()),
            "position": ccfg["T"] - 1 if position == "last" else position,
            "features": [
                {
                    "rank": i + 1,
                    "feature_id": decomp["top_features"]["indices"][i],
                    "cosine_similarity": decomp["top_features"]["similarities"][i],
                }
                for i in range(len(decomp["top_features"]["indices"]))
            ],
        }, f, indent=2)
    print(f"wrote {top_features_path}")

    summary_path = args.out / f"txc_decomposition_layer_{args.layer}.json"
    with summary_path.open("w") as f:
        json.dump({
            "layer": args.layer,
            "crosscoder": "TXC",
            "ckpt": str(args.ckpt),
            "vector_norm": float(diff_vec.float().norm()),
            "top_features": decomp["top_features"],
            "bottom_features": decomp["bottom_features"],
            "decomposition_summary": {
                "max_similarity": float(decomp["sorted_similarities"][0]),
                "min_similarity": float(decomp["sorted_similarities"][-1]),
                "mean_abs_similarity": float(decomp["similarities"].abs().mean()),
                "std_similarity": float(decomp["similarities"].std()),
            },
        }, f, indent=2)
    print(f"wrote {summary_path}")

    print("\ntop-10 preview:")
    for i in range(min(10, len(decomp["top_features"]["indices"]))):
        idx = decomp["top_features"]["indices"][i]
        sim = decomp["top_features"]["similarities"][i]
        print(f"  {i+1:2d}. feature {idx}  cos={sim:+.4f}")


if __name__ == "__main__":
    main()
