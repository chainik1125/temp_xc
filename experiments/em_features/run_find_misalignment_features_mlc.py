"""Stage A for the MLC — decompose the em-features per-layer diff vectors
onto the MLC decoder rows (summed cosine across layers), emit top-200
features in the schema consumed by feature_ablation/frontier_sweep.py.

    uv run python -m experiments.em_features.run_find_misalignment_features_mlc \
        --ckpt experiments/em_features/checkpoints/qwen_mlc_l11-13-15-17-19_k128.pt \
        --diff_vectors /root/em_features/results/qwen_l15_sae/01_differences/difference_vectors.pt \
        --out experiments/em_features/results/qwen_mlc
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

from sae_day.sae import MultiLayerCrosscoder  # noqa: E402

from experiments.em_features.crosscoder_adapter import decompose_diff_on_mlc  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--diff_vectors", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    ccfg = ckpt["config"]
    mlc = MultiLayerCrosscoder(
        d_in=ccfg["d_in"], d_sae=ccfg["d_sae"], L=ccfg["L"], k_total=ccfg["k_total"],
    )
    mlc.load_state_dict(ckpt["state_dict"])
    mlc.eval().to(args.device)

    layers = ccfg["layers"]
    diff_map = torch.load(args.diff_vectors, map_location=args.device)
    missing = [L for L in layers if L not in diff_map]
    if missing:
        raise KeyError(f"diff_vectors missing layers {missing}; have {sorted(diff_map)}")

    decomp = decompose_diff_on_mlc(diff_map, mlc, layers, top_k=args.top_k)

    top_features_path = args.out / f"top_{args.top_k}_features_mlc.json"
    with top_features_path.open("w") as f:
        json.dump({
            "crosscoder": "MLC",
            "ckpt": str(args.ckpt),
            "layers": layers,
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

    summary_path = args.out / "mlc_decomposition.json"
    with summary_path.open("w") as f:
        json.dump({
            "crosscoder": "MLC",
            "ckpt": str(args.ckpt),
            "layers": layers,
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
        print(f"  {i+1:2d}. feature {idx}  sum-cos={sim:+.4f}")


if __name__ == "__main__":
    main()
