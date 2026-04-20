"""CLI: export our trained ckpt into Venhoff's expected SAE format.

Usage (invoked by the runpod launcher in MODE=hybrid):

    python -m src.bench.venhoff.export_venhoff_ckpt \\
        --root results/venhoff_eval --model deepseek-r1-distill-llama-8b \\
        --dataset math500 --n-traces 500 --layer 6 --seed 42 \\
        --arch sae --cluster-size 15 --path path1 \\
        --venhoff-root vendor/thinking-llms-interp \\
        --thinking-model-id deepseek-r1-distill-llama-8b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.bench.venhoff.paths import ArtifactPaths, RunIdentity
from src.bench.venhoff.venhoff_format import export, venhoff_sae_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("results/venhoff_eval"))
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n-traces", type=int, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arch", required=True, choices=["sae", "tempxc", "mlc"])
    p.add_argument("--cluster-size", type=int, required=True)
    p.add_argument("--path", required=True, choices=["path1", "path3", "path_mlc"])
    p.add_argument("--venhoff-root", type=Path, default=Path("vendor/thinking-llms-interp"))
    p.add_argument(
        "--thinking-model-id",
        required=True,
        help="Short id Venhoff's load_sae() uses in the ckpt filename (basename of HF path, lowercase).",
    )
    args = p.parse_args(argv)

    paths = ArtifactPaths(
        root=args.root,
        identity=RunIdentity(
            model=args.model, dataset=args.dataset, dataset_split=args.split,
            n_traces=args.n_traces, layer=args.layer, seed=args.seed,
        ),
    )
    our_ckpt = paths.ckpt(args.arch, args.cluster_size, args.path)
    our_mean = paths.activation_mean_pkl(args.path)
    out = venhoff_sae_path(
        venhoff_root=args.venhoff_root,
        model_id=args.thinking_model_id,
        layer=args.layer,
        n_clusters=args.cluster_size,
    )
    export(
        arch=args.arch,
        our_ckpt=our_ckpt,
        our_mean_pkl=our_mean,
        out_path=out,
        model_id=args.thinking_model_id,
        layer=args.layer,
        n_clusters=args.cluster_size,
        n_examples=args.n_traces,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
