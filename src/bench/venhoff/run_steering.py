"""CLI: train all 16 steering vectors (bias + 15 clusters) for one arch.

Thin wrapper around `steering.train_all_vectors`. Invoked by the runpod
launcher in MODE=hybrid, once per arch.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.bench.venhoff.paths import ArtifactPaths, RunIdentity
from src.bench.venhoff.steering import SteeringConfig, train_all_vectors

logging.basicConfig(level=logging.INFO, format="%(message)s")


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
    p.add_argument("--venhoff-root", type=Path, default=Path("vendor/thinking-llms-interp"))
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--thinking-model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--steering-layer", type=int, default=12)
    p.add_argument("--sae-layer", type=int, default=6)
    p.add_argument("--n-clusters", type=int, default=15)
    p.add_argument("--max-iters", type=int, default=50)
    p.add_argument("--lr", default="1e-2")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    paths = ArtifactPaths(
        root=args.root,
        identity=RunIdentity(
            model=args.model, dataset=args.dataset, dataset_split=args.split,
            n_traces=args.n_traces, layer=args.layer, seed=args.seed,
        ),
    )
    cfg = SteeringConfig(
        base_model=args.base_model,
        thinking_model=args.thinking_model,
        steering_layer=args.steering_layer,
        sae_layer=args.sae_layer,
        n_clusters=args.n_clusters,
        max_iters=args.max_iters,
        lr=args.lr,
        seed=args.seed,
    )
    train_all_vectors(
        venhoff_root=args.venhoff_root,
        cfg=cfg,
        paths=paths,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
