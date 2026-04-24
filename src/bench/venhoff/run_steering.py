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
    p.add_argument("--max-iters", type=int, default=10)
    p.add_argument("--n-training-examples", type=int, default=256)
    # Default 0 (eval disabled) per web-claude speed-up suggestion #4:
    # --use_activation_perplexity_selection does inline selection for
    # cluster vectors; bias vector doesn't use eval for early-stop.
    # Eval is pure instrumentation cost during the training loop.
    p.add_argument("--n-eval-examples", type=int, default=0)
    p.add_argument("--optim-minibatch-size", type=int, default=4)
    p.add_argument("--lr", default="1e-2")
    p.add_argument("--bias-only", action="store_true",
                   help="Only train the bias vector (idx=-1); skip cluster vectors.")
    p.add_argument("--cluster-indices", type=int, nargs="*", default=None,
                   help="Only train these cluster idxs (plus bias). Default: all 0..n_clusters-1.")
    p.add_argument("--top-k-clusters", type=int, default=None,
                   help="Train only the first K cluster indices (plus bias). "
                        "Complements Venhoff's max-over-10x5-grid metric — most "
                        "clusters don't contribute to the headline Gap Recovery. "
                        "K=5 cuts Phase 2 cost by ~3x with minor quality loss.")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="Parallelize vector training across this many GPUs. "
                        "Each vector subprocess is pinned via CUDA_VISIBLE_DEVICES.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    if args.top_k_clusters is not None and args.cluster_indices is not None:
        raise SystemExit("specify --top-k-clusters OR --cluster-indices, not both")
    if args.top_k_clusters is not None:
        cluster_idxs: tuple[int, ...] | None = tuple(range(args.top_k_clusters))
    elif args.cluster_indices is not None:
        cluster_idxs = tuple(args.cluster_indices)
    else:
        cluster_idxs = None

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
        n_training_examples=args.n_training_examples,
        n_eval_examples=args.n_eval_examples,
        optim_minibatch_size=args.optim_minibatch_size,
        lr=args.lr,
        seed=args.seed,
        bias_only=args.bias_only,
        cluster_indices=cluster_idxs,
        arch=args.arch,
    )
    train_all_vectors(
        venhoff_root=args.venhoff_root,
        cfg=cfg,
        paths=paths,
        force=args.force,
        num_gpus=args.num_gpus,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
