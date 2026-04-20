"""CLI: run hybrid-model inference on MATH500 via Venhoff's hybrid_token.py.

Thin wrapper around `hybrid.run_hybrid`. Invoked by the runpod launcher
in MODE=hybrid, once per arch.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.bench.venhoff.hybrid import DEFAULT_COEFFICIENTS, DEFAULT_TOKEN_WINDOWS, HybridConfig, run_hybrid
from src.bench.venhoff.paths import ArtifactPaths, RunIdentity

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
    p.add_argument("--max-new-tokens", type=int, default=2000)
    p.add_argument("--max-thinking-tokens", type=int, default=2000)
    p.add_argument("--coefficients", type=float, nargs="+", default=list(DEFAULT_COEFFICIENTS))
    p.add_argument("--token-windows", type=int, nargs="+", default=list(DEFAULT_TOKEN_WINDOWS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    paths = ArtifactPaths(
        root=args.root,
        identity=RunIdentity(
            model=args.model, dataset=args.dataset, dataset_split=args.split,
            n_traces=args.n_traces, layer=args.layer, seed=args.seed,
        ),
    )
    cfg = HybridConfig(
        dataset=args.dataset,
        base_model=args.base_model,
        thinking_model=args.thinking_model,
        steering_layer=args.steering_layer,
        sae_layer=args.sae_layer,
        n_clusters=args.n_clusters,
        max_new_tokens=args.max_new_tokens,
        max_thinking_tokens=args.max_thinking_tokens,
        coefficients=tuple(args.coefficients),
        token_windows=tuple(args.token_windows),
        seed=args.seed,
    )
    run_hybrid(
        venhoff_root=args.venhoff_root,
        cfg=cfg,
        paths=paths,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
