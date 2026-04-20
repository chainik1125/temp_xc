"""CLI: compute Gap Recovery from Venhoff hybrid results.

Thin wrapper around `grade.compute_gap_recovery`. Writes a summary JSON
with base/thinking/per-cell accuracies and the best-cell Gap Recovery
so it's comparable directly with Venhoff's Table 2 (3.5% baseline on
the Llama-8B MATH500 cell).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from src.bench.venhoff.grade import compute_gap_recovery
from src.bench.venhoff.hybrid import DEFAULT_COEFFICIENTS, DEFAULT_TOKEN_WINDOWS

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("venhoff.run_grade")


def _expected_results_dir(
    venhoff_root: Path,
    base_model: str,
    thinking_model: str,
    dataset: str,
    steering_layer: int,
    sae_layer: int,
    n_clusters: int,
) -> Path:
    base_short = base_model.split("/")[-1]
    think_short = thinking_model.split("/")[-1]
    return (
        venhoff_root / "hybrid" / "results" / dataset
        / f"{base_short}_{think_short}_L{steering_layer}_SAEL{sae_layer}_n{n_clusters}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("results/venhoff_eval"))
    p.add_argument("--arch", required=True, choices=["sae", "tempxc", "mlc"])
    p.add_argument("--venhoff-root", type=Path, default=Path("vendor/thinking-llms-interp"))
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--thinking-model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--dataset", default="math500")
    p.add_argument("--steering-layer", type=int, default=12)
    p.add_argument("--sae-layer", type=int, default=6)
    p.add_argument("--n-clusters", type=int, default=15)
    p.add_argument("--thinking-jsonl", type=Path, default=None,
                   help="Per-problem thinking-model outputs JSONL. Defaults to Venhoff's canonical path.")
    p.add_argument("--base-jsonl", type=Path, default=None,
                   help="Optional base-model per-problem JSONL. If omitted, uses (c=0,w=0) cell.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output summary JSON. Defaults to results/{root}/grades/{arch}_{dataset}.json")
    args = p.parse_args(argv)

    results_dir = _expected_results_dir(
        venhoff_root=args.venhoff_root,
        base_model=args.base_model,
        thinking_model=args.thinking_model,
        dataset=args.dataset,
        steering_layer=args.steering_layer,
        sae_layer=args.sae_layer,
        n_clusters=args.n_clusters,
    )
    if not results_dir.exists():
        log.error("[error] hybrid_results_missing | expected=%s", results_dir)
        return 2

    thinking_jsonl = args.thinking_jsonl or (
        args.venhoff_root / "generate-responses" / "results" / "vars"
        / f"responses_{args.thinking_model.split('/')[-1].lower()}.json"
    )
    if not thinking_jsonl.exists():
        log.error("[error] thinking_outputs_missing | expected=%s", thinking_jsonl)
        return 2

    result = compute_gap_recovery(
        results_dir=results_dir,
        thinking_jsonl=thinking_jsonl,
        base_jsonl=args.base_jsonl,
        coefficients=DEFAULT_COEFFICIENTS,
        token_windows=DEFAULT_TOKEN_WINDOWS,
    )

    out_path = args.out or (args.root / "grades" / f"{args.arch}_{args.dataset}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": args.arch,
        "dataset": args.dataset,
        "base_model": args.base_model,
        "thinking_model": args.thinking_model,
        "base_accuracy": result.base_accuracy,
        "thinking_accuracy": result.thinking_accuracy,
        "best_cell": asdict(result.best_cell),
        "best_gap_recovery": result.best_gap_recovery,
        "per_cell": [asdict(c) for c in result.per_cell],
        "venhoff_baseline_gap_recovery": 0.035,  # their 3.5% on Llama-8B MATH500
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(
        "[result] arch=%s | dataset=%s | gap_recovery=%.4f (venhoff_baseline=0.035) | out=%s",
        args.arch, args.dataset, result.best_gap_recovery, out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
