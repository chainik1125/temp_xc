"""Concatenate two sweep_results.json files into one.

Useful when adding a new arch (e.g. regular_sae_kT) without rerunning the
existing arches: run the new arch into a separate output_dir, then merge.

Usage:
    uv run python scripts/merge_results.py \\
        results/three_arch_sweep/sweep_results.json \\
        results/regular_sae_kT/sweep_results.json \\
        --output results/three_arch_sweep_with_kT/sweep_results.json
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="One or more sweep_results.json files to concatenate")
    parser.add_argument("--output", required=True, help="Output path for the merged JSON")
    parser.add_argument(
        "--dedupe", action="store_true",
        help="Drop duplicate (model, rho, k, T, seed) rows, keeping the last occurrence.",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.inputs:
        with open(path) as f:
            rows.extend(json.load(f))
        print(f"loaded {len(rows)} total rows after {path}")

    if args.dedupe:
        seen: dict[tuple, dict] = {}
        for r in rows:
            key = (r["model"], r.get("rho"), r["k"], r["T"], r.get("seed"))
            seen[key] = r
        rows = list(seen.values())
        print(f"after dedupe: {len(rows)} rows")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
