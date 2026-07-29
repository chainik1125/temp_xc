"""Rebuild a combined corrective summary from existing per-layer JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.correlation_audit.corrective import write_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    args = parser.parse_args()
    results = [
        json.loads(
            (args.output_dir / f"layer_{layer}_corrective.json").read_text()
        )
        for layer in args.layers
    ]
    write_summary(results, args.output_dir / "summary.md")
    print(
        json.dumps(
            {
                "status": "ok",
                "layers": args.layers,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
