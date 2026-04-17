"""Phase 1 sanity check: label held-out features ranked 51-100 by mass.

For each arch, rank by mass, take the 51-100 slice, and produce a subset
scan JSON suitable for explain_features. Shell out to explain_features.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", default="results/nlp_sweep/gemma/scans")
    ap.add_argument("--layer-key", default="resid_L25")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--arches", nargs="+",
                    default=["stacked_sae", "crosscoder", "tfa_pos", "tfa_pos_pred"])
    ap.add_argument("--rank-start", type=int, default=50)
    ap.add_argument("--rank-end", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()

    scan_dir = Path(args.scan_dir)
    for arch in args.arches:
        src = scan_dir / f"scan__{arch}__{args.layer_key}__k{args.k}.json"
        if not src.exists():
            print(f"skip {arch}: no scan at {src}")
            continue
        d = json.load(open(src))
        ranked = sorted(
            d["features"].items(),
            key=lambda kv: kv[1]["mass"],
            reverse=True,
        )
        slice_ = ranked[args.rank_start:args.rank_end]
        subset = {
            **{k: v for k, v in d.items() if k != "features"},
            "features": {fid: feat for fid, feat in slice_},
        }
        sub_path = scan_dir / (
            f"scan__{arch}__{args.layer_key}__k{args.k}"
            f"__heldout{args.rank_start}-{args.rank_end}.json"
        )
        out_path = scan_dir / (
            f"labels__{arch}__{args.layer_key}__k{args.k}"
            f"__heldout{args.rank_start}-{args.rank_end}.json"
        )
        json.dump(subset, open(sub_path, "w"), indent=2)
        print(f"[{arch}] wrote subset ({len(slice_)} feats) -> {sub_path}")

        # explain_features reads the full scan; use top-features == len(slice)
        cmd = [
            sys.executable, "-m", "temporal_crosscoders.NLP.explain_features",
            "--scan", str(sub_path),
            "--out", str(out_path),
            "--top-features", str(len(slice_)),
            "--concurrency", str(args.concurrency),
        ]
        print("  running:", " ".join(cmd))
        r = subprocess.run(cmd, env={**os.environ})
        if r.returncode != 0:
            print(f"  explain_features failed for {arch}", file=sys.stderr)
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
