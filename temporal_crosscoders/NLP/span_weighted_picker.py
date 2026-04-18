"""Span-weighted feature picker — apples-to-apples cross-arch.

For each arch's span_all__<arch>__<layer>__k<k>.json, rank features
by `(1 - conc) * mass` ("span-weighted mass"). Emit top-N per arch
plus a content-bearing filter (via exemplar scan for those features).

For features already in scan__<arch>__<layer>__k<k>.json (the top-300
by mass) we reuse their exemplars. For features NOT in that scan, we
need a targeted scan — produced by `scan_specific_features.py`.

Outputs span_weighted_high_span__<layer>__k<k>.json with per-arch
top-N.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ARCHS = ["stacked_sae", "crosscoder", "tfa_pos", "tfa_pos_pred"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", default="results/nlp_sweep/gemma/scans")
    ap.add_argument("--layer-key", default="resid_L25")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scan_dir = Path(args.scan_dir)
    per_arch: dict[str, dict] = {}
    missing_feats: dict[str, list[int]] = {}

    for arch in ARCHS:
        span = json.load(open(
            scan_dir / f"span_all__{arch}__{args.layer_key}__k{args.k}.json"
        ))
        mass_scan = json.load(open(
            scan_dir / f"scan__{arch}__{args.layer_key}__k{args.k}.json"
        ))
        feats = span["features"]
        # Span-weighted ranking: reward both mass AND spread
        ranked = sorted(
            feats.items(),
            key=lambda kv: (1.0 - kv[1]["concentration"]) * kv[1]["mass"],
            reverse=True,
        )
        # Classic mass ranking for reference
        mass_ranked = sorted(
            feats.items(),
            key=lambda kv: kv[1]["mass"],
            reverse=True,
        )

        top_span = ranked[: args.top_n]
        top_mass = mass_ranked[: args.top_n]

        already_scanned = set(mass_scan["features"].keys())
        need_scan = [
            int(fid) for fid, _ in top_span if fid not in already_scanned
        ]
        missing_feats[arch] = need_scan

        per_arch[arch] = {
            "top_span_weighted": [
                {
                    "feat_idx": int(fid),
                    "concentration": rec["concentration"],
                    "mass": rec["mass"],
                    "span_weighted": (1 - rec["concentration"]) * rec["mass"],
                    "peak_position": rec["peak_position"],
                    "has_exemplars": fid in already_scanned,
                }
                for fid, rec in top_span
            ],
            "top_mass_reference": [
                {
                    "feat_idx": int(fid),
                    "concentration": rec["concentration"],
                    "mass": rec["mass"],
                    "peak_position": rec["peak_position"],
                }
                for fid, rec in top_mass[:15]
            ],
            "n_need_targeted_scan": len(need_scan),
        }

    out = {
        "layer_key": args.layer_key,
        "k": args.k,
        "top_n": args.top_n,
        "per_arch": per_arch,
        "feats_needing_targeted_scan": missing_feats,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")

    # Print summary
    print()
    print(
        f"{'arch':<14} {'top1_feat':>10} {'top1_conc':>10}"
        f" {'top1_mass':>12} {'top1_span':>12} {'need_scan':>10}"
    )
    for arch in ARCHS:
        p = per_arch[arch]["top_span_weighted"][0]
        print(
            f"{arch:<14} {p['feat_idx']:>10d} {p['concentration']:>10.3f}"
            f" {p['mass']:>12.3f} {p['span_weighted']:>12.3f}"
            f" {per_arch[arch]['n_need_targeted_scan']:>10d}"
        )


if __name__ == "__main__":
    main()
