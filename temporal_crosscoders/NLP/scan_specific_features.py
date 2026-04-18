"""Scan specific feature IDs — targeted version of scan_features.py.

Given an arch + layer + list of feat IDs, run TopKFinder and save only
those features' top-K exemplars into a subset scan JSON. Lets us pick
features by span-weighted ranking (not just mass) and still get the
text-context exemplars the autointerp pipeline expects.

Output JSON is scan-format compatible (drop-in replacement subset).
"""
from __future__ import annotations

import argparse
import json
import logging
import os

from temporal_crosscoders.NLP.autointerp import (
    TextContext,
    TopKFinder,
    load_model,
)
from temporal_crosscoders.NLP.config import cache_dir_for
from temporal_crosscoders.NLP.scan_features import _ckpt_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("scan_specific")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["stacked_sae", "crosscoder", "tfa_pos", "tfa_pos_pred"])
    ap.add_argument("--subject-model", default="gemma-2-2b-it")
    ap.add_argument("--cached-dataset", default="fineweb")
    ap.add_argument("--layer-key", default="resid_L25")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-chains", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--feats", nargs="+", type=int, required=True,
                    help="feature IDs to scan")
    ap.add_argument("--ckpt-dir", default="results/nlp_sweep/gemma/ckpts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cache_dir = cache_dir_for(args.subject_model, args.cached_dataset)
    text_ctx = TextContext(cache_dir, args.subject_model)

    ckpt = _ckpt_path(
        args.ckpt_dir, args.arch, args.subject_model, args.cached_dataset,
        args.layer_key, args.k, args.seed, False,
    )
    log.info(f"[{args.arch}] loading {ckpt}")
    model = load_model(
        ckpt_path=ckpt, model_type=args.arch,
        subject_model=args.subject_model, k=args.k, T=args.T,
    )
    finder = TopKFinder(
        model=model, model_type=args.arch, layer_key=args.layer_key,
        cache_dir=cache_dir, k=args.top_k,
        sample_chains=args.sample_chains, chain_batch=64,
        device=args.device,
    )
    log.info(f"[{args.arch}] scanning {args.sample_chains} chains...")
    finder.run()

    wanted = set(args.feats)
    found = [fi for fi in wanted if fi in finder.results]
    missing = [fi for fi in wanted if fi not in finder.results]
    log.info(f"[{args.arch}] requested {len(wanted)} feats;"
             f" found {len(found)} active, {len(missing)} dead in scan")

    out = {
        "arch": args.arch,
        "subject_model": args.subject_model,
        "layer_key": args.layer_key,
        "k": args.k, "T": args.T,
        "sample_chains": args.sample_chains,
        "top_k_per_feature": args.top_k,
        "d_sae": finder.d_sae,
        "num_active_features": len(finder.results),
        "requested_feats": sorted(wanted),
        "dead_feats": sorted(missing),
        "features": {},
    }
    for fi in found:
        exs = finder.results[fi]
        out["features"][str(fi)] = {
            "mass": sum(e.activation for e in exs),
            "examples": [
                {
                    "activation": e.activation,
                    "chain_idx": e.chain_idx,
                    "window_start": e.window_start,
                    "text": text_ctx.get_window_text(
                        e.chain_idx, e.window_start, args.T,
                    ),
                }
                for e in exs
            ],
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    log.info(f"[{args.arch}] wrote {args.out}")


if __name__ == "__main__":
    main()
