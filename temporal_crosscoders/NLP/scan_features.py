#!/usr/bin/env python3
"""
scan_features.py — Run TopKFinder + TextContext on one or more sweep
checkpoints and dump per-feature top-K activating windows to JSON.

No LLM calls. Output is the raw "top text windows per feature" table —
the input to any downstream interpretation (explainer, temporal-spread
metric, pred/novel analysis for TFA).

Usage:
    python -m temporal_crosscoders.NLP.scan_features \\
        --arches stacked_sae crosscoder tfa_pos \\
        --subject-model gemma-2-2b-it \\
        --cached-dataset fineweb \\
        --layer-key resid_L25 \\
        --k 50 --T 5 \\
        --sample-chains 1000 \\
        --top-features 200 \\
        --top-k 10 \\
        --ckpt-dir results/nlp_sweep/gemma/ckpts \\
        --out-dir results/nlp_sweep/gemma/scans
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict

from temporal_crosscoders.NLP.autointerp import (
    TextContext,
    TopKFinder,
    load_model,
)
from temporal_crosscoders.NLP.config import cache_dir_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("scan_features")


def _ckpt_path(ckpt_dir: str, arch: str, subject: str, dataset: str,
               layer: str, k: int, seed: int, shuffled: bool) -> str:
    # tfa_pos_pred is a load-time variant of tfa_pos that emits pred_codes
    # instead of novel_codes; the underlying checkpoint is the same file.
    ckpt_arch = "tfa_pos" if arch == "tfa_pos_pred" else arch
    tag = "_shuffled" if shuffled else ""
    return os.path.join(
        ckpt_dir,
        f"{ckpt_arch}__{subject}__{dataset}__{layer}__k{k}__seed{seed}{tag}.pt",
    )


def scan_one(
    arch: str,
    ckpt_path: str,
    subject_model: str,
    cache_dir: str,
    layer_key: str,
    k: int,
    T: int,
    sample_chains: int,
    top_features: int,
    top_k: int,
    device: str,
    text_ctx: TextContext,
) -> dict:
    log.info(f"[{arch}] loading {ckpt_path}")
    model = load_model(
        ckpt_path=ckpt_path, model_type=arch,
        subject_model=subject_model, k=k, T=T,
    )
    finder = TopKFinder(
        model=model,
        model_type=arch,
        layer_key=layer_key,
        cache_dir=cache_dir,
        k=top_k,
        sample_chains=sample_chains,
        chain_batch=64,
        device=device,
    )
    log.info(f"[{arch}] scanning {sample_chains} chains...")
    finder.run()

    # Pick top-N features by total activation mass across their top-K examples
    mass = {
        fi: sum(e.activation for e in exs)
        for fi, exs in finder.results.items()
    }
    selected = sorted(mass, key=mass.get, reverse=True)[:top_features]
    log.info(
        f"[{arch}] {len(finder.results):,} active features; "
        f"writing top {len(selected)}"
    )

    out: dict = {
        "arch": arch,
        "subject_model": subject_model,
        "layer_key": layer_key,
        "k": k, "T": T,
        "sample_chains": sample_chains,
        "top_k_per_feature": top_k,
        "d_sae": finder.d_sae,
        "num_active_features": len(finder.results),
        "features": {},
    }
    for fi in selected:
        exs = finder.results[fi]
        out["features"][fi] = {
            "mass": mass[fi],
            "examples": [
                {
                    "activation": e.activation,
                    "chain_idx": e.chain_idx,
                    "window_start": e.window_start,
                    "text": text_ctx.get_window_text(
                        e.chain_idx, e.window_start, T,
                    ),
                }
                for e in exs
            ],
        }

    # Free GPU before the next arch
    model.to("cpu")
    del model, finder
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arches", nargs="+", required=True,
                   choices=["stacked_sae", "crosscoder", "tfa", "tfa_pos",
                            "tfa_pos_pred"])
    p.add_argument("--subject-model", default="gemma-2-2b-it")
    p.add_argument("--cached-dataset", default="fineweb")
    p.add_argument("--layer-key", default="resid_L25")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffled", action="store_true",
                   help="load the shuffled-training variant of each ckpt")
    p.add_argument("--sample-chains", type=int, default=1000)
    p.add_argument("--top-features", type=int, default=200)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--ckpt-dir", default="results/nlp_sweep/gemma/ckpts")
    p.add_argument("--out-dir", default="results/nlp_sweep/gemma/scans")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cache_dir = cache_dir_for(args.subject_model, args.cached_dataset)
    os.makedirs(args.out_dir, exist_ok=True)

    log.info(f"cache_dir = {cache_dir}")
    log.info(f"loading text context...")
    text_ctx = TextContext(cache_dir, args.subject_model)

    for arch in args.arches:
        ckpt = _ckpt_path(
            args.ckpt_dir, arch, args.subject_model, args.cached_dataset,
            args.layer_key, args.k, args.seed, args.shuffled,
        )
        if not os.path.exists(ckpt):
            log.warning(f"[{arch}] checkpoint not found: {ckpt}")
            continue

        result = scan_one(
            arch=arch, ckpt_path=ckpt,
            subject_model=args.subject_model, cache_dir=cache_dir,
            layer_key=args.layer_key, k=args.k, T=args.T,
            sample_chains=args.sample_chains,
            top_features=args.top_features, top_k=args.top_k,
            device=args.device, text_ctx=text_ctx,
        )
        tag = "_shuffled" if args.shuffled else ""
        out_path = os.path.join(
            args.out_dir,
            f"scan__{arch}__{args.layer_key}__k{args.k}{tag}.json",
        )
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"[{arch}] wrote {out_path}")

    log.info("done.")


if __name__ == "__main__":
    main()
