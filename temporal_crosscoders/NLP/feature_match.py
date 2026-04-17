#!/usr/bin/env python3
"""
feature_match.py — Pairwise feature matching between two sweep archs via
decoder cosine similarity.

For each feature in arch A, find the best-match feature in arch B. Output
histogram of best-match cosine similarities → how "shared" are the feature
libraries?

Decoder convention (averaged across T positions where applicable):
  stacked_sae   : mean of saes[t].W_dec for t in 0..T-1  -> (d_in, d_sae)
  crosscoder    : mean of W_dec along T dim              -> (d_in, d_sae)
  tfa_pos       : D.T                                    -> (d_in, d_sae)

Output has both directions (A→B best match, B→A best match) and the
similarity matrix summary stats.

Usage:
    python -m temporal_crosscoders.NLP.feature_match \\
        --archs stacked_sae crosscoder tfa_pos \\
        --ckpts results/nlp_sweep/gemma/ckpts/stacked_sae__gemma-2-2b-it__fineweb__resid_L25__k50__seed42.pt \\
                results/nlp_sweep/gemma/ckpts/crosscoder__gemma-2-2b-it__fineweb__resid_L25__k50__seed42.pt \\
                results/nlp_sweep/gemma/ckpts/tfa_pos__gemma-2-2b-it__fineweb__resid_L25__k50__seed42.pt \\
        --out results/nlp_sweep/gemma/scans/feature_match__resid_L25__k50.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.bench.model_registry import get_model_config
from temporal_crosscoders.NLP.autointerp import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("feat_match")


def decoder_directions(model, arch: str) -> torch.Tensor:
    """Return (d_sae, d_in) normalized decoder direction per feature."""
    with torch.no_grad():
        if arch == "stacked_sae":
            # saes[t].W_dec: (d_in, d_sae); average over T.
            Ws = [sae.W_dec.data for sae in model.saes]          # list of (d_in, d_sae)
            W = torch.stack(Ws, dim=0).mean(dim=0)                # (d_in, d_sae)
            D = W.T                                                # (d_sae, d_in)
        elif arch in ("crosscoder", "txcdr"):
            # W_dec: (d_sae, T, d_in). Average over T.
            D = model.W_dec.data.mean(dim=1)                      # (d_sae, d_in)
        elif arch.startswith("tfa"):
            # For BenchTFAAdapter, reach through to the inner TemporalSAE.
            inner = model._inner
            D = inner.D.data                                       # (width=d_sae, d_in)
        else:
            raise ValueError(arch)
    return F.normalize(D.float(), dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--subject-model", default="gemma-2-2b-it")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    assert len(args.archs) == len(args.ckpts)
    cfg = get_model_config(args.subject_model)

    # Load decoder directions for each arch
    decoders: dict[str, torch.Tensor] = {}
    for arch, ckpt in zip(args.archs, args.ckpts):
        log.info(f"loading {arch} <- {ckpt}")
        model = load_model(
            ckpt_path=ckpt, model_type=arch,
            subject_model=args.subject_model, k=args.k, T=args.T,
        ).to(args.device)
        decoders[arch] = decoder_directions(model, arch)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Pairwise best-match
    out: dict = {
        "archs": args.archs,
        "layer_key": None,
        "k": args.k, "T": args.T,
        "pairs": {},
    }
    for ai, a in enumerate(args.archs):
        for bj, b in enumerate(args.archs):
            if ai >= bj:
                continue
            log.info(f"computing {a} <-> {b}")
            sim = decoders[a] @ decoders[b].T  # (d_sae_A, d_sae_B)
            best_b = sim.max(dim=1)
            best_a = sim.max(dim=0)
            # Summary stats (over all features)
            stats = {
                "A_to_B": {
                    "median_best_sim": float(best_b.values.median().item()),
                    "mean_best_sim":   float(best_b.values.mean().item()),
                    "p90_best_sim":    float(torch.quantile(best_b.values, 0.9).item()),
                    "features_sim_above_0.7": int((best_b.values > 0.7).sum().item()),
                    "features_sim_above_0.5": int((best_b.values > 0.5).sum().item()),
                    "features_sim_below_0.3": int((best_b.values < 0.3).sum().item()),
                },
                "B_to_A": {
                    "median_best_sim": float(best_a.values.median().item()),
                    "mean_best_sim":   float(best_a.values.mean().item()),
                    "p90_best_sim":    float(torch.quantile(best_a.values, 0.9).item()),
                    "features_sim_above_0.7": int((best_a.values > 0.7).sum().item()),
                    "features_sim_above_0.5": int((best_a.values > 0.5).sum().item()),
                    "features_sim_below_0.3": int((best_a.values < 0.3).sum().item()),
                },
                "d_sae_A": sim.shape[0],
                "d_sae_B": sim.shape[1],
            }
            out["pairs"][f"{a}__vs__{b}"] = stats
            log.info(
                f"  {a}->{b} best sim: median={stats['A_to_B']['median_best_sim']:.3f}  "
                f"sim>0.7: {stats['A_to_B']['features_sim_above_0.7']}/{sim.shape[0]}"
            )
            log.info(
                f"  {b}->{a} best sim: median={stats['B_to_A']['median_best_sim']:.3f}  "
                f"sim>0.7: {stats['B_to_A']['features_sim_above_0.7']}/{sim.shape[1]}"
            )
            del sim, best_a, best_b

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
