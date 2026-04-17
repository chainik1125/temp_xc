#!/usr/bin/env python3
"""
temporal_spread.py — For each top feature in a scan, re-evaluate its
top-K exemplars and record per-position activation (all T positions
within the window, not just the mean). Compute concentration score:

    conc(f) = mean over exemplars of max_pos / sum_pos

conc = 1 / T (=0.2 at T=5) means uniformly active across all positions;
conc = 1 means fully localized at one position.

Only meaningful for archs that emit per-token codes: stacked_sae, tfa_pos,
tfa_pos_pred. Crosscoder has a single z per window (no per-position data).

Usage:
    python -m temporal_crosscoders.NLP.temporal_spread \\
        --scan results/nlp_sweep/gemma/scans/scan__stacked_sae__resid_L25__k50.json \\
        --model-type stacked_sae \\
        --ckpt results/nlp_sweep/gemma/ckpts/stacked_sae__gemma-2-2b-it__fineweb__resid_L25__k50__seed42.pt \\
        --out results/nlp_sweep/gemma/scans/tspread__stacked_sae__resid_L25__k50.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch

from temporal_crosscoders.NLP.autointerp import load_model
from temporal_crosscoders.NLP.config import cache_dir_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("tspread")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", required=True)
    p.add_argument("--model-type", required=True,
                   choices=["stacked_sae", "tfa_pos", "tfa_pos_pred"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--subject-model", default="gemma-2-2b-it")
    p.add_argument("--cached-dataset", default="fineweb")
    p.add_argument("--layer-key", default="resid_L25")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    log.info(f"loading scan {args.scan}")
    with open(args.scan) as f:
        scan = json.load(f)

    model = load_model(
        ckpt_path=args.ckpt, model_type=args.model_type,
        subject_model=args.subject_model, k=args.k, T=args.T,
    ).to(args.device)

    cache_dir = cache_dir_for(args.subject_model, args.cached_dataset)
    data = np.load(
        os.path.join(cache_dir, f"{args.layer_key}.npy"), mmap_mode="r"
    )

    # Warm TFA scaling factor if applicable
    if args.model_type.startswith("tfa"):
        with torch.no_grad():
            warm = torch.from_numpy(
                np.stack([data[i, :args.T].copy()
                          for i in np.random.choice(data.shape[0], 4, replace=False)])
            ).float().to(args.device)
            _ = model(warm)

    # Build flat exemplar list
    flat = []
    for fi_str, rec in scan["features"].items():
        for ei, ex in enumerate(rec["examples"]):
            flat.append((int(fi_str), ei, int(ex["chain_idx"]), int(ex["window_start"])))
    log.info(f"{len(flat)} (feature, exemplar) pairs")

    BATCH = 128
    per_position: dict[int, list[list[float]]] = {}
    with torch.no_grad():
        for bstart in range(0, len(flat), BATCH):
            batch = flat[bstart : bstart + BATCH]
            x = torch.from_numpy(
                np.stack([data[ci, ws : ws + args.T].copy()
                          for (_, _, ci, ws) in batch])
            ).float().to(args.device)
            _, _, u = model(x)  # (B, T, d_sae) for these model types
            for i, (fi, ei, ci, ws) in enumerate(batch):
                vals = u[i, :, fi].detach().cpu().tolist()  # list of T floats
                per_position.setdefault(fi, []).append(vals)

    # Per-feature concentration
    summary = {}
    for fi, vals_list in per_position.items():
        arr = np.asarray(vals_list)  # (n_ex, T)
        abs_arr = np.abs(arr)
        rowsum = abs_arr.sum(axis=1)
        rowmax = abs_arr.max(axis=1)
        # avoid 0/0: skip exemplars with zero mass
        mask = rowsum > 1e-12
        if mask.any():
            conc = (rowmax[mask] / rowsum[mask]).mean()
        else:
            conc = float("nan")
        # Position with largest mean contribution
        pos_mean = abs_arr.mean(axis=0)  # (T,)
        peak_pos = int(pos_mean.argmax())
        summary[str(fi)] = {
            "n_exemplars": int(mask.sum()),
            "concentration": float(conc),
            "peak_position": peak_pos,
            "per_position_mean": pos_mean.tolist(),
            "per_exemplar": arr.tolist(),
        }

    out = {
        "arch": args.model_type,
        "layer_key": args.layer_key,
        "k": args.k, "T": args.T,
        "n_features": len(summary),
        "features": summary,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"wrote {args.out}")

    # Summary stats
    concs = [r["concentration"] for r in summary.values()
             if not np.isnan(r["concentration"])]
    concs = np.asarray(concs)
    log.info(f"concentration distribution over {len(concs)} features:")
    log.info(f"  median: {np.median(concs):.3f}  (1/T={1/args.T:.3f} = uniform)")
    log.info(f"  mean:   {concs.mean():.3f}")
    log.info(f"  localized (>0.5): {(concs > 0.5).sum()} / {len(concs)}")
    log.info(f"  spread (<0.3):    {(concs < 0.3).sum()} / {len(concs)}")

    # Peak-position distribution
    peaks = [r["peak_position"] for r in summary.values()]
    unique, counts = np.unique(peaks, return_counts=True)
    log.info(f"peak-position counts: {dict(zip(unique.tolist(), counts.tolist()))}")


if __name__ == "__main__":
    main()
