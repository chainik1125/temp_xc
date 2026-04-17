#!/usr/bin/env python3
"""
tfa_pred_novel_split.py — For each top feature in a TFA scan, re-evaluate
its top-K exemplar windows and record the pred-codes vs novel-codes
contribution.

Motivation: the original scan (scan_features.py) ranks by novel_codes only.
If TFA generalization lives in pred_codes (dense, attention-predicted from
context), the "TFA is passage-local" finding from the initial cross-arch
comparison could be a novel-only artifact. This script answers: of each
feature's total activation mass, how much comes from novel vs pred?

Output: one row per (feature, exemplar) with novel_val, pred_val, and
derived per-feature aggregates (mass, fraction, ratio). JSON + TSV.

Usage:
    python -m temporal_crosscoders.NLP.tfa_pred_novel_split \\
        --scan results/nlp_sweep/gemma/scans/scan__tfa_pos__resid_L25__k50.json \\
        --ckpt results/nlp_sweep/gemma/ckpts/tfa_pos__gemma-2-2b-it__fineweb__resid_L25__k50__seed42.pt \\
        --out results/nlp_sweep/gemma/scans/tfa_pred_novel__resid_L25__k50.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch

from temporal_crosscoders.NLP.bench_adapters import BenchTFAAdapter
from temporal_crosscoders.NLP.config import cache_dir_for
from src.bench.model_registry import get_model_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pred_novel")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", required=True, help="Path to the tfa scan JSON")
    p.add_argument("--ckpt", required=True, help="Path to TFA ckpt")
    p.add_argument("--subject-model", default="gemma-2-2b-it")
    p.add_argument("--cached-dataset", default="fineweb")
    p.add_argument("--layer-key", default="resid_L25")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--expansion-factor", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    log.info(f"loading scan {args.scan}")
    with open(args.scan) as f:
        scan = json.load(f)

    cfg = get_model_config(args.subject_model)
    d_in = cfg.d_model
    d_sae = d_in * args.expansion_factor

    log.info(f"loading TFA ckpt {args.ckpt} (keep_pred_novel=True)")
    model = BenchTFAAdapter(
        d_in=d_in, d_sae=d_sae, T=args.T, k=args.k,
        use_pos_encoding=True, keep_pred_novel=True,
    )
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict({f"_inner.{k}": v for k, v in state.items()})
    model.eval().to(args.device)

    cache_dir = cache_dir_for(args.subject_model, args.cached_dataset)
    act_path = os.path.join(cache_dir, f"{args.layer_key}.npy")
    log.info(f"opening activations mmap {act_path}")
    data = np.load(act_path, mmap_mode="r")

    # Warm the scaling factor on a random batch so it matches the scan.
    # (The adapter caches _scaling_factor on the first forward.)
    with torch.no_grad():
        warm_idx = np.random.choice(data.shape[0], 4, replace=False)
        warm = torch.from_numpy(np.stack([data[i, :args.T].copy() for i in warm_idx])).float().to(args.device)
        _ = model(warm)
    log.info(f"scaling_factor={model._scaling_factor:.4e}")

    # Collect exemplars into a flat list for batched forward
    flat_exemplars = []  # (feat_idx, exemplar_idx, chain_idx, window_start)
    for fi_str, rec in scan["features"].items():
        for ei, ex in enumerate(rec["examples"]):
            flat_exemplars.append(
                (int(fi_str), ei, int(ex["chain_idx"]), int(ex["window_start"]))
            )
    log.info(f"{len(flat_exemplars)} (feature, exemplar) pairs to evaluate")

    # Batched forward. Each batch: BATCH exemplars × T tokens × d.
    BATCH = 128
    results: dict[int, dict] = {}  # feat_idx -> {pred_vals: [...], novel_vals: [...], ...}
    with torch.no_grad():
        for bstart in range(0, len(flat_exemplars), BATCH):
            batch = flat_exemplars[bstart : bstart + BATCH]
            # Build (B, T, d) tensor
            windows = np.stack([
                data[ci, ws : ws + args.T].copy() for (_, _, ci, ws) in batch
            ])
            x = torch.from_numpy(windows).float().to(args.device)
            # Forward in adapter — it scales, runs TemporalSAE, and stashes
            # last_novel/last_pred (both shape (B, T, d_sae))
            _ = model(x)
            novel = model.last_novel  # (B, T, d_sae)
            pred = model.last_pred     # (B, T, d_sae)
            for i, (fi, ei, ci, ws) in enumerate(batch):
                fe_novel = novel[i, :, fi].mean().item()   # mean over T — matches scan ranking
                fe_pred = pred[i, :, fi].mean().item()
                rec = results.setdefault(fi, {
                    "exemplars": [], "chain_idxs": [], "window_starts": [],
                    "novel_vals": [], "pred_vals": [],
                })
                rec["exemplars"].append(ei)
                rec["chain_idxs"].append(ci)
                rec["window_starts"].append(ws)
                rec["novel_vals"].append(fe_novel)
                rec["pred_vals"].append(fe_pred)

    # Per-feature aggregates
    for fi, rec in results.items():
        novel_sum = float(np.abs(rec["novel_vals"]).sum())
        pred_sum = float(np.abs(rec["pred_vals"]).sum())
        total = novel_sum + pred_sum
        rec["novel_mass"] = novel_sum
        rec["pred_mass"] = pred_sum
        rec["pred_frac"] = pred_sum / total if total > 1e-12 else 0.0
        rec["novel_frac"] = novel_sum / total if total > 1e-12 else 0.0

    out = {
        "arch": "tfa_pos",
        "subject_model": args.subject_model,
        "layer_key": args.layer_key,
        "k": args.k, "T": args.T,
        "scaling_factor": model._scaling_factor,
        "n_features": len(results),
        "features": {str(fi): rec for fi, rec in results.items()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"wrote {args.out}")

    # Quick summary
    pred_fracs = np.array([r["pred_frac"] for r in results.values()])
    log.info(f"pred_frac distribution (over {len(results)} features):")
    log.info(f"  median: {np.median(pred_fracs):.3f}")
    log.info(f"  mean:   {pred_fracs.mean():.3f}")
    log.info(f"  pred-dominant (>0.5): {(pred_fracs > 0.5).sum()} / {len(pred_fracs)}")
    log.info(f"  pred-heavy   (>0.8): {(pred_fracs > 0.8).sum()} / {len(pred_fracs)}")
    log.info(f"  novel-heavy  (<0.2): {(pred_fracs < 0.2).sum()} / {len(pred_fracs)}")


if __name__ == "__main__":
    main()
