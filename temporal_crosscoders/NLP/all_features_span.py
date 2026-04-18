"""Compute per-feature activation-based concentration + mass for ALL
d_sae features (not just top-300 by mass).

For each arch, we forward a sample of cached windows through the model
and accumulate, per feature:
  - per-position sum of |activation|  -> per-position mean
  - total mass (L1 sum across all tokens)
  - count of windows where feature is "on" (abs > epsilon)

Then:
  concentration = max_pos / sum_pos   (of per-position mean)
  mass = total L1 activation mass

Writes `span_all__<arch>__<layer>__k<k>.json` with per-feature records
for all d_sae features.

This is the input to a span-weighted feature ranking that does not
assume a mass-ranked top-300 subset.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from temporal_crosscoders.NLP.bench_adapters import (
    BenchTFAAdapter,
    load_bench_crosscoder,
    load_bench_stacked_sae,
)


def build_model(arch: str, d_in: int, d_sae: int, T: int, k: int):
    if arch == "stacked_sae":
        return load_bench_stacked_sae(d_in, d_sae, T, k)
    if arch == "crosscoder":
        return load_bench_crosscoder(d_in, d_sae, T, k)
    if arch == "tfa_pos":
        return BenchTFAAdapter(d_in, d_sae, T, k,
                               keep_pred_novel=True, feat_source="novel")
    if arch == "tfa_pos_pred":
        return BenchTFAAdapter(d_in, d_sae, T, k,
                               keep_pred_novel=True, feat_source="pred")
    raise ValueError(arch)


def iter_windows(cache: np.ndarray, T: int, batch_size: int,
                 rng: np.random.Generator):
    n_chains, t_per_chain, d = cache.shape
    starts = list(range(0, t_per_chain - T + 1, T))
    chain_order = rng.permutation(n_chains)
    buf = []
    for c in chain_order:
        for s in starts:
            buf.append(cache[c, s: s + T])
            if len(buf) == batch_size:
                yield torch.from_numpy(np.stack(buf)).float()
                buf = []
    if buf:
        yield torch.from_numpy(np.stack(buf)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["stacked_sae", "crosscoder", "tfa_pos", "tfa_pos_pred"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--expansion-factor", type=int, default=8)
    ap.add_argument("--sample-chains", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cache = np.load(args.cache, mmap_mode="r")
    d_in = cache.shape[-1]
    d_sae = d_in * args.expansion_factor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.arch, d_in, d_sae, args.T, args.k).to(device).eval()
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    if args.arch.startswith("tfa_pos"):
        load_state = {f"_inner.{k_}": v for k_, v in state.items()}
    else:
        load_state = state
    missing, unexpected = model.load_state_dict(load_state, strict=False)
    print(f"load: missing={len(missing)} unexpected={len(unexpected)}")

    sub_cache = np.array(cache[: args.sample_chains])

    # Accumulators
    per_pos_abs = torch.zeros(args.T, d_sae, device=device)  # (T, d_sae)
    total_mass = torch.zeros(d_sae, device=device)
    total_n_tokens = 0

    n_windows = 0
    with torch.no_grad():
        for batch in iter_windows(sub_cache, args.T, args.batch_size, rng):
            batch = batch.to(device)
            loss, x_hat, feat_acts = model(batch)
            # feat_acts: (B, T, d_sae)
            a = feat_acts.abs()  # magnitude
            per_pos_abs += a.sum(dim=0)  # (T, d_sae)
            total_mass += a.sum(dim=(0, 1))
            total_n_tokens += a.shape[0] * a.shape[1]
            n_windows += a.shape[0]
    print(f"processed {n_windows} windows / {total_n_tokens} tokens")

    per_pos_mean = per_pos_abs / n_windows  # (T, d_sae) mean |act| per position
    sum_pos = per_pos_mean.sum(dim=0).clamp(min=1e-12)  # (d_sae,)
    max_pos = per_pos_mean.max(dim=0).values  # (d_sae,)
    peak_pos = per_pos_mean.argmax(dim=0)  # (d_sae,)
    concentration = (max_pos / sum_pos).cpu().numpy()
    peak_pos_np = peak_pos.cpu().numpy()
    per_pos_mean_np = per_pos_mean.cpu().numpy()  # (T, d_sae)
    mass_np = total_mass.cpu().numpy()

    # emit per-feature record only for features with nonzero mass
    out = {
        "arch": args.arch,
        "ckpt": args.ckpt,
        "layer_cache": args.cache,
        "T": args.T,
        "k": args.k,
        "d_sae": int(d_sae),
        "n_windows_processed": int(n_windows),
        "n_tokens_processed": int(total_n_tokens),
        "features": {},
    }
    for fi in range(d_sae):
        if mass_np[fi] <= 1e-8:
            continue  # skip dead features
        out["features"][str(fi)] = {
            "mass": float(mass_np[fi]),
            "concentration": float(concentration[fi]),
            "peak_position": int(peak_pos_np[fi]),
            "per_position_mean": [float(x) for x in per_pos_mean_np[:, fi]],
        }
    print(f"emitted {len(out['features'])} live features out of {d_sae}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
