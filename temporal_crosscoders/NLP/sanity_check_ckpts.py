"""Phase 1 sanity check: verify every committed checkpoint loads clean.

For each .pt in results/nlp_sweep/gemma/ckpts/:
  1. load state dict
  2. assert no NaN/Inf in any tensor parameter
  3. run forward on 16 real cached windows
  4. assert loss, x_hat, feat_acts all finite

Prints a summary table and exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

from temporal_crosscoders.NLP.bench_adapters import (
    BenchTFAAdapter,
    load_bench_crosscoder,
    load_bench_stacked_sae,
)


CKPT_RE = re.compile(
    r"(?P<arch>stacked_sae|crosscoder|tfa_pos)__"
    r"(?P<model>[^_]+-[^_]+-[^_]+-[^_]+)__"
    r"(?P<dataset>[^_]+)__"
    r"(?P<layer>resid_L\d+)__"
    r"k(?P<k>\d+)__seed(?P<seed>\d+)"
    r"(?P<shuf>_shuffled)?\.pt"
)


def build_model(arch: str, d_in: int, d_sae: int, T: int, k: int):
    if arch == "stacked_sae":
        return load_bench_stacked_sae(d_in, d_sae, T, k)
    if arch == "crosscoder":
        return load_bench_crosscoder(d_in, d_sae, T, k)
    if arch == "tfa_pos":
        return BenchTFAAdapter(d_in, d_sae, T, k, keep_pred_novel=False)
    raise ValueError(arch)


def find_nan_inf(state: dict) -> list[str]:
    bad = []
    for name, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if torch.isnan(v).any() or torch.isinf(v).any():
            bad.append(name)
    return bad


def sample_windows(layer_cache: np.ndarray, T: int, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_chains, t_per_chain, d = layer_cache.shape
    assert t_per_chain >= T, f"need at least {T} tokens, got {t_per_chain}"
    chain_idx = rng.integers(0, n_chains, size=n)
    start = rng.integers(0, t_per_chain - T + 1, size=n)
    windows = np.stack(
        [layer_cache[c, s : s + T] for c, s in zip(chain_idx, start)]
    )
    return torch.from_numpy(windows).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="results/nlp_sweep/gemma/ckpts")
    p.add_argument(
        "--cache",
        default="data/cached_activations/gemma-2-2b-it/fineweb/resid_L25.npy",
    )
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--expansion-factor", type=int, default=8)
    p.add_argument("--n-windows", type=int, default=16)
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        print(f"no ckpts found in {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    # load cache once (mmap; only a small slice is materialized)
    layer_cache = np.load(args.cache, mmap_mode="r")
    d_in = layer_cache.shape[-1]
    windows = sample_windows(layer_cache, args.T, args.n_windows)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    windows = windows.to(device)

    d_sae = d_in * args.expansion_factor

    rows = []
    all_ok = True
    for ckpt in ckpts:
        m = CKPT_RE.match(ckpt.name)
        if not m:
            print(f"SKIP (unparsed): {ckpt.name}", file=sys.stderr)
            continue
        arch = m["arch"]
        k = int(m["k"])
        model = build_model(arch, d_in, d_sae, args.T, k).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)

        # The TFA adapter wraps TemporalSAE under self._inner; sweep saves
        # the inner state dict directly, so strip / add the prefix as needed.
        if arch == "tfa_pos":
            load_state = {f"_inner.{k_}": v for k_, v in state.items()}
        else:
            load_state = state
        missing, unexpected = model.load_state_dict(load_state, strict=False)
        # remove buffers intentionally left uninit (e.g. pos_enc caches)
        missing = [m for m in missing if "_scaling_factor" not in m]

        bad = find_nan_inf(load_state)

        with torch.no_grad():
            model.eval()
            loss, x_hat, feat_acts = model(windows)
        loss_finite = torch.isfinite(loss).item()
        x_hat_finite = torch.isfinite(x_hat).all().item()
        feat_finite = torch.isfinite(feat_acts).all().item()
        feat_nnz = (feat_acts != 0).float().mean().item()

        ok = (
            len(bad) == 0
            and loss_finite
            and x_hat_finite
            and feat_finite
            and len(missing) == 0
        )
        all_ok &= ok
        rows.append(
            dict(
                ckpt=ckpt.name,
                arch=arch,
                k=k,
                shuf=bool(m["shuf"]),
                nan_params=len(bad),
                loss=float(loss.item()),
                loss_finite=loss_finite,
                x_hat_finite=x_hat_finite,
                feat_finite=feat_finite,
                feat_nnz=feat_nnz,
                missing=len(missing),
                unexpected=len(unexpected),
                ok=ok,
            )
        )

    # print table
    header = (
        f"{'ckpt':<70} {'nan':>3} {'loss':>10} {'x_hat':>5} {'feat':>5}"
        f" {'nnz':>6} {'miss':>4} {'unex':>4} {'ok':>3}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['ckpt']:<70} {r['nan_params']:>3d} {r['loss']:>10.4f}"
            f" {int(r['x_hat_finite']):>5d} {int(r['feat_finite']):>5d}"
            f" {r['feat_nnz']:>6.3f} {r['missing']:>4d} {r['unexpected']:>4d}"
            f" {int(r['ok']):>3d}"
        )
    print()
    print("ALL OK" if all_ok else "FAILED")
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
