"""Phase 1 sanity check: does TFA pred vs novel partition hold at top-500?

Runs the trained TFA ckpt on a sample of cached activations, accumulates
per-feature novel_mass and pred_mass (L1 over all tokens), then ranks
both and reports overlap for N in {50, 100, 200, 500, 1000, 2000}.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from temporal_crosscoders.NLP.bench_adapters import BenchTFAAdapter


def iter_windows(cache: np.ndarray, T: int, batch_size: int, rng: np.random.Generator):
    """Yield (B, T, d) float32 tensors of non-overlapping windows sampled from
    random chains. Deterministic once seeded."""
    n_chains, t_per_chain, d = cache.shape
    # use all chains; within each chain step through non-overlapping windows
    # to match how the log's numbers were computed.
    starts = list(range(0, t_per_chain - T + 1, T))  # non-overlapping
    chain_order = rng.permutation(n_chains)
    buf = []
    for c in chain_order:
        for s in starts:
            buf.append(cache[c, s : s + T])
            if len(buf) == batch_size:
                yield torch.from_numpy(np.stack(buf)).float()
                buf = []
    if buf:
        yield torch.from_numpy(np.stack(buf)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="results/nlp_sweep/gemma/ckpts/tfa_pos__gemma-2-2b-it__fineweb__resid_L25__k50__seed42.pt",
    )
    ap.add_argument(
        "--cache",
        default="data/cached_activations/gemma-2-2b-it/fineweb/resid_L25.npy",
    )
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
    model = BenchTFAAdapter(
        d_in=d_in,
        d_sae=d_sae,
        T=args.T,
        k=args.k,
        keep_pred_novel=True,
        feat_source="novel",
    ).to(device).eval()
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    load_state = {f"_inner.{k_}": v for k_, v in state.items()}
    missing, unexpected = model.load_state_dict(load_state, strict=False)
    print(f"load: missing={len(missing)} unexpected={len(unexpected)}")

    # only process the first N chains (subsample deterministically)
    sub_cache = np.array(cache[: args.sample_chains])

    novel_mass = torch.zeros(d_sae, device=device)
    pred_mass = torch.zeros(d_sae, device=device)
    n_windows = 0
    with torch.no_grad():
        for batch in iter_windows(sub_cache, args.T, args.batch_size, rng):
            batch = batch.to(device)
            _ = model(batch)  # populates model.last_novel/last_pred
            novel = model.last_novel
            pred = model.last_pred
            # L1 mass per feature, summed over batch and time
            novel_mass += novel.abs().sum(dim=(0, 1))
            pred_mass += pred.abs().sum(dim=(0, 1))
            n_windows += batch.shape[0]
    print(f"processed {n_windows} windows")

    novel_order = torch.argsort(novel_mass, descending=True).cpu().numpy()
    pred_order = torch.argsort(pred_mass, descending=True).cpu().numpy()

    print("Top-N overlap between novel-ranked and pred-ranked features:")
    for N in [50, 100, 200, 500, 1000, 2000]:
        a = set(novel_order[:N].tolist())
        b = set(pred_order[:N].tolist())
        inter = len(a & b)
        # expected random overlap: N^2 / d_sae
        exp = N * N / d_sae
        print(
            f"  top-{N:5d}: overlap={inter:5d}  (rand expect≈{exp:.1f}  ratio={inter/exp if exp>0 else 0:.2f})"
        )


if __name__ == "__main__":
    main()
