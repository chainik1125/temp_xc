"""Phase 6 TSNE — paper-faithful projection variant.

Paper's `src/experiments/tsne.py` uses `sklearn.manifold.TSNE` with
`random_state=42, n_components=2`. Our `run_umap.py` UMAP didn't show
clean semantic clustering; this variant tests whether the projection
method is the bottleneck.

Shares the same ARCH_SPLITS + label conventions as run_umap.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
Z_DIR = REPO / "experiments/phase6_qualitative_latents/z_cache"
OUT_DIR = REPO / "experiments/phase6_qualitative_latents/results/tsne"

# reuse from run_umap
import sys
sys.path.insert(0, str(REPO / "experiments/phase6_qualitative_latents"))
from run_umap import ARCH_SPLITS, slice_high_low, load_z_and_labels, plot_umap  # noqa


def _run_tsne(z: np.ndarray, seed: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE
    red = TSNE(n_components=2, random_state=seed, metric="cosine",
               perplexity=30, init="pca")
    return red.fit_transform(z)


def _silhouette(x, labels):
    from sklearn.metrics import silhouette_score
    n = x.shape[0]
    if n > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=2000, replace=False)
        x = x[idx]; labels = labels[idx]
    if len(set(labels)) < 2: return float("nan")
    return float(silhouette_score(x, labels, metric="cosine"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", type=str, nargs="+",
                   default=["agentic_txc_02", "agentic_mlc_08",
                            "tsae_paper", "tsae_ours"])
    p.add_argument("--concat", type=str, default="concat_C_v2")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for arch in args.archs:
        z_path = Z_DIR / args.concat / f"{arch}__z.npy"
        if not z_path.exists():
            print(f"skip {arch}/{args.concat}: no z cache")
            continue
        print(f"[{arch}] TSNE on {args.concat}")
        z_flat, subj, qid, pos = load_z_and_labels(arch, args.concat)
        for kind, z_slice in (("high", slice_high_low(arch, z_flat)[0]),
                              ("low", slice_high_low(arch, z_flat)[1])):
            if z_slice is None: continue
            t0 = time.time()
            coords = _run_tsne(z_slice)
            dt = time.time() - t0
            print(f"  {arch}/{kind}: TSNE {dt:.1f}s")
            for ln, lbl in [("semantic", subj), ("context", qid), ("pos", pos)]:
                plot_umap(
                    coords, lbl,
                    title=f"{arch} — {kind} — {ln} — {args.concat} (TSNE)",
                    dst=OUT_DIR / f"{args.concat}__tsne_{kind}__{arch}__{ln}.png",
                )
                rows.append({"concat": args.concat, "arch": arch, "prefix": kind,
                             "label": ln, "silhouette": _silhouette(z_slice, lbl)})

    import csv
    dst = OUT_DIR / f"{args.concat}__tsne_silhouette_scores.csv"
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concat", "arch", "prefix", "label", "silhouette"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
