"""UMAP embedding of TXC decoder rows, colored by D+/D- selectivity.

UMAP install is optional; if missing, fall back to a 2-D PCA projection
which conveys the same shape/clustering information at lower fidelity.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.ward_backtracking_txc.plot._common import (
    load_cfg, plots_dir, features_npz_path, iter_arch_hookpoint,
)


def _embed_2d(X: np.ndarray, method: str = "auto") -> np.ndarray:
    if method in ("auto", "umap"):
        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine")
            return reducer.fit_transform(X)
        except Exception as e:
            if method == "umap":
                raise
            print(f"[fallback] umap missing ({e}) — using PCA")
    # PCA
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def _decoder_pos0_from_ckpt(arch: str, ckpt_path: Path) -> np.ndarray:
    """Return (d_sae, d) decoder rows for the pos-0 / single-slot view, per arch."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    if arch == "txc":
        W_dec = sd["W_dec"]                 # (d_sae, T, d)
        return W_dec[:, 0, :].numpy()
    if arch == "stacked_sae":
        # Per-position decoders live inside saes.0.W_dec, etc.
        W_dec0 = sd["saes.0.W_dec"]         # (d, d_sae)
        return W_dec0.T.numpy()
    if arch == "topk_sae":
        W_dec = sd["W_dec"]                 # (d, d_sae)
        return W_dec.T.numpy()
    if arch == "tsae":
        D = sd["D"]                         # (d_sae, d)
        return D.numpy()
    raise ValueError(f"unknown arch: {arch}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", default="auto", choices=["auto", "umap", "pca"])
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    out_dir = plots_dir(cfg)
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])

    for arch, hp in iter_arch_hookpoint(cfg):
        feat_path = features_npz_path(cfg, arch, hp["key"])
        ckpt_filename = f"txc_{hp['key']}.pt" if arch == "txc" else f"{arch}_{hp['key']}.pt"
        ckpt_path = ckpt_dir / ckpt_filename
        if not (feat_path.exists() and ckpt_path.exists()):
            continue
        z = np.load(feat_path, allow_pickle=True)
        all_scores = z["all_scores"]
        top = z["top_features"][: int(cfg["mining"]["top_k_for_steering"])]
        dec_pos0 = _decoder_pos0_from_ckpt(arch, ckpt_path)

        # Subsample for tractability
        n = dec_pos0.shape[0]
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=min(2048, n), replace=False)
        sub = dec_pos0[idx]
        sub_scores = all_scores[idx]
        emb = _embed_2d(sub, method=args.method)

        fig, ax = plt.subplots(figsize=(6.5, 6))
        v = max(np.percentile(np.abs(sub_scores), 99), 1e-6)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=sub_scores, cmap="RdBu_r",
                        vmin=-v, vmax=v, s=8, alpha=0.7, linewidths=0)
        # Highlight top features if present in sub
        in_sub = {int(j): i for i, j in enumerate(idx)}
        for f in top:
            if int(f) in in_sub:
                pi = in_sub[int(f)]
                ax.scatter(emb[pi, 0], emb[pi, 1], s=80, edgecolor="black",
                           facecolor="none", lw=1.2)
                ax.annotate(f"f{int(f)}", emb[pi], fontsize=7,
                            xytext=(4, 4), textcoords="offset points")
        fig.colorbar(sc, ax=ax, label="D+ − D-")
        ax.set_title(f"Decoder UMAP/PCA — {arch}/{hp['key']}")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        out = out_dir / f"decoder_umap_{arch}_{hp['key']}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
