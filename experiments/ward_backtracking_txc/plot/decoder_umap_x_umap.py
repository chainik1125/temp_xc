"""Per-hookpoint decoder UMAPs side-by-side with bipartite-feature matching.

For each hookpoint pair, draws cos > 0.5 edges between top-K features,
overlaid on per-hookpoint UMAP/PCA scatter.
"""

from __future__ import annotations
import argparse, itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.ward_backtracking_txc.plot._common import load_cfg, plots_dir
from experiments.ward_backtracking_txc.plot.decoder_umap import _embed_2d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", default="auto", choices=["auto", "umap", "pca"])
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    feat_dir = Path(cfg["paths"]["features_dir"])
    out_dir = plots_dir(cfg)
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])

    enabled = [hp for hp in cfg["hookpoints"] if hp.get("enabled", True)]
    if len(enabled) < 1:
        return

    # Load top-K decoder rows per hookpoint
    bundle: dict[str, dict] = {}
    K = int(cfg["mining"]["top_k_for_steering"])
    for hp in enabled:
        feat_path = feat_dir / f"{hp['key']}.npz"
        ckpt_path = ckpt_dir / f"txc_{hp['key']}.pt"
        if not (feat_path.exists() and ckpt_path.exists()):
            continue
        z = np.load(feat_path, allow_pickle=True)
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        W_dec = obj["state_dict"]["W_dec"]
        dec_pos0 = W_dec[:, 0, :].numpy()
        all_scores = z["all_scores"]
        top = z["top_features"][:K]
        bundle[hp["key"]] = {
            "top": top.tolist(),
            "top_vecs": dec_pos0[top.astype(int)],
            "all_dec": dec_pos0,
            "all_scores": all_scores,
        }

    if len(bundle) == 0:
        print("[skip] no hookpoints ready"); return

    n = len(bundle)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    embeddings = {}
    for col, (key, d) in enumerate(bundle.items()):
        rng = np.random.default_rng(42)
        sub_n = min(1024, d["all_dec"].shape[0])
        idx = rng.choice(d["all_dec"].shape[0], size=sub_n, replace=False)
        sub = d["all_dec"][idx]
        sub_scores = d["all_scores"][idx]
        emb = _embed_2d(sub, method=args.method)
        embeddings[key] = (idx, emb, sub_scores)
        ax = axes[0, col]
        v = max(np.percentile(np.abs(sub_scores), 99), 1e-6)
        ax.scatter(emb[:, 0], emb[:, 1], c=sub_scores, cmap="RdBu_r",
                   vmin=-v, vmax=v, s=6, alpha=0.6, linewidths=0)
        in_sub = {int(j): i for i, j in enumerate(idx)}
        for f in d["top"]:
            if int(f) in in_sub:
                pi = in_sub[int(f)]
                ax.scatter(emb[pi, 0], emb[pi, 1], s=80, edgecolor="black",
                           facecolor="none", lw=1.2)
        ax.set_title(key); ax.set_xticks([]); ax.set_yticks([])

    # Bipartite matches
    keys = list(bundle.keys())
    for k1, k2 in itertools.combinations(keys, 2):
        v1 = bundle[k1]["top_vecs"]; v2 = bundle[k2]["top_vecs"]
        n1 = np.linalg.norm(v1, axis=1, keepdims=True).clip(min=1e-8)
        n2 = np.linalg.norm(v2, axis=1, keepdims=True).clip(min=1e-8)
        cos = (v1 / n1) @ (v2 / n2).T   # (K, K)
        # report matching pairs
        for i in range(cos.shape[0]):
            for j in range(cos.shape[1]):
                if cos[i, j] > 0.5:
                    print(f"[match] {k1}/f{bundle[k1]['top'][i]} ~ "
                          f"{k2}/f{bundle[k2]['top'][j]}  cos={cos[i,j]:.3f}")

    fig.suptitle("Per-hookpoint decoder UMAPs (top-K highlighted; matches printed to stdout)")
    fig.tight_layout()
    out = out_dir / "decoder_umap_x_umap.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
