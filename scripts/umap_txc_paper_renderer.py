"""Render the TXC autointerp UMAP in c7-paper style.

Re-renders the `figs/umap_txc.png` using the precomputed coords +
HDBSCAN labels + cluster names from the qualitative-latents case study.
No heavy deps
(sentence-transformers / umap-learn / hdbscan) — those ran upstream.

Inputs:
  --coords       coords.npy  (150 x 2 UMAP positions)
  --labels       labels.npy  (150 cluster ids, -1 = noise)
  --summary      summary.json (cluster lexical labels + sizes)
  --output-dir

Output: umap_txc.png  — c7-paper style (no title, no top/right
spines, light grid alpha 0.25, legend outside axes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# c7 rcParams (matches the rest of the paper figs).
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Categorical palette — 20 colors via matplotlib tab20, sufficient for the
# 15-cluster TXC UMAP. UMAP cluster ids are not arch-aligned, so the
# c7 PAPER_ARCH_COLOR palette doesn't apply here.
CLUSTER_PALETTE = [matplotlib.cm.tab20(i) for i in range(20)]
NOISE_COLOR = "#BFBFBF"


def _truncate(s: str, n: int = 30) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _spread_centroids(centroids: list[tuple[int, float, float]],
                      min_sep: float) -> list[tuple[int, float, float]]:
    """Nudge centroid badges apart if they sit within `min_sep` of each
    other. Iterative pairwise repulsion, capped at a few passes."""
    pts = [(cid, np.array([cx, cy], dtype=float)) for cid, cx, cy in centroids]
    for _ in range(8):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = pts[j][1] - pts[i][1]
                dist = float(np.linalg.norm(d))
                if dist < 1e-6:
                    d = np.array([min_sep, 0.0])
                    dist = min_sep
                if dist < min_sep:
                    push = (min_sep - dist) / 2.0
                    direction = d / dist
                    pts[i] = (pts[i][0], pts[i][1] - direction * push)
                    pts[j] = (pts[j][0], pts[j][1] + direction * push)
                    moved = True
        if not moved:
            break
    return [(cid, float(p[0]), float(p[1])) for cid, p in pts]


def main(*, coords_path: Path, labels_path: Path, summary_path: Path,
         output_dir: Path) -> None:
    coords = np.load(coords_path)
    labels = np.load(labels_path)
    summary = json.loads(summary_path.read_text())
    cluster_meta = {c["cluster"]: c for c in summary["clusters"]}

    output_dir.mkdir(parents=True, exist_ok=True)
    # Landscape aspect — fills the paper text block when drawn at
    # \linewidth. Equal-aspect is dropped so the embedding fills the
    # panel; we set explicit xlim/ylim with a small margin instead.
    fig, ax = plt.subplots(figsize=(16.0, 6.0))

    n_total = len(labels)
    # Auto-size points by total count: many points need smaller markers.
    # At 5k points we still want clearly visible markers (s ~ 32).
    point_size = 60 if n_total <= 200 else (40 if n_total <= 1000 else 32)
    point_alpha = 0.9 if n_total <= 200 else (0.8 if n_total <= 1000 else 0.75)
    point_edge = 0.3 if n_total <= 1000 else 0.0  # drop edge for dense scatters

    # Plot noise first so it sits behind the clusters visually.
    noise_mask = labels < 0
    if noise_mask.any():
        ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
                   s=max(point_size - 2, 4), c=NOISE_COLOR, alpha=0.45,
                   edgecolors="none", label="noise", zorder=2)

    # Plot each cluster in id order so legend reads c0, c1, …
    # Use a generous truncation since legend lives outside the axes.
    cluster_ids = sorted({int(c) for c in labels if c >= 0})
    raw_centroids: list[tuple[int, float, float, int]] = []
    for idx, cid in enumerate(cluster_ids):
        m = labels == cid
        color = CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]
        meta = cluster_meta.get(cid, {})
        name = _truncate(meta.get("name", "?"), 48)
        n = int(m.sum())
        ax.scatter(coords[m, 0], coords[m, 1],
                   s=point_size, c=[color], alpha=point_alpha,
                   edgecolors="black" if point_edge > 0 else "none",
                   linewidths=point_edge,
                   label=f"c{cid} ({n}): {name}", zorder=5)
        cx = float(np.median(coords[m, 0]))
        cy = float(np.median(coords[m, 1]))
        raw_centroids.append((cid, cx, cy, n))

    # Drop badges for very small clusters — at <20 points the cluster is
    # better identified by its color in the legend than by an in-plot
    # badge that would crowd a denser neighbour. Keep larger clusters.
    BADGE_MIN_POINTS = 20
    badge_centroids = [(cid, cx, cy) for cid, cx, cy, n in raw_centroids
                       if n >= BADGE_MIN_POINTS]

    # Spread badges that landed too close so the digits stay readable.
    # min_sep chosen as a fraction of the larger UMAP axis range.
    xr = float(coords[:, 0].max() - coords[:, 0].min())
    yr = float(coords[:, 1].max() - coords[:, 1].min())
    min_sep = 0.06 * max(xr, yr)
    badge_centroids = _spread_centroids(badge_centroids, min_sep=min_sep)

    for cid, cx, cy in badge_centroids:
        ax.text(cx, cy, str(cid), ha="center", va="center",
                fontsize=9, fontweight="bold", color="#111",
                zorder=10,
                bbox=dict(facecolor="white", edgecolor="#555",
                          boxstyle="round,pad=0.14", linewidth=0.5,
                          alpha=0.94))

    # Tighten axes to the data with a small margin so clusters fill the
    # panel rather than floating in empty UMAP whitespace.
    margin_x = 0.04 * xr
    margin_y = 0.06 * yr
    ax.set_xlim(coords[:, 0].min() - margin_x, coords[:, 0].max() + margin_x)
    ax.set_ylim(coords[:, 1].min() - margin_y, coords[:, 1].max() + margin_y)

    # Axes — UMAP coords are arbitrary; label as UMAP-1 / UMAP-2 and
    # hide the tick numbers (they have no scientific meaning).
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend outside axes on the right — never overlaps points.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, ncol=1, handlelength=1.0,
              handletextpad=0.5, labelspacing=0.5,
              borderaxespad=0.0)

    fig.tight_layout()
    out = output_dir / "umap_txc.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[umap_txc] wrote {out} "
          f"(n_features={summary['n_features']}, "
          f"n_clusters={summary['n_clusters']}, "
          f"silhouette={summary['silhouette']:.2f}, "
          f"noise_frac={summary['noise_frac']:.2f})")


def _purified_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cli() -> None:
    root = _purified_root()
    umap_data = root / "experiments" / "c4_qualitative" / "umap_data"
    ap = argparse.ArgumentParser(description=(
        "C4 qualitative-latent UMAP renderer. Defaults to in-repo "
        "precomputed coords/labels/summary."
    ))
    ap.add_argument(
        "--coords", type=Path, default=umap_data / "coords.npy",
        help="UMAP coords .npy (default: in-repo).",
    )
    ap.add_argument(
        "--labels", type=Path, default=umap_data / "labels.npy",
        help="HDBSCAN cluster labels .npy (default: in-repo).",
    )
    ap.add_argument(
        "--summary", type=Path, default=umap_data / "summary.json",
        help="UMAP summary json (default: in-repo).",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=root / "figs",
        help=("Output directory (default: figs/, matching the "
              "paper's \\includegraphics{figs/umap_txc.png} path)."),
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(coords_path=args.coords, labels_path=args.labels,
         summary_path=args.summary, output_dir=args.output_dir)


if __name__ == "__main__":
    cli()
