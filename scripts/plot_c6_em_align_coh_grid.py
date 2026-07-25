"""Per-cell coherence-vs-alignment α-trajectories for the 14B-finance
5-arch × 2-seed comparison.

Each subplot shows the headline finalist feature's α-sweep in
(mean_coh, mean_align) space, combining the canonical Wang stage-4 grid
(27 α points, in wang_full.json) with the dense extension grid
(30 α points in {±10..±90, ±110..±150, ±200}, in wang_full_extended.json).

Two alternate finalist features are drawn as faded grey curves for context.
Each subplot annotates two summary metrics on the headline:
  Δalign|coh≥70   = max(align) − min(align) over α with mean_coh ≥ 70
  peak align     = max(align) over all α
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
import numpy as np

LOCAL = Path(
    "/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc/"
    "dmitry/pre_purified/c6_em_overnight"
)

# (arch, seed, train_key, organism)
CELLS = [
    ("sae_arditi", 1,  "caa331cb08fce8bf", "14B-finance"),
    ("sae_arditi", 42, "700b9ff4d7c297af", "14B-finance"),
    ("tsae_paper", 1,  "2ebee6c2a3552ece", "14B-finance"),
    ("tsae_paper", 42, "98c9dd4cfd77a1dc", "14B-finance"),
    ("txc_base",   1,  "f9a1d482e5a48221", "14B-finance"),
    ("txc_base",   42, "99bc3c9f739c8f1b", "14B-finance"),
    ("txc_pro",    1,  "10d82cab97bace0a", "14B-finance"),
    ("txc_pro",    42, "ba79122b41fc96f6", "14B-finance"),
    ("tfa",        1,  "676c390321a106c3", "14B-finance"),
    ("tfa",        42, "da6a9fb42ed4e797", "14B-finance"),
]

ARCH_ORDER = ["sae_arditi", "tsae_paper", "txc_base", "txc_pro", "tfa"]
ARCH_LABEL = {
    "sae_arditi": "SAE",
    "tsae_paper": "T-SAE",
    "txc_base":   "TXC",
    "txc_pro":    "TXC-pro",
    "tfa":        "TFA",
}


def load_cell(train_key: str) -> dict:
    """Return {'finalists': [{'feature_id', 'rows', 'is_headline'}], 'dense_rows'}.

    Rows are dicts with keys (alpha, mean_align, mean_coh).
    """
    wang = json.loads((LOCAL / "runs" / f"c6_{train_key}" / "wang_full.json").read_text())
    headline_fid = (wang.get("headline") or {}).get("feature_id")
    finalists = []
    for f in wang.get("stage4", {}).get("finalists", []):
        finalists.append({
            "feature_id": f["feature_id"],
            "rows": [{"alpha": r["alpha"],
                      "mean_align": r["mean_align"],
                      "mean_coh":   r["mean_coh"]} for r in f.get("rows", [])],
            "is_headline": f["feature_id"] == headline_fid,
        })

    dense_path = LOCAL / "sweep_outputs" / f"c6_{train_key}" / "wang_full_extended.json"
    dense_rows = []
    if dense_path.exists():
        d = json.loads(dense_path.read_text())
        # dense rows are tagged with their feature_id; keep only the headline's
        dense_rows = [
            {"alpha": r["alpha"], "mean_align": r["mean_align"], "mean_coh": r["mean_coh"]}
            for r in d.get("rows", [])
            if r.get("feature_id") == headline_fid
        ]
    return {"finalists": finalists, "dense_rows": dense_rows, "headline_fid": headline_fid}


def merged_headline_curve(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (alphas, align, coh) for the headline feature, canonical+dense merged
    and sorted by alpha (no dedup so overlapping α=±10 lights the curve twice — fine)."""
    head = next(f for f in payload["finalists"] if f["is_headline"])
    rows = list(head["rows"]) + list(payload["dense_rows"])
    rows.sort(key=lambda r: r["alpha"])
    a = np.array([r["alpha"] for r in rows])
    al = np.array([r["mean_align"] for r in rows])
    co = np.array([r["mean_coh"] for r in rows])
    return a, al, co


def headline_metrics(payload: dict) -> tuple[float, float, int]:
    """Return (delta_align_coh70, peak_align, n_points_at_coh70).

    delta_align_coh70 = max(align) − min(align) over α with mean_coh ≥ 70.
    peak_align        = max(align) over all α (no coherence filter).
    """
    alphas, al, co = merged_headline_curve(payload)
    mask = co >= 70.0
    if mask.any():
        delta = float(al[mask].max() - al[mask].min())
        n70 = int(mask.sum())
    else:
        delta = float("nan")
        n70 = 0
    peak = float(al.max())
    return delta, peak, n70


def main():
    n_archs = len(ARCH_ORDER)
    n_cols = 2  # seed=1, seed=42 (14B-finance only)
    fig, axes = plt.subplots(n_archs, n_cols, figsize=(8, 14),
                             sharex=True, sharey=True)
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-200, vcenter=0, vmax=200)

    cells_by_key = {}
    for arch, seed, tk, organism in CELLS:
        cells_by_key[(arch, seed, organism)] = tk

    summary_rows = []
    for r, arch in enumerate(ARCH_ORDER):
        for c, seed in enumerate([1, 42]):
            ax = axes[r, c]
            tk = cells_by_key.get((arch, seed, "14B-finance"))
            if tk is None:
                ax.set_visible(False)
                continue
            payload = load_cell(tk)
            delta70, peak, n70 = headline_metrics(payload)
            summary_rows.append((arch, seed, tk, delta70, peak, n70))

            # alternate finalists in light grey
            for f in payload["finalists"]:
                if f["is_headline"]:
                    continue
                rows = sorted(f["rows"], key=lambda r: r["alpha"])
                ax.plot([r["mean_coh"] for r in rows],
                        [r["mean_align"] for r in rows],
                        color="lightgrey", lw=0.8, zorder=1)

            # headline trajectory: scatter colored by α
            alphas, al, co = merged_headline_curve(payload)
            order = np.argsort(alphas)
            ax.plot(co[order], al[order], color="0.4", lw=0.6, zorder=2)
            ax.scatter(co, al, c=alphas, cmap=cmap, norm=norm,
                       s=14, edgecolors="black", linewidths=0.3, zorder=3)

            # mark α=0 baseline with a star
            i0 = int(np.argmin(np.abs(alphas)))
            ax.scatter([co[i0]], [al[i0]], marker="*", s=70,
                       color="white", edgecolors="black", linewidths=0.7, zorder=4)

            ax.axhline(50, color="grey", lw=0.4, ls=":")
            ax.axvline(70, color="grey", lw=0.4, ls=":")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            head_fid = payload["headline_fid"]
            ax.set_title(f"{ARCH_LABEL[arch]}, s={seed}  feat={head_fid}",
                         fontsize=9)
            metric_str = (f"Δalign|coh≥70 = {delta70:.1f}\n"
                          f"peak align = {peak:.1f}")
            ax.text(0.03, 0.97, metric_str, transform=ax.transAxes,
                    fontsize=8, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="0.7", alpha=0.85))

    # Axis labels (only on edges)
    for r in range(n_archs):
        axes[r, 0].set_ylabel("alignment (%)")
    for c in range(n_cols):
        axes[-1, c].set_xlabel("coherence (%)")

    # Single shared colorbar
    fig.subplots_adjust(right=0.90, hspace=0.45, wspace=0.15)
    cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.76])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label("steering α", fontsize=9)

    fig.suptitle("c6 EM 14B-finance α-sweep: alignment vs coherence per (arch, seed)\n"
                 "headline feature coloured; alternate finalists in grey; "
                 "★ marks α≈0 baseline",
                 y=0.995, fontsize=11)

    out_dir = Path(
        "/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc/"
        "plots/2026-05-07_c6_em_align_coh_grid"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "align_coh_grid_14b_finance.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "align_coh_grid_14b_finance.pdf", bbox_inches="tight")
    print(f"wrote {out_dir / 'align_coh_grid_14b_finance.png'}")

    # Print summary table
    print()
    print(f"{'arch':<11} {'seed':>4}  {'train_key':<18} "
          f"{'Δalign|coh≥70':>14} {'peak_align':>11} {'n_pts_coh≥70':>13}")
    print("-" * 75)
    for arch, seed, tk, delta70, peak, n70 in summary_rows:
        d_str = f"{delta70:.1f}" if delta70 == delta70 else "n/a"
        print(f"{ARCH_LABEL[arch]:<11} {seed:>4}  {tk:<18} "
              f"{d_str:>14} {peak:>11.1f} {n70:>13d}")

    # Sidecar JSON for downstream use
    sidecar = [{"arch": a, "seed": s, "train_key": tk,
                "delta_align_coh70": d if d == d else None,
                "peak_align": p, "n_pts_coh70": n}
               for (a, s, tk, d, p, n) in summary_rows]
    (out_dir / "summary_metrics.json").write_text(
        json.dumps(sidecar, indent=2))
    print(f"\nwrote {out_dir / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()
