"""(n_features, n_offsets) heatmap of D+/D- pre-activation diff per hookpoint."""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.ward_backtracking_txc.plot._common import load_cfg, plots_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    feat_dir = Path(cfg["paths"]["features_dir"])
    out_dir = plots_dir(cfg)

    for hp in cfg["hookpoints"]:
        if not hp.get("enabled", True):
            continue
        path = feat_dir / f"{hp['key']}.npz"
        if not path.exists():
            print(f"[skip] {path}"); continue
        z = np.load(path, allow_pickle=True)
        per_off = z["per_offset"]   # (K, T) — D+ minus D- per slot
        offsets = z["offsets_window"].tolist()
        K, T = per_off.shape
        fig, ax = plt.subplots(figsize=(7, max(4, K * 0.18)))
        v = np.abs(per_off).max() or 1.0
        im = ax.imshow(per_off, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(T))
        ax.set_xticklabels([f"{o}" for o in offsets])
        ax.set_xlabel("Offset (rel. to backtracking sentence start)")
        ax.set_yticks(range(K))
        ax.set_yticklabels([f"f{int(f)}" for f in z["top_features"]])
        ax.set_ylabel("Feature (top-K, ranked)")
        ax.set_title(f"D+/D- pre-activation diff — {hp['key']}")
        fig.colorbar(im, ax=ax, label="mean(D+) − mean(D-)")
        fig.tight_layout()
        out = out_dir / f"feature_firing_heatmap_{hp['key']}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
