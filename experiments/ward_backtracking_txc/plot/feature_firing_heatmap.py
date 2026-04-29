"""(n_features, n_offsets) heatmap of D+/D- pre-activation diff per hookpoint."""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.ward_backtracking_txc.plot._common import (
    load_cfg, plots_dir, features_npz_path, iter_arch_hookpoint,
)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    out_dir = plots_dir(cfg)

    for arch, hp in iter_arch_hookpoint(cfg):
        path = features_npz_path(cfg, arch, hp["key"])
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
        ax.set_title(f"D+/D- pre-activation diff — {arch}/{hp['key']}")
        fig.colorbar(im, ax=ax, label="mean(D+) − mean(D-)")
        fig.tight_layout()
        out = out_dir / f"feature_firing_heatmap_{arch}_{hp['key']}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
