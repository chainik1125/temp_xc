"""B2 plot: base vs reasoning per-offset firing curves for chosen B1 features.

Reads results/ward_backtracking_txc/b2/<hp>.npz and overlays base vs
reasoning mean activation curves for each top feature, ±1 SE shading,
vertical dashed lines at Ward window edges (-13, -8).
"""

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
    b2_dir = Path(cfg["paths"]["b2_dir"])
    out_dir = plots_dir(cfg)

    for hp in cfg["hookpoints"]:
        if not hp.get("enabled", True):
            continue
        path = b2_dir / f"{hp['key']}.npz"
        if not path.exists():
            print(f"[skip] {path}"); continue
        z = np.load(path, allow_pickle=True)
        offsets = z["offsets"]; fids = z["feature_ids"]
        K = len(fids)
        cols = min(K, 4); rows = (K + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
        for i, fid in enumerate(fids):
            ax = axes[i // cols, i % cols]
            for tag, color, key, key_se in (
                ("base D+", "tab:blue", "base_pos", "base_pos_se"),
                ("base D-", "lightblue", "base_neg", "base_neg_se"),
                ("reasoning D+", "tab:red", "reasoning_pos", "reasoning_pos_se"),
                ("reasoning D-", "lightsalmon", "reasoning_neg", "reasoning_neg_se"),
            ):
                y = z[key][i]; se = z[key_se][i]
                style = "-" if "D+" in tag else "--"
                ax.plot(offsets, y, style, label=tag, color=color, lw=1.5)
                ax.fill_between(offsets, y - se, y + se, color=color, alpha=0.15)
            ax.axvline(-13, color="gray", ls=":", lw=1)
            ax.axvline(-8, color="gray", ls=":", lw=1)
            ax.set_title(f"feature {int(fid)}", fontsize=10)
            ax.set_xlabel("Offset (rel. backtrack-sentence start)")
            ax.set_ylabel("encoder pre-act")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7, loc="best")
        # Hide empty subplots
        for j in range(K, rows * cols):
            axes[j // cols, j % cols].axis("off")
        fig.suptitle(f"B2 — per-offset firing (base vs reasoning) — {hp['key']}", fontsize=12)
        fig.tight_layout()
        out = out_dir / f"per_offset_firing_{hp['key']}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
