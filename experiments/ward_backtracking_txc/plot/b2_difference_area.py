"""B2 quantitative bar chart: integrated |reasoning(o) - base(o)| per feature."""

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

    n_hp = sum(1 for hp in cfg["hookpoints"] if hp.get("enabled", True))
    fig, axes = plt.subplots(1, max(1, n_hp), figsize=(5 * max(1, n_hp), 4),
                             squeeze=False)
    col = 0
    for hp in cfg["hookpoints"]:
        if not hp.get("enabled", True): continue
        path = b2_dir / f"{hp['key']}.npz"
        if not path.exists():
            col += 1; continue
        z = np.load(path, allow_pickle=True)
        offsets = z["offsets"]
        # Use D+ curves (diff base vs reasoning on the D+ subset)
        diff = np.abs(z["reasoning_pos"] - z["base_pos"])  # (K, n_off)
        # Trapezoidal integration over offset
        area = np.trapezoid(diff, x=offsets, axis=1)
        order = np.argsort(-area)
        ax = axes[0, col]
        ax.bar(range(len(area)), area[order], color="tab:purple")
        ax.set_xticks(range(len(area)))
        ax.set_xticklabels([f"f{int(z['feature_ids'][i])}" for i in order],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("∫|reasoning(o) − base(o)| do")
        ax.set_title(f"{hp['key']} — cross-model temporal diff")
        ax.grid(alpha=0.3, axis="y")
        col += 1
    fig.tight_layout()
    out = out_dir / "b2_difference_area.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
