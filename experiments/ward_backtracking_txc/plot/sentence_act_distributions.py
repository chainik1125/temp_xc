"""Per-feature violin plots: D+ vs D- per-sentence activation distributions."""

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
    K_show = int(cfg["mining"]["top_k_for_steering"])

    for hp in cfg["hookpoints"]:
        if not hp.get("enabled", True): continue
        path = feat_dir / f"{hp['key']}.npz"
        if not path.exists(): continue
        z = np.load(path, allow_pickle=True)
        pos_act = z["pos_act"][:, :K_show]
        neg_act = z["neg_act"][:, :K_show]
        feats = z["top_features"][:K_show]
        K = pos_act.shape[1]
        fig, ax = plt.subplots(figsize=(max(6, K * 1.1), 4))
        # Build interleaved positions: 2i for D+ and 2i+0.6 for D-
        positions_pos = [2 * i for i in range(K)]
        positions_neg = [2 * i + 0.7 for i in range(K)]
        # avoid empty distributions for violinplot
        pos_data = [pos_act[:, i] if pos_act.shape[0] > 1 else np.array([0.0]) for i in range(K)]
        neg_data = [neg_act[:, i] if neg_act.shape[0] > 1 else np.array([0.0]) for i in range(K)]
        vp = ax.violinplot(pos_data, positions=positions_pos, widths=0.55, showmeans=True)
        for b in vp["bodies"]:
            b.set_facecolor("tab:red"); b.set_alpha(0.5)
        vn = ax.violinplot(neg_data, positions=positions_neg, widths=0.55, showmeans=True)
        for b in vn["bodies"]:
            b.set_facecolor("tab:blue"); b.set_alpha(0.5)
        ax.set_xticks([2 * i + 0.35 for i in range(K)])
        ax.set_xticklabels([f"f{int(f)}" for f in feats])
        ax.set_ylabel("encoded activation z")
        ax.set_title(f"D+ (red) vs D- (blue) per-sentence z — {hp['key']}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = out_dir / f"sentence_act_distributions_{hp['key']}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
