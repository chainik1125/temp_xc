"""Plot the coherence/suppression frontier from one or more
frontier_sweep.py output JSONs.

    uv run python -m experiments.em_features.plot_frontier \
        --sweep sae=~/em_features/results/qwen_l15_sae_frontier.json \
        --sweep txc=~/em_features/results/qwen_l15_txc_frontier.json \
        --out analysis/frontier_comparison_qwen_l15.png

Each ``--sweep LABEL=path.json`` adds one curve to the main panel, labeled
``LABEL``. Points are parameterised by α and coloured on a diverging
sequential colormap to make direction-of-sweep legible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="append", required=True,
                   help="LABEL=path/to/frontier_sweep.json. Repeat for multiple curves.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="Coherence / suppression frontier on Qwen-2.5-7B bad-medical (layer 15)")
    p.add_argument("--xlabel", default="mean_coherence (OpenAI judge)")
    p.add_argument("--ylabel", default="mean_alignment (OpenAI judge)")
    return p.parse_args()


def load_curve(path: Path):
    data = json.loads(Path(path).expanduser().read_text())
    rows = [r for r in data["rows"] if r.get("mean_alignment") is not None
            and r.get("mean_coherence") is not None]
    rows.sort(key=lambda r: r["alpha"])
    alphas = np.array([r["alpha"] for r in rows], dtype=float)
    align = np.array([r["mean_alignment"] for r in rows], dtype=float)
    coh = np.array([r["mean_coherence"] for r in rows], dtype=float)
    return alphas, coh, align


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")

    for i, spec in enumerate(args.sweep):
        if "=" not in spec:
            raise ValueError(f"--sweep expects LABEL=path, got {spec!r}")
        label, path = spec.split("=", 1)
        alphas, coh, align = load_curve(Path(path))
        if len(alphas) == 0:
            print(f"skipping {label}: no judged rows")
            continue

        ax.plot(coh, align, "-", color=f"C{i}", alpha=0.4, zorder=1)
        sc = ax.scatter(
            coh, align, c=alphas, cmap="coolwarm_r",
            marker=markers[i % len(markers)], s=60, edgecolor="k",
            linewidth=0.5, label=label, zorder=2,
        )
        # Annotate each α
        for a, x, y in zip(alphas, coh, align):
            ax.annotate(f"{a:+.2f}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, alpha=0.7)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("steering coefficient α")
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
