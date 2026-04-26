"""Plot reconstruction loss and dead-feature trajectory for one or more
training runs from their ``*_training.meta.json`` files.

    uv run python -m experiments.em_features.plot_training_history \\
        --metas /root/em_features/checkpoints/*_training.meta.json \\
        --out /root/em_features/results/training_history.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metas", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    metas = []
    for path in args.metas:
        if not path.exists():
            continue
        try:
            d = json.loads(path.read_text())
        except Exception as e:
            print(f"skip {path}: {e}", file=sys.stderr)
            continue
        # Accept training-meta and (legacy) sae loss_history shapes.
        history = d.get("history") or d.get("loss_history") or []
        if not history:
            continue
        # name: stem minus "_training"
        name = path.stem.replace("_training", "")
        metas.append((name, history))

    if not metas:
        print("no metas with history; nothing to plot", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_loss, ax_dead = axes

    for name, history in metas:
        # Robust to dicts (TXC/MLC/Han/SAE-new) and floats (legacy SAE loss_history).
        steps, losses, dead_pct = [], [], []
        for entry in history:
            if isinstance(entry, dict):
                steps.append(int(entry.get("step", len(steps) + 1)))
                losses.append(float(entry.get("loss", float("nan"))))
                n_d = entry.get("n_dead")
                n_f = entry.get("n_features")
                if n_d is not None and n_f:
                    dead_pct.append(100.0 * n_d / n_f)
                else:
                    dead_pct.append(float("nan"))
            else:
                steps.append(len(steps) + 1)
                losses.append(float(entry))
                dead_pct.append(float("nan"))

        ax_loss.plot(steps, losses, label=name, alpha=0.85)
        if any(p == p for p in dead_pct):  # any non-NaN
            ax_dead.plot(steps, dead_pct, label=name, alpha=0.85)

    ax_loss.set_ylabel("recon loss (log scale)")
    ax_loss.set_yscale("log")
    ax_loss.legend(fontsize=8, loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    ax_dead.set_xlabel("training step")
    ax_dead.set_ylabel("dead features (%)")
    ax_dead.set_ylim(0, 100)
    ax_dead.legend(fontsize=8, loc="upper right")
    ax_dead.grid(True, alpha=0.3)

    fig.suptitle("Training trajectory: reconstruction loss + dead-feature %")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
