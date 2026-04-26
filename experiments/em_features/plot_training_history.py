"""Plot reconstruction loss and dead-feature trajectory for one or more
training runs from their ``*_training.meta.json`` files.

    uv run python -m experiments.em_features.plot_training_history \\
        --metas /root/em_features/checkpoints/*_training.meta.json \\
        --out /root/em_features/results/training_history.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Parses lines like:
#   [sae] step    500/10000  loss=3245.8130  L0=128.00  dead=26128/32768 (79.7%) ...
#   [brickenauxk] step    500/10000  loss=2381.7  auxk=2381.76  dead=22920/32768 (69.9%) ...
#   [han_champ] step    500/10000  loss=12500  auxk=1.02  dead=25783/32768 (78.7%) ...
LOG_LINE_RE = re.compile(
    r"\[(?P<tag>[^\]]+)\]\s+step\s+(?P<step>\d+)/\d+\s+loss=(?P<loss>[-\d.eE]+).*?dead=(?P<n_dead>\d+)/(?P<n_feat>\d+)"
)


def parse_log_file(path: Path) -> tuple[str, list[dict]]:
    """Returns (run_name, history) extracted from a training log."""
    history: list[dict] = []
    tag = None
    for line in path.read_text(errors="ignore").splitlines():
        m = LOG_LINE_RE.search(line)
        if not m:
            continue
        tag = m.group("tag")
        history.append({
            "step": int(m.group("step")),
            "loss": float(m.group("loss")),
            "n_dead": int(m.group("n_dead")),
            "n_features": int(m.group("n_feat")),
        })
    return (tag or path.stem), history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metas", type=Path, nargs="*", default=[],
                   help="*_training.meta.json files (preferred — full history)")
    p.add_argument("--logs", type=Path, nargs="*", default=[],
                   help="Training log files to parse — fallback for in-progress runs")
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
        history = d.get("history") or d.get("loss_history") or []
        if not history:
            continue
        name = path.stem.replace("_training", "")
        metas.append((name, history))

    seen_names = {n for n, _ in metas}
    for path in args.logs:
        if not path.exists():
            continue
        name, history = parse_log_file(path)
        if not history:
            continue
        # Disambiguate by appending log stem if a meta with the same tag already exists
        if name in seen_names:
            name = f"{name}@{path.stem}"
        metas.append((name, history))

    if not metas:
        print("no metas/logs with history; nothing to plot", file=sys.stderr)
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
