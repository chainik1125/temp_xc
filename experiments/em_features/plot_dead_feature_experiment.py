"""Plot the output of test_dead_feature_resample.py + test_gao_stack.py.
Auto-detects whichever conditions are present in the JSON as top-level
keys that are lists of log dicts with a 'step' field.

    uv run python -m experiments.em_features.plot_dead_feature_experiment \
        --json /root/em_features/results/dead_feature_experiment/dead_feature_experiment.json \
        --out /root/em_features/results/figures/dead_feature_experiment.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.json.read_text())

    conditions = []
    for k, v in data.items():
        if k.endswith("_args") or k == "args":
            continue
        if isinstance(v, list) and v and isinstance(v[0], dict) and "step" in v[0]:
            conditions.append((k, v))
    if not conditions:
        raise SystemExit("no condition-shaped entries found in JSON")

    def col(rows, key):
        return [r.get(key) for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    n_feat = conditions[0][1][0]["n_features"]

    for i, (name, rows) in enumerate(conditions):
        style = dict(marker=MARKERS[i % len(MARKERS)], linestyle="-",
                     color=f"C{i}", label=name)
        steps = col(rows, "step")
        losses = col(rows, "loss")
        dead_frac = [100 * r["n_dead"] / r["n_features"] for r in rows]
        max_fires = col(rows, "max_fire")
        axes[0].plot(steps, losses, **style)
        axes[1].plot(steps, dead_frac, **style)
        axes[2].plot(steps, max_fires, **style)
        if "loss_auxk" in rows[0] and rows[0]["loss_auxk"] is not None:
            axes[3].plot(steps, col(rows, "loss_auxk"), **style)
        elif "n_resampled_so_far" in rows[0] and rows[0]["n_resampled_so_far"]:
            axes[3].plot(steps, col(rows, "n_resampled_so_far"), **style)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("main loss (log)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"TXC d_sae={n_feat}, T=5, k=128 — dead-feature ablation")

    axes[1].set_ylabel("dead features (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("max single-feature fire count (of 2048 probes)")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_ylabel("auxk loss / cumulative resamples")
    axes[3].set_xlabel("training step")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
