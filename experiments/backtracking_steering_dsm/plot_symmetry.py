"""Excess directional component per source, against the random-direction control.

    uv run --no-sync --with matplotlib python \
        experiments/backtracking_steering_dsm/plot_symmetry.py \
        --symmetry <symmetry.json> --out <fig.png>

The single figure for the wave-1 headline. Each bar is a source's odd
(direction) component minus the control's at matched |alpha|, with a
prompt-resampling bootstrap 95% CI. Zero means "does nothing a norm-matched
random vector would not also do".

House convention: black for conventional steering (the DoM baseline), grey
dashed for the random control's own zero line, Wong palette for the dictionary
arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WONG = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7"]
DOM_TAG = "dom_base_union"
CONTROL_TAG = "control_random"


def plot(sym: dict, out: Path, title: str = "") -> Path:
    rows = [(t, d["bootstrap_excess_anti"]) for t, d in sym.items()
            if "bootstrap_excess_anti" in d]
    rows.sort(key=lambda kv: kv[1]["excess_anti"])
    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(rows) + 2.0))

    ci = 0
    for i, (tag, b) in enumerate(rows):
        col = "#000000" if tag == DOM_TAG else WONG[ci % len(WONG)]
        if tag != DOM_TAG:
            ci += 1
        x = b["excess_anti"]
        lo, hi = b["ci95"]
        sig = b["excludes_zero"]
        ax.barh(i, x, height=0.62, color=col, alpha=0.95 if sig else 0.42,
                edgecolor=col, linewidth=1.2, zorder=3)
        ax.plot([lo, hi], [i, i], color="0.15", lw=1.5, zorder=4)
        ax.plot([lo, lo], [i - 0.14, i + 0.14], color="0.15", lw=1.5, zorder=4)
        ax.plot([hi, hi], [i - 0.14, i + 0.14], color="0.15", lw=1.5, zorder=4)

    ax.axvline(0, color="0.45", lw=1.4, ls="--", zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([t for t, _ in rows], fontsize=9)
    ax.set_xlabel("excess directional component vs norm-matched random direction\n"
                  r"(odd part of the $\Delta$gc curve, minus the control's)",
                  fontsize=10)
    ax.text(0.0, len(rows) - 0.35, "  random control", fontsize=8.5,
            color="0.35", va="center")
    if title:
        ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="0.9", lw=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symmetry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="")
    a = p.parse_args(argv)
    print(plot(json.loads(a.symmetry.read_text()), a.out, a.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
