"""Two-panel alpha-curve figure: effect on top, quality underneath.

    uv run --no-sync python experiments/backtracking_steering_dsm/plot_curves.py \
        --curves <curves.json> --out <fig.png> [--only tagA tagB]

The quality panel sits directly under the effect panel on a shared alpha axis
because the two are only interpretable together: a large Delta-gc at a magnitude
where the coherence floor has already collapsed is not a result. Magnitudes
whose cell fails the run-length floor are drawn hollow in the effect panel, so a
peak that only exists among incoherent generations is visible as one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Wong colour-blind-safe palette, the house style for these figures.
WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def plot(curves: dict, out: Path, only: list[str] | None = None,
         title: str = "") -> Path:
    tags = [t for t in curves if not only or t in only]
    fig, (ax, axq) = plt.subplots(
        2, 1, figsize=(8.2, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08})

    for i, tag in enumerate(tags):
        c = sorted(curves[tag], key=lambda x: x["magnitude"])
        col = WONG[(i % (len(WONG) - 1)) + 1] if tag != "dom_base_union" else WONG[0]
        mags = [x["magnitude"] for x in c]
        gc0 = next(x["gc"] for x in c if x["magnitude"] == 0.0)
        d = [x["gc"] - gc0 for x in c]
        ax.plot(mags, d, "-", color=col, lw=1.6, label=tag, zorder=3)
        coh = [x["cell_coh_run"] for x in c]
        ax.scatter([m for m, k in zip(mags, coh) if k],
                   [v for v, k in zip(d, coh) if k],
                   s=26, color=col, zorder=4)
        ax.scatter([m for m, k in zip(mags, coh) if not k],
                   [v for v, k in zip(d, coh) if not k],
                   s=26, facecolors="none", edgecolors=col, zorder=4)
        axq.plot(mags, [x["frac_coh_run"] for x in c], "-", color=col, lw=1.6)
        axq.plot(mags, [x["frac_coh_sonnet"] for x in c], "--", color=col,
                 lw=1.2, alpha=0.8)

    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.axvline(0, color="0.6", lw=0.8, zorder=1)
    ax.set_ylabel(r"$\Delta$ genuine events / generation")
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.95)
    if title:
        ax.set_title(title, fontsize=11)

    axq.set_ylim(-0.03, 1.03)
    axq.axvline(0, color="0.6", lw=0.8)
    axq.set_ylabel("fraction coherent")
    axq.set_xlabel(r"steering magnitude $\alpha$")
    box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0}
    axq.text(0.99, 0.08, "solid: run-length floor   dashed: Sonnet grade $\\geq$ 2",
             transform=axq.transAxes, ha="right", fontsize=8, color="0.35",
             bbox=box)
    ax.text(0.99, 0.97, "hollow markers: cell fails the run-length floor",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="0.35", bbox=box)

    for a in (ax, axq):
        a.spines[["top", "right"]].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--curves", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--title", default="")
    a = p.parse_args(argv)
    print(plot(json.loads(a.curves.read_text()), a.out, a.only, a.title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
