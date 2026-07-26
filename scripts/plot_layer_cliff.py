"""The last third of every model is unsteerable, and it ends at one layer.

`||Gbar||_F` is the norm of the metric's gradient with respect to a `(T, d)` write, in units
where a write of size 1 is one mean activation norm. It is a SIZE. Every screen this sprint
used before it -- `c`, `r1` -- is a SHAPE, which is why both fire identically on cells that
steer and cells that cannot: they describe how the optimal write is distributed over
positions and rank, never whether it does anything.

Measured at every layer of three models, 32 documents each, no dictionary trained.

WHAT THE FIGURE SHOWS. A one-layer collapse in all three: 8.4x after L20 of 28, 15.1x after
L14 of 24, 20.5x after L13 of 24 -- 0.58 to 0.75 of depth. Past it the metric cannot be moved
by any write of reasonable size, so a null measured there says nothing about dictionaries.
Three of the six depths the transfer negative sampled in SmolLM2 sit past its cliff.

IT IS NOT AN ARTEFACT OF THE RELATIVE-NORM CONVENTION. `act_norm` grows monotonically with
depth in all three models -- 10x over the stack, and 25x in SmolLM2 -- so a write of fixed
RELATIVE size is a growing ABSOLUTE perturbation, which could saturate the metric and
manufacture a cliff. Dividing it out (the dashed curves) makes the collapse SHARPER, not
flatter: 10.8x, 18.1x, 22.3x. The cliff is a property of the models.

THE POINTS ARE MEASURED STEERING, not screen values, and they are what makes this a
prediction rather than a description. Within SmolLM2 `||Gbar||` ranks all six tested depths
with one adjacent transposition. Across models it does not -- Qwen2.5-0.5B carries 27x the
`||Gbar||` of SmolLM2's L21 and moves a third as much -- so it is necessary and not
sufficient, the same status `c` has.

Reads results/txc_wins/layerscreen_{q15,smol,q05}.json and the transfer runs.
"""
import glob
import json
import pathlib
import sys

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "txc_wins"
OUT = ROOT / "plots" / "2026-07-26_txcwins" / "layer_cliff.png"

# Wong palette, consistent with every other figure in this sprint.
MODELS = [("_q15", "Qwen2.5-1.5B — the model that works", "#E69F00"),
          ("_q3b", "Qwen2.5-3B", "#CC79A7"),
          ("_smol", "SmolLM2-1.7B", "#0072B2"),
          ("_q05", "Qwen2.5-0.5B", "#009E73")]


def peak(arm):
    """Max over signs on a symmetric grid. Signed-positive indexing is the error three
    people on this sprint made independently; it does not get to appear in a new figure."""
    i = max(range(len(arm["delta_margin"])), key=lambda i: arm["delta_margin"][i])
    return arm["delta_margin"][i]


def measured():
    out = {}
    for f in glob.glob(str(RES / "recency*.json")):
        d = json.loads(pathlib.Path(f).read_text())
        if "grad_slab" not in d.get("arms", {}):
            continue
        out.setdefault((d.get("model"), d.get("layer")), peak(d["arms"]["grad_slab"]))
    return out


def main() -> int:
    meas = measured()
    have = [(t, lab, col) for t, lab, col in MODELS if (RES / f"layerscreen{t}.json").exists()]
    if not have:
        print("[skip] no layerscreen files")
        return 1

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.1),
                                  gridspec_kw={"width_ratios": [1.42, 1]})

    for tag, label, col in have:
        d = json.loads((RES / f"layerscreen{tag}.json").read_text())["models"][0]
        Ls = sorted(d["layers"], key=int)
        x = [int(L) / (len(Ls) - 1) for L in Ls]           # fraction of depth, comparable
        y = [d["layers"][L]["Gbar_fro"] for L in Ls]
        ax.plot(x, y, "-o", ms=3.4, lw=1.9, color=col, label=label, zorder=3)
        # Absolute units: the control that rules out the relative-norm convention.
        ya = [d["layers"][L]["Gbar_fro"] / d["layers"][L]["act_norm"] for L in Ls]
        sc = max(y) / max(ya)
        ax.plot(x, [v * sc for v in ya], ls="--", lw=1.2, color=col, alpha=0.5, zorder=2)
        # The cliff, excluding the always-degenerate final layer.
        i = max(range(len(Ls) - 2), key=lambda i: y[i] / max(y[i + 1], 1e-9))
        ax.axvline((x[i] + x[i + 1]) / 2, color=col, lw=1.0, ls=":", alpha=0.75, zorder=1)
        ax.annotate(f"{y[i] / max(y[i + 1], 1e-9):.0f}×", ((x[i] + x[i + 1]) / 2, max(y) * 0.62),
                    color=col, fontsize=9, ha="center",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
        # Right panel: screen value against the supervised arm measured at that layer.
        name = d["model"]
        for L in Ls:
            m = meas.get((name, int(L)))
            if m is not None:
                ax2.plot(d["layers"][L]["Gbar_fro"], m, "o", ms=9, color=col, zorder=3)
                ax2.annotate(f"L{L}", (d["layers"][L]["Gbar_fro"], m), fontsize=8,
                             textcoords="offset points", xytext=(7, -3), color="#444444")

    ax.set_yscale("log")
    ax.set_xlabel("fraction of model depth")
    ax.set_ylabel("‖Ḡ‖$_F$  — how much the metric moves per unit write\n"
                  "(solid: relative units;  dashed: absolute, rescaled)")
    ax.set_title("Every model has a one-layer steerability cliff\n"
                 "and it is not caused by the relative-norm convention", fontsize=11.5)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    ax2.set_xscale("log")
    ax2.set_xlabel("‖Ḡ‖$_F$ from the screen (no dictionary trained)")
    ax2.set_ylabel("measured Δ margin, supervised gradient write\n(peak over a symmetric dose grid)")
    ax2.set_title("The screen predicts within a model,\nnot across models", fontsize=11.5)
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.axhline(0, color="#999999", lw=0.8)
    ax2.annotate("Qwen2.5-0.5B carries 27× the ‖Ḡ‖\nof SmolLM2's L21 and moves a third\n"
                 "as much — necessary, not sufficient",
                 xy=(0.03, 0.97), xycoords="axes fraction", fontsize=8.5, va="top",
                 style="italic", color="#555555",
                 bbox=dict(fc="white", ec="#cccccc", alpha=0.9))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
