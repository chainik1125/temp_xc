"""Entrainment figure: does steering a prefix make the model finish the pattern?

Steer only the first W sentences, then release. Plot accuracy on the UNSTEERED tail
minus the analytic persistence null for that profile family, so every family is on a
common zero. Predictable families can exceed zero; unpredictable ones cannot.

    uv run python scripts/plot_entrain.py
"""

import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "results/temporal_screen/entrain2.json"))
OUT = ROOT / "plots/2026-07-24_trajectory_steering"

FAMS = [
    ("period2", "#0072B2", "o", "period-2 profile (predictable, ℓ=2)"),
    ("alt", "#009E73", "s", "alternating profile (predictable, ℓ=1)"),
    ("iid", "#E69F00", "^", "independent coin flips (unpredictable)"),
    ("balanced", "#CC79A7", "d", "balanced random (unpredictable)"),
]
WS = [1, 2, 3, 4]

fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
for fam, color, mk, label in FAMS:
    xs, ys, es = [], [], []
    for W in WS:
        c = R["cells"].get(f"{fam}_W{W}")
        if not c or c["unsteered_acc"] is None:
            continue
        xs.append(W)
        ys.append(c["unsteered_acc"] - c["analytic_persistence_null"])
        es.append(c["unsteered_sem"])
    ax.errorbar(xs, ys, yerr=es, color=color, marker=mk, ms=7, lw=2, capsize=3,
                label=label)
ax.axhline(0, color="0.45", lw=1.3, ls="--")
ax.annotate("no entrainment\n(model persists, nothing more)", (1.02, 0.0),
            textcoords="offset points", xytext=(4, -30), fontsize=8.5, color="0.4")
ax.axvline(3, color="0.85", lw=8, zorder=0)
ax.annotate("W = ℓ+1: the first flip is visible,\nso the period is knowable",
            (3, 0.235), ha="center", fontsize=8.5, color="#0072B2",
            fontweight="bold")
ax.set_xticks(WS)
ax.set_xlabel("W — number of leading sentences that were steered (of 6)")
ax.set_ylabel("tail accuracy above the analytic null")
ax.set_title("Steering a wide enough prefix makes the model carry the pattern on\n"
             "its own — but only when the pattern is knowable from that prefix",
             fontsize=10.5)
ax.legend(fontsize=8.5, frameon=True, edgecolor="0.85", loc="lower left")
ax.grid(axis="y", color="0.92", lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"entrainment.{ext}", bbox_inches="tight")
print("saved", OUT / "entrainment.png")
