"""Linearity figure: predicted vs measured across a continuum, including negative.

For any balanced profile π and any block-constant coefficient vector c, a linear
steering response predicts R = W·⟨c, μ⟩ / k. Sampling random (profile, W, c) gives many
distinct predictions spanning [−0.42, +0.42], so the claim is tested as a regression
rather than at two points — and the negative half is the risky part: a write that
opposes the target should move the margin proportionally the wrong way.

    uv run python scripts/plot_linfit.py
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "results/temporal_screen/linfit.json"))
OUT = ROOT / "plots/2026-07-24_trajectory_steering"

P = np.array([p["pred_R"] for p in R["points"]])
O = np.array([p["obs_R"] for p in R["points"]])
LO = np.array([p["ci95"][0] for p in R["points"]])
HI = np.array([p["ci95"][1] for p in R["points"]])
W = np.array([p["W"] for p in R["points"]])
f = R["fit"]

fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=150)
lim = 0.55
ax.plot([-lim, lim], [-lim, lim], color="0.55", lw=1.4, ls="--", zorder=1,
        label="linear response (no fitted parameters)")
xs = np.linspace(-lim, lim, 10)
ax.plot(xs, f["slope"] * xs + f["intercept"], color="#0072B2", lw=1.6, zorder=2,
        label=f"fit: slope {f['slope']:.2f}, intercept {f['intercept']:+.2f}")
ax.axhline(0, color="0.9", lw=1, zorder=0)
ax.axvline(0, color="0.9", lw=1, zorder=0)
ax.errorbar(P, O, yerr=[O - LO, HI - O], fmt="none", ecolor="0.75", lw=1, zorder=3)
COL = {2: "#0072B2", 3: "#009E73", 4: "#E69F00", 6: "#CC79A7"}
for w in sorted(set(W.tolist())):
    m = W == w
    ax.scatter(P[m], O[m], s=54, color=COL.get(w, "0.4"), edgecolor="white",
               linewidth=0.8, zorder=4, label=f"block width W = {w}")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("predicted effect  =  projection of the write onto the target trajectory")
ax.set_ylabel("measured effect  (fraction of the full-schedule effect)")
ax.set_title("A block-constant write moves the model by exactly its projection\n"
             "onto the target — including when it opposes the target\n"
             f"{len(P)} random (profile, width, coefficient) conditions · "
             f"R² = {f['r2']:.2f} · mean |error| = {f['mean_abs_err']:.3f}",
             fontsize=10.5)
ax.legend(fontsize=8.5, frameon=True, edgecolor="0.85", loc="upper left")
ax.grid(color="0.94", lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.annotate("writes that oppose the target push\nthe margin the wrong way, in proportion",
            (0.30, -0.46), fontsize=8.5, color="0.35", ha="center")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"linearity.{ext}", bbox_inches="tight")
print("saved", OUT / "linearity.png")
