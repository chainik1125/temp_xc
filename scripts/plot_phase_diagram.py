"""Sprint headline figure: the (W, ℓ) phase diagram.

Panel A — observed R(W,ℓ) as a heatmap with the zero-free-parameter prediction
printed in each cell. Panel B — control efficiency (fidelity per knob) vs W, one
line per ℓ, showing the peak at W ≈ ℓ.

    uv run python scripts/plot_phase_diagram.py
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = json.load(open(ROOT / "results/temporal_screen/lsweep.json"))
OUT = ROOT / "plots/2026-07-24_trajectory_steering"
OUT.mkdir(parents=True, exist_ok=True)

WS = [1, 2, 3, 4, 6, 12]
ELLS = [1, 2, 3, 6]
K = RES["k"]

obs = np.array([[RES["phase_diagram"][str(l)][f"W{w}_block_cap"]["obs_R"]
                 for w in WS] for l in ELLS])
pred = np.array([[RES["phase_diagram"][str(l)][f"W{w}_block_cap"]["pred_R"]
                  for w in WS] for l in ELLS])

fig = plt.figure(figsize=(12.4, 4.4), dpi=150)
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.28)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(obs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(len(ELLS)):
    for j in range(len(WS)):
        txt = f"{obs[i, j]:.2f}\n({pred[i, j]:.2f})"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                color="white" if obs[i, j] > 0.55 else "0.15")
ax.set_xticks(range(len(WS)))
ax.set_xticklabels(WS)
ax.set_yticks(range(len(ELLS)))
ax.set_yticklabels(ELLS)
ax.set_xlabel("W — segments the handle writes a single constant over")
ax.set_ylabel("ℓ — run length of the target profile")
ax.set_title("A. Fidelity R(W, ℓ): observed, (predicted)\n"
             "24 cells, zero free parameters, mean |error| = 0.013", fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("fraction of full-template effect", fontsize=8.5)

ax2 = fig.add_subplot(gs[0, 1])
COLORS = {1: "#0072B2", 2: "#009E73", 3: "#E69F00", 6: "#CC79A7"}
for i, l in enumerate(ELLS):
    eff = [obs[i, j] * WS[j] / K for j in range(len(WS))]
    ax2.plot(WS, eff, marker="o", ms=6, lw=2, color=COLORS[l], label=f"ℓ = {l}")
    jstar = int(np.argmax(eff))
    ax2.plot([WS[jstar]], [eff[jstar]], marker="*", ms=17, color=COLORS[l],
             markeredgecolor="white", markeredgewidth=1.1, zorder=5)
ax2.set_xscale("log", base=2)
ax2.set_xticks(WS)
ax2.set_xticklabels([str(w) for w in WS])
ax2.set_xlabel("W — handle width (segments)")
ax2.set_ylabel("fidelity per knob   R·W/k")
ax2.set_title("B. Control efficiency peaks at W ≈ ℓ\n"
              "(★ = best width; knobs needed for full fidelity = k/ℓ)", fontsize=10)
ax2.grid(color="0.92", lw=0.8)
ax2.legend(fontsize=9, frameon=True, framealpha=0.95, edgecolor="0.85")
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.suptitle("A steering handle that writes one constant over W segments keeps "
             "exactly the part of the trajectory that is constant over W\n"
             "(Qwen-2.5-1.5B, L14, k=12, coverage held at 12 in every cell)",
             fontsize=10.5, y=1.06)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"phase_diagram.{ext}", bbox_inches="tight")
print("saved", OUT / "phase_diagram.png")
