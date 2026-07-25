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
RES = json.load(open(ROOT / "results/temporal_screen/lsweep_qwen1.5b.json"))
RES7 = json.load(open(ROOT / "results/temporal_screen/lsweep_qwen7b.json"))
OUT = ROOT / "plots/2026-07-24_trajectory_steering"
OUT.mkdir(parents=True, exist_ok=True)

WS = [1, 2, 3, 4, 6, 12]
ELLS = [1, 2, 3, 6]
K = RES["k"]

grab = lambda R, key: np.array([[R["phase_diagram"][str(l)][f"W{w}_block_cap"][key]
                                 for w in WS] for l in ELLS])
obs, pred = grab(RES, "obs_R"), grab(RES, "pred_R")
obs7 = grab(RES7, "obs_R")

fig = plt.figure(figsize=(16.2, 4.4), dpi=150)
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 0.85], wspace=0.3)

def cell_kind(ell, W):
    """measured | full-template-by-construction | structural zero (no write)."""
    pi = np.array([1.0 if (t // ell) % 2 == 0 else -1.0 for t in range(K)])
    mu = np.array([pi[b * W:(b + 1) * W].mean() for b in range(K // W)])
    if np.all(np.sign(mu) == 0):
        return "zero"
    return "identity" if W == 1 else "measured"


ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(obs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i, ell in enumerate(ELLS):
    for j, W in enumerate(WS):
        kind = cell_kind(ell, W)
        txt = f"{obs[i, j]:.2f}\n({pred[i, j]:.2f})"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                color="white" if obs[i, j] > 0.55 else "0.15")
        if kind != "measured":
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       hatch="///", edgecolor="0.55",
                                       linewidth=0.0, alpha=0.55))
ax.plot([], [], marker="s", ls="", color="0.55", markersize=9,
        label="fixed by construction")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.30), fontsize=8,
          frameon=False, handletextpad=0.5)
ax.set_xticks(range(len(WS)))
ax.set_xticklabels(WS)
ax.set_yticks(range(len(ELLS)))
ax.set_yticklabels(ELLS)
ax.set_xlabel("W — segments the handle writes a single constant over")
ax.set_ylabel("ℓ — run length of the target profile")
ax.set_title("A. Fidelity R(W, ℓ): observed, (predicted)\n"
             "11 measured cells, no fitted parameters, mean |error| = 0.029",
             fontsize=10)
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

ax3 = fig.add_subplot(gs[0, 2])
jit = np.linspace(-0.012, 0.012, len(WS))[None, :] * np.ones((len(ELLS), 1))
ax3.plot([0, 1], [0, 1], color="0.6", lw=1.2, ls="--", zorder=1)
msk = np.array([[cell_kind(l, w) == "measured" for w in WS] for l in ELLS])
ax3.scatter((pred + jit)[msk], obs[msk], s=52, color="#0072B2",
            edgecolor="white", linewidth=0.7, zorder=3, label="Qwen-2.5-1.5B")
ax3.scatter((pred - jit)[msk], obs7[msk], s=52, marker="^", color="#CC79A7",
            edgecolor="white", linewidth=0.7, zorder=3, label="Qwen-2.5-7B")
ax3.set_xlabel("predicted R (no fitted parameters)")
ax3.set_ylabel("measured R")
ax3.set_title("C. Same law in both models\n"
              "6 fractional cells: mean |error| 0.053 / 0.045", fontsize=10)
ax3.set_xlim(-0.06, 1.06)
ax3.set_ylim(-0.06, 1.06)
ax3.grid(color="0.92", lw=0.8)
ax3.legend(fontsize=8.5, frameon=True, framealpha=0.95, edgecolor="0.85",
           loc="upper left")
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)

fig.suptitle("A steering handle that writes one constant over W segments keeps "
             "exactly the part of the trajectory that is constant over W\n"
             "(layer 14, k=12, coverage held at 12 segments in every cell)",
             fontsize=10.5, y=1.06)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"phase_diagram.{ext}", bbox_inches="tight")
print("saved", OUT / "phase_diagram.png")
