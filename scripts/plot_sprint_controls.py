"""Two control figures for the sprint.

A. The correction: with permuted foils the effect appears to grow with trajectory
   length; with fixed-Hamming foils (H=2 at every k) it is flat. The growth was
   bookkeeping.
B. The mechanism: permuting the schedule INSIDE the block — coverage, contiguity and
   injected norm held fixed — collapses the effect. It is the order that matters,
   not the mass.

    uv run python scripts/plot_sprint_controls.py
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "plots/2026-07-24_trajectory_steering"
OUT.mkdir(parents=True, exist_ok=True)
CTRL = json.load(open(ROOT / "results/temporal_screen/controls.json"))
FULL = json.load(open(ROOT / "results/temporal_screen/trajectory_full.json"))
CVX = json.load(open(ROOT / "results/temporal_screen/convex.json"))

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=150)

# ---- A: permuted vs fixed-Hamming ----
ax = axes[0]
ks = sorted(CTRL["fixed_hamming"], key=int)
kk = [int(x) for x in ks]
fixed = [CTRL["fixed_hamming"][k]["peak"]["mean"] for k in ks]
fixed_e = [CTRL["fixed_hamming"][k]["peak"]["sem"] for k in ks]
perm = [FULL["lang_profile"][k]["curves"]["template"]["peak"]["mean"] for k in ks]
perm_e = [FULL["lang_profile"][k]["curves"]["template"]["peak"]["sem"] for k in ks]
ax.errorbar(kk, perm, yerr=perm_e, color="#E69F00", marker="^", ms=6, lw=2,
            capsize=3, label="permuted foil (differs in ~k/2 slots)")
ax.errorbar(kk, fixed, yerr=fixed_e, color="#0072B2", marker="o", ms=6, lw=2,
            capsize=3, label="fixed-Hamming foil (differs in exactly 2 slots)")
ax.annotate("+189%: tracks how many slots\nthe foil differs in",
            (kk[-2], perm[-2]), textcoords="offset points", xytext=(-118, 14),
            fontsize=8.5, color="#B87A00", fontweight="bold")
ax.annotate("flat: per-slot efficacy does not\ndecay with trajectory length",
            (kk[-1], fixed[-1]), textcoords="offset points", xytext=(-158, -30),
            fontsize=8.5, color="#0072B2", fontweight="bold")
ax.set_ylim(55, 245)
ax.set_xlabel("k — trajectory length (segments)")
ax.set_ylabel("peak Δmargin")
ax.set_xticks(kk)
ax.set_title("A. What looked like growth was bookkeeping", fontsize=10.5)
ax.legend(loc="upper left", fontsize=8.5, frameon=True, edgecolor="0.85")
ax.grid(axis="y", color="0.92", lw=0.8)

# ---- B: scramble control ----
ax2 = axes[1]
fr = "0.35"
Ws = sorted(CVX["blocks"][fr], key=int)
x = np.arange(len(Ws))
intact = [CVX["blocks"][fr][W]["mean"] for W in Ws]
intact_e = [CVX["blocks"][fr][W]["sem"] for W in Ws]
scram = [CVX["scrambled"][fr][W]["mean"] for W in Ws]
scram_e = [CVX["scrambled"][fr][W]["sem"] for W in Ws]
ax2.bar(x - 0.19, intact, 0.38, yerr=intact_e, capsize=3, color="#0072B2",
        label="correct schedule inside the block")
ax2.bar(x + 0.19, scram, 0.38, yerr=scram_e, capsize=3, color="#E69F00",
        label="same block, schedule permuted inside")
for i, (a, b) in enumerate(zip(intact, scram)):
    ax2.annotate(f"{100*b/a:.0f}%" if a > 1 else "", (i + 0.19, max(b, 0) + 1.6),
                 ha="center", fontsize=8.5, color="#B87A00", fontweight="bold")
ax2.axhline(0, color="0.6", lw=1)
ax2.set_xticks(x)
ax2.set_xticklabels([f"W={W}" for W in Ws])
ax2.set_xlabel("block width (segments written — identical across the pair)")
ax2.set_ylabel("Δmargin")
ax2.set_title("B. Same coverage, same contiguity, same injected norm —\n"
              "only the order differs", fontsize=10.5)
ax2.legend(loc="upper left", fontsize=8.5, frameon=True, edgecolor="0.85")
ax2.grid(axis="y", color="0.92", lw=0.8)
for a in (ax, ax2):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

fig.suptitle("Controls: what the steering effect is, and what it is not "
             "(Qwen-2.5-1.5B, L14)", fontsize=11, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"controls.{ext}", bbox_inches="tight")
print("saved", OUT / "controls.png")
