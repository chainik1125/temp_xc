"""Staged-refusal figure: the real-behavior transfer.

A. Teacher-forced k-sweep, four arms — the scheduled handle works, a constant write
   and a random direction do nothing.
B. Behavioural check (calibrated candidate choice): fraction of slots where steering
   moved the model's refuse/comply choice in the intended direction.

    uv run python scripts/plot_stance.py
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "plots/2026-07-24_trajectory_steering"
S = json.load(open(ROOT / "results/temporal_screen/stance.json"))
C = json.load(open(ROOT / "results/temporal_screen/controls.json"))

ARMS = [("template", "#0072B2", "o", "scheduled handle (per-segment)"),
        ("single", "#009E73", "s", "single segment"),
        ("broadcast", "#E69F00", "^", "constant write (per-token recipe)"),
        ("random_template", "#999999", "d", "random direction, matched magnitude")]

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=150)
ax = axes[0]
ks = sorted(S["k_sweep"], key=int)
kk = [int(x) for x in ks]
for arm, color, mk, label in ARMS:
    ys = [S["k_sweep"][k][arm]["mean"] for k in ks]
    es = [S["k_sweep"][k][arm]["sem"] for k in ks]
    ax.errorbar(kk, ys, yerr=es, color=color, marker=mk, ms=6, lw=2, capsize=3,
                label=label)
ax.axhline(0, color="0.75", lw=1)
ax.set_xticks(kk)
ax.set_xlabel("k — sentences in the response (one \"slot\" each)")
ax.set_ylabel("Δ log-prob margin toward the intended stance order\n(steered − unsteered, summed over the response)", fontsize=9)
ax.set_title("A. Steering the refuse/comply order within one response\n"
             "(target and foil are the same sentences, reordered)", fontsize=10.5)
ax.legend(fontsize=8, frameon=True, edgecolor="0.85", loc="lower right",
          framealpha=0.96)
ax.grid(axis="y", color="0.92", lw=0.8)

ax2 = axes[1]
sc = C["stance_calibrated"]
arms = ["template", "broadcast", "single"]
labels = ["scheduled\nhandle", "constant\nwrite", "single\nsegment"]
colors = ["#0072B2", "#E69F00", "#009E73"]
vals = [sc[f"{a}@0.5"]["frac_correct_direction"] for a in arms]
bars = ax2.bar(np.arange(3), vals, 0.55, color=colors)
ax2.axhline(0.5, color="0.35", lw=1.4, ls="--")
ax2.annotate("0.5 — where a constant write is expected to land:\nit is right on exactly "
             "the half of slots whose\nintended stance matches its sign",
             (0.62, 0.735), fontsize=7.5, color="0.35", ha="left")
ax2.annotate("", xy=(1.42, 0.505), xytext=(1.18, 0.715),
             arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9))
ax2.axhline(1 / 8, color="0.7", lw=1, ls=":")
ax2.annotate("1/8 — ceiling for a write that touches\nonly one of the eight slots",
             (1.62, 0.245), fontsize=7.5, color="0.45", ha="left")
ax2.annotate("", xy=(2.0, 0.135), xytext=(1.9, 0.235),
             arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9))
for b, v in zip(bars, vals):
    ax2.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                 textcoords="offset points", xytext=(0, 4), fontsize=10,
                 fontweight="bold")
ax2.set_xticks(np.arange(3))
ax2.set_xticklabels(labels)
ax2.set_ylim(0, 1.08)
ax2.set_ylabel("fraction of sentence slots whose declining-vs-helping\npreference moved the intended way", fontsize=9)
ax2.set_title("B. Behavioural check: does the model's own choice follow?\n"
              "(held-out candidates, model's intrinsic preference differenced out)",
              fontsize=10.5)
ax2.grid(axis="y", color="0.92", lw=0.8)
ax2.set_axisbelow(True)
for a in (ax, ax2):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

fig.suptitle("Ordering declining and helping inside one response: the useful control is a "
             "schedule, not a level\nQwen-2.5-1.5B, layer 14 · target and foil hold the "
             "SAME sentences in a different order · bars = SEM, n = 32 pairs (A), "
             "160 slots (B)", fontsize=9.5, y=1.02)
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"stance.{ext}", bbox_inches="tight")
print("saved", OUT / "stance.png")
