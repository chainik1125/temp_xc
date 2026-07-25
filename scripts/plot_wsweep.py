"""Centerpiece figure: steering performance vs WINDOW SIZE at fixed knob budget.

Reads results/temporal_screen/wsweep.json. Two panels (lang_profile, alt_phase):
peak Δmargin vs W for m=1 and m=2 knobs, with the additive prediction
Δ_full·min(mW,k)/k as dashed lines, plus full-template and broadcast references.

    uv run python scripts/plot_wsweep.py
"""

import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = json.load(open(ROOT / "results/temporal_screen/wsweep.json"))
OUT = ROOT / "plots/2026-07-24_trajectory_steering"
OUT.mkdir(parents=True, exist_ok=True)
K = RES["k"]
WS = [1, 2, 3, 4, 6, 12]

SERIES = [(1, "#0072B2", "o", "1 knob"), (2, "#009E73", "s", "2 knobs")]
PANELS = [("lang_profile", "lang_profile — EN/FR random profile (k=12)"),
          ("alt_phase", "alt_phase — tense/calm alternation (k=12)")]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=150)
for ax, (task, title) in zip(axes, PANELS):
    d = RES[task]
    full, bcast = d["full"]["mean"], d["broadcast"]["mean"]
    for m, color, marker, label in SERIES:
        pts = [(w, d[f"W{w}_m{m}"]) for w in WS if f"W{w}_m{m}" in d]
        xs = [w for w, _ in pts]
        ys = [p["mean"] for _, p in pts]
        es = [p["sem"] for _, p in pts]
        ax.errorbar(xs, ys, yerr=es, color=color, marker=marker, ms=6, lw=2,
                    capsize=3, label=f"window-W handle, {label}", zorder=4)
        pred = [full * min(m * w, K) / K for w in xs]
        ax.plot(xs, pred, color=color, lw=1.2, ls="--", alpha=0.6, zorder=2,
                label=f"additive prediction, {label}")
    ax.axhline(full, color="0.25", lw=1.2, ls=":", zorder=1)
    ax.annotate("full template", (WS[0], full), textcoords="offset points",
                xytext=(2, 5), fontsize=8.5, color="0.25")
    ax.axhline(bcast, color="#E69F00", lw=1.6, ls="-", zorder=1)
    ax.annotate("per-token broadcast (all 12 segments)", (2.6, bcast),
                textcoords="offset points", xytext=(0, -13), fontsize=8.5,
                color="#B87A00", fontweight="bold")
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("W — window size of one steering knob (segments)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(WS)
    ax.set_xticklabels([str(w) for w in WS])
    ax.grid(axis="y", color="0.92", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("peak Δmargin toward target trajectory")
axes[0].legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.95,
               edgecolor="0.85")
fig.suptitle("Steering performance grows with window size at fixed control budget "
             "(Qwen-2.5-1.5B, L14, mean ± SEM, n=32)", fontsize=11, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"wsweep_dmargin.{ext}", bbox_inches="tight")
print("saved", OUT / "wsweep_dmargin.png")
