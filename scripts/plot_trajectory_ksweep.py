"""Two-panel k-sweep figure for the trajectory-steering result.

Reads results/temporal_screen/trajectory_full.json (Modal run, Qwen-2.5-1.5B L14)
and plots peak teacher-forced Δmargin (mean ± SEM over 32 eval pairs) vs k for the
three arms, on lang_profile and alt_phase. Wong palette, CVD-validated triple.

    uv run python scripts/plot_trajectory_ksweep.py
"""

import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = json.load(open(ROOT / "results/temporal_screen/trajectory_full.json"))
OUT = ROOT / "plots/2026-07-24_trajectory_steering"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = [
    ("template", "#0072B2", "o", "TXC template (per-segment schedule)"),
    ("single", "#009E73", "s", "single segment"),
    ("broadcast", "#E69F00", "^", "per-token SAE broadcast"),
]
PANELS = [
    ("lang_profile", "lang_profile — EN/FR per random balanced profile"),
    ("alt_phase", "alt_phase — tense/calm alternation (clock at $\\omega=\\pi$)"),
]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
for ax, (task, title) in zip(axes, PANELS):
    data = RES[task]
    ks = sorted(int(k) for k in data)
    for arm, color, marker, label in ARMS:
        means = [data[str(k)]["curves"][arm]["peak"]["mean"] for k in ks]
        sems = [data[str(k)]["curves"][arm]["peak"]["sem"] for k in ks]
        ax.errorbar(ks, means, yerr=sems, color=color, marker=marker, ms=6,
                    lw=2, capsize=3, label=label, zorder=3)
        ax.annotate(f"{means[-1]:+.0f}", (ks[-1], means[-1]),
                    textcoords="offset points", xytext=(8, -3),
                    fontsize=9, color=color, fontweight="bold")
    ax.axhline(0, color="0.75", lw=1, zorder=1)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("k  (segments the profile spans)")
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 0.4, ks[-1] + 1.3)
    ax.grid(axis="y", color="0.92", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("peak Δmargin  logP(target) − logP(matched foil)")
axes[0].legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.95,
               edgecolor="0.85")
fig.suptitle("Trajectory steering: windowed template vs per-token broadcast "
             "(Qwen-2.5-1.5B, L14, mean ± SEM, n=32)", fontsize=11, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"ksweep_dmargin.{ext}", bbox_inches="tight")
print("saved", OUT / "ksweep_dmargin.png")
