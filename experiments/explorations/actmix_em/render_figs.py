"""ACTMIX P2 — the T-sweep exhibit figure (CARD § 5.2).

House conventions (Han's figs 1–2 / diafaces/make_fig4.py): Okabe-Ito,
log2-T x-axis, per-token baselines as flat bands, untrained floor,
matplotlib Agg. Reads results/table.json (written by analyze.py — the
single mechanical reader of the leaderboard).

One panel per seed + a mean panel: TXC-post btk-only vs T (solid,
orange), its within-window-shuffled twin (dashed, same hue), SAE band
(blue), TSAE band (green), untrained TXC floor (gray dotted), chance
= positive rate 0.323 (thin gray).

Run: .venv/bin/python -m experiments.explorations.actmix_em.render_figs
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
TABLE = HERE / "results" / "table.json"
OUT = HERE / "figs"

C_TXC, C_SAE, C_TSAE, C_UNTR = "#D55E00", "#0072B2", "#009E73", "#7f7f7f"
T_GRID = (1, 2, 4, 8, 16)
SEEDS = (42, 1)
POS_RATE = 0.323
TXC, SAE, TSAE = ("txc_batchtopk_post_btkonly", "batchtopk_sae_btkonly",
                  "tsae_btkonly")


def _cells():
    d = json.loads(TABLE.read_text())["cells"]
    out = {}
    for k, v in d.items():
        arch, T, kind, seed = k.split("|")
        out[(arch, int(T), kind, int(seed))] = v
    return out


def _series(cells, seed, arch, kind, field):
    def one(s):
        vals = {}
        for T in T_GRID:
            c = cells.get((arch, T, kind, s))
            if c and c.get(field) is not None:
                vals[T] = c[field]
        return vals
    if seed != "mean":
        return one(seed)
    per = [one(s) for s in SEEDS]
    out = {}
    for T in T_GRID:
        vs = [p[T] for p in per if T in p]
        if vs:
            out[T] = sum(vs) / len(vs)
    return out


def panel(ax, cells, seed):
    x = {T: np.log2(T) for T in T_GRID}

    def plot_line(vals, **kw):
        Ts = sorted(vals)
        ax.plot([x[T] for T in Ts], [vals[T] for T in Ts], **kw)

    txc = _series(cells, seed, TXC, "trained", "pr_auc_S16")
    if txc:
        plot_line(txc, color=C_TXC, marker="o", ms=5, lw=2,
                  label="TXC-post btk-only (20·T/win)")
    sh = _series(cells, seed, TXC, "trained", "pr_auc_shuffled_S16")
    if sh:
        plot_line(sh, color=C_TXC, marker="o", ms=4, lw=1.6, ls="--",
                  alpha=0.75, label="TXC, within-window shuffled")
    unt = _series(cells, seed, TXC, "untrained", "pr_auc_S16")
    if unt:
        plot_line(unt, color=C_UNTR, marker=".", lw=1.4, ls=":",
                  label="untrained TXC twin")

    for arch, color, name in ((SAE, C_SAE, "SAE btk-only (per-token)"),
                              (TSAE, C_TSAE, "T-SAE btk-only (per-token)")):
        band = _series(cells, seed, arch, "trained", "pr_auc_S16")
        if band:
            v = band[1]
            ax.axhline(v, color=color, lw=1.8, label=name)
        ub = _series(cells, seed, arch, "untrained", "pr_auc_S16")
        if ub:
            ax.axhline(ub[1], color=color, lw=1.0, ls=":", alpha=0.5)

    ax.axhline(POS_RATE, color="#bbbbbb", lw=0.8, zorder=0)
    ax.text(0.02, POS_RATE + 0.004, "chance (positive rate)",
            fontsize=7, color="#888888")
    ax.set_xticks([np.log2(T) for T in T_GRID])
    ax.set_xticklabels([str(T) for T in T_GRID])
    ax.set_xlabel("window length T")
    ax.set_title(f"seed {seed}" if seed != "mean"
                 else f"mean over seeds {SEEDS}", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)


def main():
    cells = _cells()
    OUT.mkdir(exist_ok=True)
    panels = [42, 1, "mean"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, seed in zip(axes, panels):
        panel(ax, cells, seed)
    axes[0].set_ylabel("detection PR-AUC (S = 16)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("EM § 5.3 medical L15 — btk-only arm: detection vs "
                 "window length, with the missing shuffle control",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"actmix_em_tsweep.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"[figs] -> {OUT}/actmix_em_tsweep.{{png,pdf}}")


if __name__ == "__main__":
    main()
