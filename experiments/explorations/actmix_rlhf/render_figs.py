"""ACTMIX RLHF — the T-sweep exhibit figure (CARD § 6).

House conventions (Okabe-Ito, log2-T axis, bands for per-token,
untrained floor, shuffle overlay dashed). Reads rlhf_table.json +
papermatch.json (analyze.py runs first).

Run: .venv/bin/python -m experiments.explorations.actmix_rlhf.render_figs
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "figs"

C_TXC, C_SAE, C_TSAE, C_UNTR, C_SHIP = ("#D55E00", "#0072B2", "#009E73",
                                        "#7f7f7f", "#CC79A7")
TXC, SAE, TSAE = ("txc_batchtopk_post_btkonly", "batchtopk_sae_btkonly",
                  "tsae_btkonly")


def main():
    tbl = json.loads((RES / "rlhf_table.json").read_text())
    pm = json.loads((RES / "papermatch.json").read_text())
    cells = {}
    for k, v in tbl["btk_cells"].items():
        a, T, kk, kind, s = k.split("|")
        cells[(a, int(T), int(kk), kind, int(s))] = v

    def series(arch, kind, seed, field="preference_auc_k20"):
        out = {}
        for (a, T, k, kd, s), c in cells.items():
            if a == arch and kd == kind and s == seed and c.get(field) is not None:
                out[T] = c[field]
        return out

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = lambda T: np.log2(T)

    tr = series(TXC, "trained", 42)
    Ts = sorted(tr)
    ax.plot([x(t) for t in Ts], [tr[t] for t in Ts], color=C_TXC,
            marker="o", ms=5, lw=2,
            label="TXC-post btk-only (k_win = 100·T)")
    sh = series(TXC, "trained", 42, "shuffled_preference_auc_k20")
    Ts2 = sorted(t for t in sh if t > 1)
    if Ts2:
        ax.plot([x(t) for t in Ts2], [sh[t] for t in Ts2], color=C_TXC,
                marker="o", ms=4, lw=1.6, ls="--", alpha=0.75,
                label="TXC, within-window shuffled")
    unt = series(TXC, "untrained", 42)
    Ts3 = sorted(unt)
    if Ts3:
        ax.plot([x(t) for t in Ts3], [unt[t] for t in Ts3], color=C_UNTR,
                marker=".", lw=1.4, ls=":", label="untrained TXC twin")

    for (arch, k, color, name) in ((SAE, 500, C_SAE, "SAE btk-only k500"),
                                   (SAE, 100, C_SAE, "SAE btk-only k100"),
                                   (TSAE, 500, C_TSAE, "T-SAE btk-only k500")):
        c = cells.get((arch, 1, k, "trained", 42))
        if c and c.get("preference_auc_k20") is not None:
            ls = "-" if k == 500 else (0, (4, 2))
            ax.axhline(c["preference_auc_k20"], color=color, lw=1.6, ls=ls,
                       alpha=0.9 if k == 500 else 0.6, label=name)

    ship = pm["cells"]["agentic_txc_02"]
    p = ship["variants"]["plain"]["preference_auc"]["auc_mean"]
    s = ship["variants"]["shuffled"]["preference_auc"]["auc_mean"]
    ax.scatter([x(5)], [p], marker="D", s=55, color=C_SHIP, zorder=5,
               label="shipped agentic_txc_02 (paper-match, T=5)")
    ax.scatter([x(5)], [s], marker="D", s=40, facecolor="none",
               edgecolor=C_SHIP, zorder=5,
               label="shipped, shuffled")

    u500 = cells.get((SAE, 1, 500, "untrained", 42))
    if u500 and u500.get("preference_auc_k20") is not None:
        ax.axhline(u500["preference_auc_k20"], color=C_UNTR, lw=1.2,
                   ls="-.", alpha=0.8,
                   label="untrained k500 twins (sae ≡ tsae)")
    ax.axhline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
    ax.text(0.02, 0.503, "chance", fontsize=7, color="#888888")

    all_T = sorted(set(list(tr) + [8, 16]))
    ax.set_xticks([x(t) for t in all_T])
    ax.set_xticklabels([str(t) for t in all_T])
    ax.set_xlabel("window length T")
    ax.set_ylabel("preference AUC (top-20 diff probe, 5-fold)")
    ax.set_title("HH-RLHF § 5.4 — btk-only T-sweep + the missing shuffle "
                 "control (seed 42)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rlhf_tsweep.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"[figs] -> {OUT}/rlhf_tsweep.{{png,pdf}}")


if __name__ == "__main__":
    main()
