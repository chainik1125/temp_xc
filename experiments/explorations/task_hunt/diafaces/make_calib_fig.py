"""diafaces/make_calib_fig.py — the CALIB_CARD § 6 figure
(relu-mix vs btk-only, per arm; ACTMIX Stage 2 deliverable).

Source: `results/calib_score.json` ONLY — itself built from the
canonical leaderboard (relu-mix arm by the card's cited eval_keys) and
the pin-asserted calib panel (btk-only arm), so figure and score cannot
disagree. Conventions match figs 1–4: Okabe-Ito arch hues (validated
CVD-safe as a trio; untrained is a MUTED grey reference with linestyle
+ label, never a fourth categorical slot), arm distinguished by open
(relu-mix) vs filled (btk-only) markers — secondary encoding, color
stays on the arch entity.

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.make_calib_fig
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs"
# Okabe-Ito, fixed arch->hue (fig1-4 convention; entity-stable).
C = {"batchtopk_sae": "#0072B2", "tsae": "#009E73",
     "txc_batchtopk_post": "#D55E00"}
GREY = "#7f7f7f"
ARMS = [("batchtopk_sae", 1, "BatchTopK\nSAE @T1"),
        ("tsae", 1, "T-SAE\n@T1"),
        ("txc_batchtopk_post", 4, "TXC-post\n@T4"),
        ("txc_batchtopk_post", 16, "TXC-post\n@T16"),
        ("txc_batchtopk_post", 32, "TXC-post\n@T32")]
NOMINAL_K = 8.0
DX = 0.16                              # relu-mix left, btk-only right


def _cells(score):
    by = {}
    for c in score["cells"]:
        by[(c["arch"], c["T"], c["seed"], c["kind"])] = c
    return by


def main():
    score = json.loads((HERE / "results" / "calib_score.json").read_text())
    by = _cells(score)
    seeds = score["seeds"]
    OUT.mkdir(exist_ok=True)

    fig, (ax, axl) = plt.subplots(
        1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})

    for i, (arch, T, _lbl) in enumerate(ARMS):
        col = C[arch]
        for kind, alpha, lw in (("trained", 1.0, 1.6), ("untrained", 0.45, 1.0)):
            colk = col if kind == "trained" else GREY
            rm = [by[(arch, T, s, kind)]["relu_mix"]["recovery"] for s in seeds]
            bo = [by[(arch, T, s, kind)]["btk_only"]["recovery"] for s in seeds]
            m_rm, m_bo = sum(rm) / len(rm), sum(bo) / len(bo)
            ls = "-" if kind == "trained" else "--"
            ax.plot([i - DX, i + DX], [m_rm, m_bo], color=colk, lw=lw,
                    ls=ls, alpha=alpha, zorder=3)
            ax.plot(i - DX, m_rm, "o", mfc="white", mec=colk, ms=7,
                    alpha=alpha, zorder=4)
            ax.plot(i + DX, m_bo, "o", mfc=colk, mec=colk, ms=7,
                    alpha=alpha, zorder=4)
            for s_rm, s_bo in zip(rm, bo):     # per-seed pairs, recessive
                ax.plot([i - DX, i + DX], [s_rm, s_bo], color=colk, lw=0.6,
                        alpha=0.35 * alpha, zorder=2)
            if kind == "trained":
                ax.annotate(f"{m_bo - m_rm:+.3f}", (i + DX, m_bo),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=7.5, color="#333333")

        # Panel B: realized l0 / nominal (trained cells).
        rm_l0 = [by[(arch, T, s, "trained")]["relu_mix"]["l0"] / NOMINAL_K
                 for s in seeds]
        bo_l0 = [by[(arch, T, s, "trained")]["btk_only"]["l0"] / NOMINAL_K
                 for s in seeds]
        m_rm, m_bo = sum(rm_l0) / len(rm_l0), sum(bo_l0) / len(bo_l0)
        axl.plot([i - DX, i + DX], [m_rm, m_bo], color=col, lw=1.6, zorder=3)
        axl.plot(i - DX, m_rm, "o", mfc="white", mec=col, ms=7, zorder=4)
        axl.plot(i + DX, m_bo, "o", mfc=col, mec=col, ms=7, zorder=4)

    slopes = score["post_slope_dlog2T"]
    ax.text(0.02, 0.97,
            "post slope d(rec)/dlog$_2$T:\n"
            f"relu-mix {slopes['relu_mix']['mean']:+.4f} → "
            f"btk-only {slopes['btk_only']['mean']:+.4f}",
            transform=ax.transAxes, fontsize=7.5, va="top", color="#333333")

    axl.axhspan(6.5 / NOMINAL_K, 9.6 / NOMINAL_K, color="#F0E442", alpha=0.2,
                lw=0, zorder=0)
    axl.axhline(1.0, color="#333333", lw=0.8, ls=":", zorder=1)
    axl.text(len(ARMS) - 0.55, 1.015, "nominal k=8", fontsize=7,
             color="#333333", ha="right")
    axl.text(len(ARMS) - 0.55, 6.55 / NOMINAL_K, "card band [6.5, 9.6]",
             fontsize=6.5, color="#7a6a00", ha="right", va="bottom")

    for a, ttl, yl in ((ax, "λ recovery (v1, canonical)", "λ recovery"),
                       (axl, "realized l0 / nominal (trained)",
                        "realized l0 fraction")):
        a.set_xticks(range(len(ARMS)))
        a.set_xticklabels([lbl for _, _, lbl in ARMS], fontsize=7.5)
        a.set_title(ttl, fontsize=9)
        a.set_ylabel(yl, fontsize=8)
        a.grid(alpha=0.22, lw=0.5)
        a.tick_params(labelsize=7.5)

    handles = [
        Line2D([], [], marker="o", mfc="white", mec="#333333", ls="", ms=7,
               label="relu-mix (reused rows)"),
        Line2D([], [], marker="o", mfc="#333333", mec="#333333", ls="", ms=7,
               label="btk-only (this card)"),
        Line2D([], [], color=GREY, ls="--", lw=1.0, label="untrained control"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="center left", framealpha=0.9)

    fig.suptitle("ACTMIX calibration — ttrend (gpt2/hs7), seeds {3,4}: "
                 "relu-mix vs btk-only per arm", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        p = OUT / f"calib_relu_vs_btk.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None)
        print(f"[calib-fig] wrote {p}")


if __name__ == "__main__":
    main()
