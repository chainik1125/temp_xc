"""Screen figures — per-Δ recovery vs T + the lag4 order dissociation.

Reads results/screen_<model>.json (all three models) and writes:
  figs/det_gap_vs_T.png    window−token linear AUC gap per Δ-bucket vs T
                           (the card's P1 money plot axis; markers hollow
                           where cov(B,T) = 0)
  figs/lag4_order_vs_T.png lag4 accuracy vs T for the MLP pair —
                           ordered window vs context-shuffled vs
                           per-token (the P4 order receipt; MLP cells
                           present at T ∈ {8,32} + escalation T's)

Run: .venv/bin/python -m experiments.explorations.task_hunt.replag.render_screen
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIGS = HERE / "results", HERE / "figs"
KEYS = ["gpt2", "gemma2_2b", "llama31_8b"]
NICE = {"gpt2": "GPT-2 small (124M)", "gemma2_2b": "Gemma-2-2B base",
        "llama31_8b": "Llama-3.1-8B base"}
T_GRID = [2, 4, 8, 16, 32]
BUCKETS = ["B4", "B8", "B16", "B32"]
# dataviz reference palette, categorical slots 1-4 (documented order)
C = {"B4": "#2a78d6", "B8": "#eb6834", "B16": "#1baf7a", "B32": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load():
    return {k: json.loads((RES / f"screen_{k}.json").read_text())
            for k in KEYS}


def det_gap_fig(data):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True,
                             facecolor="#fcfcfb")
    for ax, k in zip(axes, KEYS):
        d, cov = data[k]["cells"], data[k]["meta"]["coverage"]
        for b in BUCKETS:
            t = f"det{b[1:]}"
            tok = d[f"{t}/tok_linear"]["auc"]
            gaps = [d[f"{t}/T{T}/win_linear"]["auc"] - tok for T in T_GRID]
            ax.plot(range(len(T_GRID)), gaps, "-", color=C[b], lw=2,
                    zorder=3)
            for i, T in enumerate(T_GRID):
                covered = cov[b][f"T{T}"] > 0
                ax.plot(i, gaps[i], "o", ms=6, color=C[b], zorder=4,
                        mfc=C[b] if covered else "#fcfcfb", mew=1.6)
        ax.axhline(0, color=INK2, lw=1, ls="--", zorder=2)
        ax.set_xticks(range(len(T_GRID)),
                      [f"T={T}" for T in T_GRID])
        ax.set_title(NICE[k], fontsize=10, color=INK)
        ax.set_xlabel("window size T", fontsize=9, color=INK2)
        style(ax)
    axes[0].set_ylabel("window − per-token linear AUC", fontsize=9,
                       color=INK2)
    labels = {"B4": "Δ∈[2,4]", "B8": "Δ∈[5,8]",
              "B16": "Δ∈[9,16]", "B32": "Δ∈[17,32]"}
    handles = [plt.Line2D([], [], color=C[b], lw=2, marker="o", ms=6,
                          label=labels[b]) for b in BUCKETS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Repetition-lag detection: window−token gap vs T "
                 "(hollow = bucket not covered at this T)",
                 fontsize=11, color=INK, y=1.14)
    fig.tight_layout()
    fig.savefig(FIGS / "det_gap_vs_T.png", dpi=200, bbox_inches="tight",
                facecolor="#fcfcfb")
    plt.close(fig)


def lag4_fig(data):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True,
                             facecolor="#fcfcfb")
    series = [("win_mlp", "#2a78d6", "ordered window (MLP)"),
              ("win_shuf_mlp", "#eb6834", "context-shuffled (MLP)")]
    for ax, k in zip(axes, KEYS):
        d = data[k]["cells"]
        for cell, col, lab in series:
            pts = [(i, d[f"lag4/T{T}/{cell}"]["acc_test"])
                   for i, T in enumerate(T_GRID)
                   if f"lag4/T{T}/{cell}" in d]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                    color=col, lw=2, ms=6, zorder=3, label=lab)
        tok = d["lag4/tok_mlp"]["acc_test"]
        ax.axhline(tok, color=INK2, lw=1.6, ls=":", zorder=2)
        ax.text(0.02, tok, "per-token (MLP)", fontsize=8, color=INK2,
                va="bottom", transform=ax.get_yaxis_transform())
        ax.axhline(0.25, color=GRID, lw=1, zorder=1)
        ax.text(0.02, 0.25, "chance", fontsize=8, color=INK2,
                va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xticks(range(len(T_GRID)), [f"T={T}" for T in T_GRID])
        ax.set_title(NICE[k], fontsize=10, color=INK)
        ax.set_xlabel("window size T", fontsize=9, color=INK2)
        style(ax)
    axes[0].set_ylabel("lag-bucket accuracy (4-class)", fontsize=9,
                       color=INK2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Reading the lag VALUE needs order: shuffle collapses "
                 "the window advantage to the per-token ceiling",
                 fontsize=11, color=INK, y=1.14)
    fig.tight_layout()
    fig.savefig(FIGS / "lag4_order_vs_T.png", dpi=200,
                bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    data = load()
    det_gap_fig(data)
    lag4_fig(data)
    print(f"-> {FIGS}/det_gap_vs_T.png, {FIGS}/lag4_order_vs_T.png")


if __name__ == "__main__":
    main()
