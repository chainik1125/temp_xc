"""Plain-language figures for the report. No internal jargon in any label.

Three figures:
  1. headline   — at the model's input layer, can each kind of reader answer the
                  question? (one token / whole window simple / whole window flexible)
  2. depth      — how far into the model does a single token stay unable to answer?
  3. coverage   — which of the ten candidate tasks were actually tested

Every number comes from results/gate_*.json. Light and dark variants are emitted
because an embedded PNG cannot follow the page theme.

Palette: the dataviz skill's validated reference instance (categorical slots
1/2/3/7), used unchanged — `node` is unavailable on this pod, so the validated
defaults are used rather than an eyeballed custom palette.

Run:  .venv/bin/python -m experiments.explorations.relational.report_figs
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figs"

# task -> (plain-language name, short question the label asks)
TASKS = {
    "agreement": ("Grammar agreement",
                  "does the verb match the subject in number?"),
    "contradiction": ("Fact consistency",
                      "do the two statements agree?"),
    "role": ("Text provenance",
             "is this text the user's instruction, or quoted material?"),
    "parity": ("Nesting structure",
               "were the last two markers the same kind?"),
}
ORDER = ["parity", "agreement", "contradiction", "role"]

READERS = [
    ("per_token", "Reader A — sees ONE token", "s1"),
    ("window_flat", "Reader B — sees the whole window, simple readout", "s2"),
    ("window_mlp", "Reader C — sees the whole window, flexible readout", "s3"),
]

THEME = {
    "light": {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#e4e6e8", "s1": "#2a78d6", "s2": "#eb6834",
              "s3": "#1baf7a", "s4": "#4a3aa7", "null": "#9aa0a6",
              "good": "#1baf7a", "bad": "#c74634", "idle": "#b9bcc0"},
    "dark": {"surface": "#111a21", "ink": "#e8eef1", "ink2": "#a7bac4",
             "grid": "#22323d", "s1": "#3987e5", "s2": "#d95926",
             "s3": "#199e70", "s4": "#9085e9", "null": "#778d99",
             "good": "#199e70", "bad": "#e0776b", "idle": "#4a5a66"},
}


def _style(mode):
    t = THEME[mode]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.labelcolor": t["ink"], "axes.edgecolor": t["grid"],
        "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "grid.color": t["grid"], "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })
    return t


def _load():
    cells = {}
    for f in sorted(RESULTS.glob("gate_*.json")):
        pl = json.loads(f.read_text())
        cells.setdefault(pl["meta"]["task"], []).extend(
            [c for c in pl["cells"] if "per_token" in c and c["stratum"] == "all"])
    return cells


def fig_headline(mode: str) -> Path:
    """At the input layer: who can answer the question, and who cannot."""
    t = _style(mode)
    data = _load()
    fig, ax = plt.subplots(figsize=(10.4, 5.0), dpi=170)
    width = 0.26
    xs = np.arange(len(ORDER))
    for j, (key, label, slot) in enumerate(READERS):
        vals, los, his = [], [], []
        for task in ORDER:
            L0 = [c for c in data.get(task, []) if c["layer"] == 0]
            best = max(L0, key=lambda c: c["nonlinear_residual"]) if L0 else None
            v = best[key]["value"] if best else np.nan
            vals.append(v)
            los.append(v - best[key]["ci_lo"] if best else 0)
            his.append(best[key]["ci_hi"] - v if best else 0)
        pos = xs + (j - 1) * (width + 0.015)
        ax.bar(pos, vals, width, color=t[slot], label=label, zorder=3,
               edgecolor=t["surface"], linewidth=1.6)
        ax.errorbar(pos, vals, yerr=[los, his], fmt="none", ecolor=t["ink2"],
                    elinewidth=1.1, capsize=3, zorder=4)
        for x, v, hi in zip(pos, vals, his):
            ax.annotate(f"{v:.2f}", xy=(x, v + hi), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=t["ink2"])
    ax.axhline(0.5, color=t["null"], linestyle=(0, (4, 3)), linewidth=1.3, zorder=2)
    ax.annotate("coin flip — 0.50", xy=(3.42, 0.5), xytext=(0, 7),
                textcoords="offset points", fontsize=8.5, color=t["ink2"],
                ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [TASKS[k][0] + "\n" + textwrap.fill(TASKS[k][1], 26) for k in ORDER],
        fontsize=9)
    ax.set_ylabel("Accuracy at telling the two cases apart\n(1.00 = perfect, 0.50 = coin flip)")
    ax.set_ylim(0.4, 1.09)
    ax.grid(axis="y", linewidth=0.7)
    ax.set_title("At the model's input layer, only a reader that combines positions "
                 "flexibly can answer", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncols=1,
              fontsize=9)
    fig.tight_layout()
    out = FIGS / f"report_headline_{mode}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_depth(mode: str) -> Path:
    """How deep does a single token stay unable to answer? (Answer: barely at all.)"""
    t = _style(mode)
    data = _load()
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=170)
    slots = {"parity": "s4", "agreement": "s1", "contradiction": "s2", "role": "s3"}
    for task in ORDER:
        cells = data.get(task, [])
        by_layer = {}
        for c in cells:
            if c["layer"] not in by_layer or c["T"] > by_layer[c["layer"]]["T"]:
                by_layer[c["layer"]] = c
        Ls = sorted(by_layer)
        ys = [by_layer[L]["per_token"]["value"] for L in Ls]
        ax.plot(Ls, ys, color=t[slots[task]], linewidth=2.2, marker="o",
                markersize=5.5, markeredgecolor=t["surface"], markeredgewidth=1.5,
                label=TASKS[task][0], zorder=3)
    ax.axhline(0.5, color=t["null"], linestyle=(0, (4, 3)), linewidth=1.3)
    ax.annotate("coin flip — 0.50", xy=(6.5, 0.5), xytext=(0, 6),
                textcoords="offset points", fontsize=8.5, color=t["ink2"])
    ax.axvspan(-0.4, 0.4, color=t["s4"], alpha=0.10, linewidth=0)
    ax.annotate("the model's input\n(before any attention)", xy=(0.0, 0.575),
                xytext=(1.6, 0.545), fontsize=8.5, color=t["ink2"], va="center",
                arrowprops=dict(arrowstyle="->", color=t["ink2"], lw=1))
    ax.annotate("after ONE attention layer a single token\nalready answers all four questions",
                xy=(1, 1.0), xytext=(4.5, 0.80), fontsize=9, color=t["ink"],
                arrowprops=dict(arrowstyle="->", color=t["ink2"], lw=1.2))
    ax.set_xlabel("How many transformer layers the text has passed through")
    ax.set_ylabel("Accuracy of a reader that sees ONE token\n(1.00 = perfect, 0.50 = coin flip)")
    ax.set_ylim(0.44, 1.05)
    ax.set_xticks([0, 1, 2, 3, 4, 8, 16, 24])
    ax.grid(axis="y", linewidth=0.7)
    ax.set_title("The model solves these puzzles almost immediately, then carries "
                 "the answer on every token", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIGS / f"report_depth_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


COVERAGE = [
    ("Nesting structure (was 1b)", "TESTED", "killed — but proved the theory"),
    ("Grammar agreement (was 5)", "TESTED", "killed by early conversion"),
    ("Fact consistency (was 4)", "TESTED", "killed by early conversion"),
    ("Text provenance (was 1a)", "TESTED", "killed; also turned out additive"),
    ("Will the model obey an injected instruction? (3)", "NOT TESTED",
     "needs text generation — the most promising untested one"),
    ("Did the model do what it was told? (2)", "NOT TESTED", "needs new templates"),
    ("Refusal that flips to compliance (6)", "NOT TESTED", "parked: order, not equality"),
    ("Winograd pronoun reference (7)", "NOT TESTED", "parked: likely already solved per token"),
    ("Grammatical islands / filler-gap (8)", "NOT TESTED", "parked: continuously useful to the model"),
    ("Who is speaking, tags removed (9)", "NOT TESTED", "parked: speaker ≈ turn parity"),
    ("Multi-turn jailbreak escalation (10)", "NOT TESTED", "dropped: needs a paid judge"),
]


def fig_coverage(mode: str) -> Path:
    t = _style(mode)
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=170)
    ys = np.arange(len(COVERAGE))[::-1]
    for y, (name, status, note) in zip(ys, COVERAGE):
        done = status == "TESTED"
        ax.barh(y, 1.0 if done else 0.0, height=0.62, color=t["s4"] if done else t["idle"],
                edgecolor=t["surface"], linewidth=1.4, zorder=3)
        if not done:
            ax.barh(y, 1.0, height=0.62, color=t["idle"], alpha=0.22,
                    edgecolor=t["surface"], linewidth=1.4, zorder=2)
        ax.annotate(name, xy=(0.02, y), va="center", fontsize=9,
                    color=t["surface"] if done else t["ink"], zorder=5)
        ax.annotate(note, xy=(1.04, y), va="center", fontsize=8.5,
                    color=t["ink2"], zorder=5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 2.5)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("Four of eleven candidate tasks were actually run — "
                 "filled bars are tested", loc="left")
    fig.tight_layout()
    out = FIGS / f"report_coverage_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_winner(mode: str) -> Path:
    """Hyper-focused: the one puzzle where the architectures genuinely separate,
    shown at the input layer and then one attention layer later."""
    t = _style(mode)
    data = _load()
    cells = data.get("parity", [])
    def pick(layer):
        c = [x for x in cells if x["layer"] == layer and x["T"] == 32]
        return c[0] if c else None
    c0, c1 = pick(0), pick(1)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0), dpi=170, sharey=True)
    for ax, c, head in zip(axes, (c0, c1),
                           ("At the model's INPUT (before any attention)",
                            "After just ONE attention layer")):
        if c is None:
            continue
        vals = [c[k]["value"] for k, _, _ in READERS]
        los = [c[k]["value"] - c[k]["ci_lo"] for k, _, _ in READERS]
        his = [c[k]["ci_hi"] - c[k]["value"] for k, _, _ in READERS]
        cols = [t[slot] for _, _, slot in READERS]
        xs = np.arange(3)
        ax.bar(xs, vals, 0.62, color=cols, zorder=3,
               edgecolor=t["surface"], linewidth=1.8)
        ax.errorbar(xs, vals, yerr=[los, his], fmt="none", ecolor=t["ink2"],
                    elinewidth=1.2, capsize=4, zorder=4)
        for x, v, hi in zip(xs, vals, his):
            ax.annotate(f"{v:.2f}", xy=(x, v + hi), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=13,
                        color=t["ink"], fontweight="600")
        ax.axhline(0.5, color=t["null"], linestyle=(0, (4, 3)), linewidth=1.3, zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Reader A\none token\n\n(ordinary SAE,\nT-SAE, MLC)",
                            "Reader B\nwindow, added up\n\n(Stacked SAE,\nTXC-pre)",
                            "Reader C\nwindow, mixed\n\n(the paper's TXC,\nTFA)"],
                           fontsize=9)
        ax.set_title(head, loc="left", fontsize=10.5)
        ax.grid(axis="y", linewidth=0.7)
    axes[0].set_ylabel("Accuracy at answering the puzzle\n(1.00 = perfect, 0.50 = coin flip)")
    axes[0].set_ylim(0.4, 1.16)
    axes[0].annotate("coin flip", xy=(2.42, 0.5), xytext=(0, 6),
                     textcoords="offset points", fontsize=8.5, color=t["ink2"],
                     ha="center")
    if c0:
        # gap is measured against the BEST additive reader (A or B), which is the
        # quantity the record reports — not simply C minus B.
        base = max(c0["window_flat"]["value"], c0["per_token"]["value"])
        gap = c0["window_mlp"]["value"] - base
        axes[0].annotate("", xy=(1.0, base), xytext=(1.0, c0["window_mlp"]["value"]),
                         arrowprops=dict(arrowstyle="<->", color=t["s4"], lw=2))
        axes[0].annotate(f"+{gap:.2f}\nover the best reader\nthat only adds up\npositions",
                         xy=(1.07, 0.74), fontsize=10.5, color=t["s4"],
                         fontweight="600", va="center")
    if c1:
        axes[1].annotate("the advantage is gone —\nevery reader is perfect",
                         xy=(1.0, 0.66), fontsize=10.5, color=t["ink2"],
                         ha="center", va="center")
    fig.suptitle("THE WINNER — \u201cwere the last two document markers the same kind?\u201d",
                 x=0.012, ha="left", fontsize=13, fontweight="600")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIGS / f"report_winner_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        for fn in (fig_winner, fig_headline, fig_depth, fig_coverage):
            print("wrote", fn(mode))


if __name__ == "__main__":
    main()
