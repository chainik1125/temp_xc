"""Visual summary for Phase 2 autointerp:
    - confidence distribution bar chart per category
    - concrete feature example cards (one HIGH-confidence per category with 3 text examples)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

AUTOINTERP = Path("/home/elysium/temp_xc/results/analysis/autointerp")
OUT_DIR = Path("/home/elysium/temp_xc/results/analysis/autointerp")

CATS = [
    ("stacked_unique", "Stacked SAE unique", "#1f77b4"),
    ("txcdr_unique", "TXCDR unique", "#ff7f0e"),
    ("tfa_pred_only", "TFA pred-only", "#d62728"),
    ("tfa_novel_only", "TFA novel-only", "#2ca02c"),
]


def load_summary(cat: str):
    with open(AUTOINTERP / cat / "_summary.json") as f:
        return json.load(f)


def load_feature(cat: str, fi: int) -> dict:
    return json.load(open(AUTOINTERP / cat / f"feat_{fi:05d}.json"))


def make_confidence_chart():
    """Stacked bar chart: HIGH/MED/LOW per category."""
    data = {}
    for cat, _, _ in CATS:
        summary = load_summary(cat)
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for d in summary:
            c = d.get("confidence", "LOW")
            if c in counts:
                counts[c] += 1
        data[cat] = counts

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    labels = [c[1] for c in CATS]
    highs = [data[c[0]]["HIGH"] for c in CATS]
    meds = [data[c[0]]["MEDIUM"] for c in CATS]
    lows = [data[c[0]]["LOW"] for c in CATS]

    x = np.arange(len(labels))
    ax.bar(x, highs, color="#2ca02c", label="HIGH", edgecolor="black", linewidth=0.5)
    ax.bar(x, meds, bottom=highs, color="#ff7f0e", label="MEDIUM", edgecolor="black", linewidth=0.5)
    ax.bar(x, lows, bottom=np.array(highs) + np.array(meds), color="#d62728", label="LOW",
           edgecolor="black", linewidth=0.5)

    # Annotate with percentages
    for i, (h, m, l) in enumerate(zip(highs, meds, lows)):
        total = h + m + l
        if total == 0:
            continue
        pct = (h + m) / total * 100
        ax.annotate(f"{int(pct)}% coherent\n(n={total})", (i, total + 1),
                    ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("# features")
    ax.set_title("Autointerp confidence distribution per category\n"
                 "(% coherent = HIGH + MEDIUM; higher = more interpretable features)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 35)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confidence_summary.png", dpi=140)
    plt.close()
    print(f"  -> {OUT_DIR / 'confidence_summary.png'}")


def truncate(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    if "." in s[:n]:
        return s[:s[:n].rfind(".") + 1]
    return s[:n] + "…"


def make_feature_examples_card():
    """Grid: one HIGH-confidence feature per category with 3 activation examples."""
    # Pick the most interesting HIGH-confidence feature per category
    picks = {
        "stacked_unique": 9426,     # "motor" detector
        "txcdr_unique":   6655,     # function words
        "tfa_pred_only":  15410,    # second digit of HH:MM
        "tfa_novel_only": 4826,     # opening quotation marks
    }

    fig, axes = plt.subplots(4, 1, figsize=(13, 8))
    for ax, (cat, title, color) in zip(axes, CATS):
        fi = picks[cat]
        record = load_feature(cat, fi)
        expl = record["explanation"]
        examples = record["examples"][:3]

        # Background color band
        ax.set_facecolor("#fafafa")

        # Header
        header = f"{title}   ·   feature {fi}   ·   {record.get('confidence', '?')} confidence"
        ax.text(0.01, 0.95, header, transform=ax.transAxes, fontsize=11,
                fontweight="bold", color=color, va="top")
        # Explanation
        ax.text(0.01, 0.78, f'"{truncate(expl, 160)}"', transform=ax.transAxes,
                fontsize=9.5, style="italic", va="top", color="#333", wrap=True)

        # Examples
        y = 0.55
        for i, ex in enumerate(examples):
            act = ex["activation"]
            text = ex["text"]
            # Highlight peak token with bold guillemets (already in text as «token»)
            ax.text(0.01, y, f"  ({act:.1f})  {text!r}", transform=ax.transAxes,
                    fontsize=8.5, va="top", family="monospace", color="#111")
            y -= 0.18

        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5); spine.set_color("#888")

    fig.suptitle("Representative features per architecture — concrete examples",
                 fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT_DIR / "feature_examples.png", dpi=140)
    plt.close()
    print(f"  -> {OUT_DIR / 'feature_examples.png'}")


if __name__ == "__main__":
    make_confidence_chart()
    make_feature_examples_card()
