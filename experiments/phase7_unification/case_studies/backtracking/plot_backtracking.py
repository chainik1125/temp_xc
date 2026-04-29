#!/usr/bin/env python3
"""Stage 5c: magnitude-sweep, top-feature, and Pareto coherence-vs-backtracking
plots.

Three figures, each saved with thumbnails via `src.plotting.save_figure`:

    plots/magnitude_sweep.png   keyword fraction vs steering magnitude per
                                (mode, target). Useful for diagnosing where the
                                model collapses, but the magnitude axis is
                                arbitrary so don't read causation off it.

    plots/top_features.png      bar chart of the top features in the order
                                they were ranked (delta / tstat / ratio per
                                decompose meta) with annotation of n_active.

    plots/pareto_coherence.png  *headline plot*: x = keyword fraction, y =
                                Sonnet coherence (0–3), one curve per
                                (mode, target). Magnitudes annotate the
                                points. The Pareto comparison is "what's the
                                highest backtracking rate I can get while
                                staying coherent?"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.plotting.save_figure import save_figure  # noqa: E402

from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    DECOMPOSE_DIR,
    INTERVENE_DIR,
    PLOTS_DIR,
    ensure_dirs,
)


def _load_rates() -> list[dict]:
    p = INTERVENE_DIR / "keyword_rates.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}; run evaluate_backtracking.py first")
    rows: list[dict] = []
    with p.open() as f:
        for r in csv.DictReader(f):
            r["magnitude"] = float(r["magnitude"])
            r["mean_keyword_fraction"] = float(r["mean_keyword_fraction"])
            r["sem_keyword_fraction"] = float(r.get("sem_keyword_fraction") or 0.0)
            r["n_prompts"] = int(r["n_prompts"])
            r["n_coherence_graded"] = int(r.get("n_coherence_graded") or 0)
            r["mean_coherence"] = float(r["mean_coherence"]) if r.get("mean_coherence") not in (None, "", "nan") else None
            r["sem_coherence"] = float(r["sem_coherence"]) if r.get("sem_coherence") not in (None, "", "nan") else 0.0
            rows.append(r)
    return rows


def _grouped(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    g: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        g[(r["mode"], r["target"])].append(r)
    for series in g.values():
        series.sort(key=lambda x: x["magnitude"])
    return g


def plot_magnitude_sweep(rows: list[dict], out_path: Path) -> None:
    grouped = _grouped(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab10")
    for k, ((mode, target), s) in enumerate(sorted(grouped.items())):
        xs = [r["magnitude"] for r in s]
        ys = [r["mean_keyword_fraction"] for r in s]
        es = [r["sem_keyword_fraction"] for r in s]
        marker = {"raw_dom": "o", "sae_additive": "s", "sae_clamp": "^"}.get(mode, "x")
        color = "black" if mode == "raw_dom" else cmap(k % 10)
        ax.errorbar(xs, ys, yerr=es, label=f"{mode} {target}", marker=marker, color=color, linewidth=1.4)
    ax.set_xlabel("steering magnitude (additive α  /  clamp strength s)")
    ax.set_ylabel("keyword fraction (B = {wait, hmm})")
    ax.set_title("Backtracking keyword fraction vs intervention magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_top_features(out_path: Path) -> None:
    p = DECOMPOSE_DIR / "top_features.json"
    if not p.exists():
        raise SystemExit(f"missing {p}; run Stage 3 first")
    top = json.loads(p.read_text())[:10]
    meta = json.loads((DECOMPOSE_DIR / "decompose_meta.json").read_text())
    rank_by = meta.get("rank_by", "delta")
    score_key = {"tstat": "tstat", "delta": "delta", "ratio": "ratio"}[rank_by]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = list(range(len(top)))
    scores = [r[score_key] for r in top]
    bars = ax.bar(xs, scores, color=["tab:blue" if d >= 0 else "tab:red" for d in scores])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"f={r['feature_idx']}" for r in top], rotation=45, ha="right")
    ax.set_ylabel({"tstat": "Welch t-statistic", "delta": "Δⱼ = mean(D₊) − mean(D)", "ratio": "mean(D₊) / mean(D)"}[rank_by])
    ax.set_title(f"Top-10 SAE features ranked by {rank_by} (Llama-Scope L10R-8x)")
    ax.axhline(0, color="black", linewidth=0.5)
    for i, (bar, r) in enumerate(zip(bars, top)):
        ax.text(
            i,
            bar.get_height(),
            f"  n+={r['n_active_plus']}",
            ha="center",
            va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=7,
        )
    fig.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_pareto(rows: list[dict], out_path: Path) -> None:
    grouped = _grouped(rows)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")
    plotted = 0
    for k, ((mode, target), s) in enumerate(sorted(grouped.items())):
        s = [r for r in s if r["mean_coherence"] is not None]
        if not s:
            continue
        xs = [r["mean_keyword_fraction"] for r in s]
        ys = [r["mean_coherence"] for r in s]
        x_err = [r["sem_keyword_fraction"] for r in s]
        y_err = [r["sem_coherence"] for r in s]
        marker = {"raw_dom": "o", "sae_additive": "s", "sae_clamp": "^"}.get(mode, "x")
        color = "black" if mode == "raw_dom" else cmap(k % 10)
        ax.errorbar(xs, ys, xerr=x_err, yerr=y_err, marker=marker, color=color, linewidth=1.5, label=f"{mode} {target}")
        for r, x, y in zip(s, xs, ys):
            ax.annotate(
                f"α={r['magnitude']:g}",
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
                alpha=0.7,
            )
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "No coherence grades available\nrun grade_coherence.py first",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_xlabel("backtracking rate — keyword fraction in B = {wait, hmm}")
    ax.set_ylabel("Sonnet coherence (0 = incoherent loops … 3 = fully coherent)")
    ax.set_title("Pareto: backtracking induction vs coherence (DeepSeek-R1-Distill-Llama-8B, L10)")
    ax.set_ylim(-0.1, 3.1)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    ensure_dirs()
    rows = _load_rates()
    plot_magnitude_sweep(rows, PLOTS_DIR / "magnitude_sweep.png")
    plot_top_features(PLOTS_DIR / "top_features.png")
    plot_pareto(rows, PLOTS_DIR / "pareto_coherence.png")


if __name__ == "__main__":
    main()
