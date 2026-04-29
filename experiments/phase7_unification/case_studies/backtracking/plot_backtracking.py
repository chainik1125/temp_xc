#!/usr/bin/env python3
"""Stage 5b: magnitude-sweep + top-feature plots.

Two figures, both saved with thumbnails via `src.plotting.save_figure`:

    plots/magnitude_sweep.png   keyword fraction vs steering magnitude, with
                                separate lines for raw_dom and each top-K SAE
                                feature (additive + clamp). Mirrors paper Fig 3
                                (right half: reasoning model only).
    plots/top_features.png      bar chart of top-10 features by |Δ_j| with
                                feature_idx labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
        raise SystemExit(f"missing {p}; run Stage 5a first")
    rows: list[dict] = []
    with p.open() as f:
        for r in csv.DictReader(f):
            r["magnitude"] = float(r["magnitude"])
            r["mean_keyword_fraction"] = float(r["mean_keyword_fraction"])
            r["sem"] = float(r["sem"])
            r["n_prompts"] = int(r["n_prompts"])
            rows.append(r)
    return rows


def plot_magnitude_sweep(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["mode"], r["target"])].append(r)
    for series in grouped.values():
        series.sort(key=lambda x: x["magnitude"])

    # Left: additive modes (raw_dom + sae_additive)
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    if ("raw_dom", "raw_dom") in grouped:
        s = grouped[("raw_dom", "raw_dom")]
        xs = [r["magnitude"] for r in s]
        ys = [r["mean_keyword_fraction"] for r in s]
        es = [r["sem"] for r in s]
        ax.errorbar(xs, ys, yerr=es, label="raw DoM (paper baseline)", marker="o", color="black", linewidth=2.0)
    for k, ((mode, target), s) in enumerate(sorted(grouped.items())):
        if mode != "sae_additive":
            continue
        xs = [r["magnitude"] for r in s]
        ys = [r["mean_keyword_fraction"] for r in s]
        es = [r["sem"] for r in s]
        ax.errorbar(xs, ys, yerr=es, label=f"SAE add: {target}", marker="s", color=cmap(k % 10), linewidth=1.2)
    ax.set_xlabel("steering magnitude α")
    ax.set_ylabel("keyword fraction (B = {wait, hmm})")
    ax.set_title("additive: raw DoM vs SAE feature decoders")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # Right: SAE paper-clamp on each feature
    ax = axes[1]
    if ("raw_dom", "raw_dom") in grouped:
        # Re-plot raw_dom as faint reference line
        s = grouped[("raw_dom", "raw_dom")]
        xs = [r["magnitude"] for r in s]
        ys = [r["mean_keyword_fraction"] for r in s]
        ax.plot(xs, ys, label="raw DoM (additive ref.)", color="grey", linestyle=":", marker="o")
    for k, ((mode, target), s) in enumerate(sorted(grouped.items())):
        if mode != "sae_clamp":
            continue
        xs = [r["magnitude"] for r in s]
        ys = [r["mean_keyword_fraction"] for r in s]
        es = [r["sem"] for r in s]
        ax.errorbar(xs, ys, yerr=es, label=f"SAE clamp: {target}", marker="^", color=cmap(k % 10), linewidth=1.2)
    ax.set_xlabel("clamp strength")
    ax.set_title("paper-clamp on top-K SAE features")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Backtracking keyword fraction vs intervention magnitude (DeepSeek-R1-Distill-Llama-8B, L10)")
    fig.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"[plot] wrote {out_path} (+ thumb)")


def plot_top_features(out_path: Path) -> None:
    p = DECOMPOSE_DIR / "top_features.json"
    if not p.exists():
        raise SystemExit(f"missing {p}; run Stage 3 first")
    top = json.loads(p.read_text())[:10]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = list(range(len(top)))
    deltas = [r["delta"] for r in top]
    bars = ax.bar(xs, deltas, color=["tab:blue" if d >= 0 else "tab:red" for d in deltas])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"f={r['feature_idx']}" for r in top], rotation=45, ha="right")
    ax.set_ylabel("Δⱼ = mean(z_j | D₊) − mean(z_j | D)")
    ax.set_title("Top-10 SAE features by |Δⱼ| (Llama-Scope L10R-8x)")
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
    print(f"[plot] wrote {out_path} (+ thumb)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    ensure_dirs()
    rows = _load_rates()
    plot_magnitude_sweep(rows, PLOTS_DIR / "magnitude_sweep.png")
    plot_top_features(PLOTS_DIR / "top_features.png")


if __name__ == "__main__":
    main()
