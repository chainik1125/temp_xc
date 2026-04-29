#!/usr/bin/env python3
"""Aggregate every available `intervene<_suffix>/keyword_rates.csv` into one
Pareto coherence-vs-backtracking plot.

Walks `RESULTS_DIR` for every directory whose name starts with `intervene`,
loads its `keyword_rates.csv` (skipping any without coherence grades), and
plots one curve per (variant, mode, target). Colour-codes by variant family
so the eye can group "all top-1 SAE features", "all sum_topk", "all
multi-position", etc.

Output: `plots_summary/pareto_all.{png,thumb.png}` plus a CSV
`plots_summary/all_variants.csv` with one row per (variant, mode, target,
magnitude) for downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
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
    RESULTS_DIR,
)


def discover_variants() -> list[tuple[str, Path]]:
    """Return list of (variant_label, intervene_dir) for every available
    keyword_rates.csv. The variant label is the suffix after `intervene_`,
    or "main" for the un-suffixed directory."""
    out: list[tuple[str, Path]] = []
    for child in sorted(RESULTS_DIR.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name == "intervene":
            label = "main"
        elif name.startswith("intervene_"):
            label = name[len("intervene_"):]
        else:
            continue
        kr = child / "keyword_rates.csv"
        if not kr.exists():
            continue
        out.append((label, child))
    return out


def load_rows(variant: str, intervene_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with (intervene_dir / "keyword_rates.csv").open() as f:
        for r in csv.DictReader(f):
            r["variant"] = variant
            r["magnitude"] = float(r["magnitude"])
            r["mean_keyword_fraction"] = float(r["mean_keyword_fraction"])
            r["sem_keyword_fraction"] = float(r.get("sem_keyword_fraction") or 0.0)
            mc = r.get("mean_coherence")
            r["mean_coherence"] = float(mc) if mc not in (None, "", "nan") else None
            sc = r.get("sem_coherence")
            r["sem_coherence"] = float(sc) if sc not in (None, "", "nan") else 0.0
            rows.append(r)
    return rows


# Stable colour assignment by variant family (so similar variants cluster).
_FAMILY_COLOURS = {
    "raw_dom":   "#000000",  # always black
    "main_top1": "#1f77b4",  # blue
    "topk":      "#2ca02c",  # green
    "multipos":  "#ff7f0e",  # orange
    "32x":       "#9467bd",  # purple
    "modeldiff": "#d62728",  # red
    "ratio":     "#17becf",  # cyan
    "other":     "#7f7f7f",  # grey
}


def family_for(variant: str, mode: str, target: str) -> str:
    if mode == "raw_dom":
        return "raw_dom"
    if "modeldiff" in variant:
        return "modeldiff"
    if "ratio" in variant:
        return "ratio"
    if "32x" in variant:
        return "32x"
    if "multipos" in variant:
        return "multipos"
    if "topk" in variant or target.startswith("sum_top"):
        return "topk"
    if variant == "main" and mode == "sae_additive":
        return "main_top1"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-suffix", default="summary", help="write to plots<_suffix>/")
    parser.add_argument("--include-clamp", action="store_true",
                        help="also draw sae_clamp curves (default: skip — clamp uses different magnitude scale)")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / f"plots_{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = discover_variants()
    if not variants:
        raise SystemExit(f"no intervene*/keyword_rates.csv found under {RESULTS_DIR}")

    all_rows: list[dict] = []
    for label, idir in variants:
        all_rows.extend(load_rows(label, idir))

    csv_path = out_dir / "all_variants.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["variant", "mode", "target", "magnitude", "n_prompts",
                      "mean_keyword_fraction", "sem_keyword_fraction",
                      "n_coherence_graded", "mean_coherence", "sem_coherence"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            r2 = {**r}
            for k in ("mean_coherence", "sem_coherence"):
                if r2.get(k) is None:
                    r2[k] = ""
            w.writerow(r2)
    print(f"[summarise] wrote {csv_path} ({len(all_rows)} rows from {len(variants)} variants)")

    # Group rows into curves: (variant, mode, target).
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in all_rows:
        if not args.include_clamp and r["mode"] == "sae_clamp":
            continue
        grouped[(r["variant"], r["mode"], r["target"])].append(r)
    for series in grouped.values():
        series.sort(key=lambda x: x["magnitude"])

    # Pareto plot.
    fig, ax = plt.subplots(figsize=(11, 6.5))
    legend_seen: set[str] = set()
    for (variant, mode, target), s in sorted(grouped.items()):
        s = [r for r in s if r["mean_coherence"] is not None]
        if not s:
            continue
        family = family_for(variant, mode, target)
        colour = _FAMILY_COLOURS.get(family, "#7f7f7f")
        marker = {"raw_dom": "o", "sae_additive": "s", "sae_clamp": "^"}.get(mode, "x")
        # raw_dom from "main" is the canonical baseline; emphasise it.
        if variant == "main" and mode == "raw_dom":
            label = "raw DoM (baseline, every-position)"
            lw, ms, alpha = 2.4, 8, 1.0
        elif mode == "raw_dom":
            label = f"raw DoM ({variant})"
            lw, ms, alpha = 1.0, 4, 0.5
        else:
            label = f"{variant}: {mode} {target}"
            lw, ms, alpha = 1.2, 5, 0.85
        # de-dup legend entries that look identical
        legend_label = label if label not in legend_seen else None
        if legend_label is not None:
            legend_seen.add(label)
        xs = [r["mean_keyword_fraction"] for r in s]
        ys = [r["mean_coherence"] for r in s]
        ax.plot(xs, ys, marker=marker, color=colour, linewidth=lw, markersize=ms,
                alpha=alpha, label=legend_label)
    ax.set_xlabel("backtracking rate (keyword fraction in B = {wait, hmm})")
    ax.set_ylabel("Sonnet coherence (0 incoherent ↔ 3 fully coherent)")
    ax.set_ylim(-0.1, 3.1)
    ax.set_xlim(left=0)
    ax.set_title("Backtracking case study — all Llama-Scope variants on one Pareto plane")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    save_figure(fig, str(out_dir / "pareto_all.png"))
    plt.close(fig)
    print(f"[summarise] wrote {out_dir / 'pareto_all.png'}")

    # Brief table to stdout.
    print("\n=== Peak keyword fraction at coherence ≥ 2.0 (most useful operating point) ===")
    rows_by_var: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        if r["mean_coherence"] is not None and r["mean_coherence"] >= 2.0:
            rows_by_var[(r["variant"], r["mode"], r["target"])].append(r)
    print(f"{'variant':<22} {'mode':<14} {'target':<14} {'best kw%':>10} {'@ α':>8} {'coh':>6}")
    for key, rs in sorted(rows_by_var.items(), key=lambda kv: -max(r["mean_keyword_fraction"] for r in kv[1])):
        best = max(rs, key=lambda r: r["mean_keyword_fraction"])
        print(f"{key[0]:<22} {key[1]:<14} {key[2]:<14} {best['mean_keyword_fraction']*100:>9.2f}% {best['magnitude']:>8.1f} {best['mean_coherence']:>6.2f}")


if __name__ == "__main__":
    main()
