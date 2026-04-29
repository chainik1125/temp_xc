#!/usr/bin/env python3
"""Render only the Pareto frontier across every (variant, mode, target,
magnitude) cell.

The cross-variant `pareto_all.png` plots every curve, which is hard to
read at 10+ variants. This script reduces every cell (one row in
`all_variants.csv`) to a 2D point (kw, coherence) and computes which
points are *not dominated* by any other on (-kw, +coherence). The
non-dominated set is the Pareto frontier — points where you can't
improve on either axis without sacrificing the other.

Plots:
  * the full point cloud, faded
  * the Pareto frontier in bold, with each frontier point labelled
    by `<variant>:<target>@α=<magnitude>`

Output: `plots_summary/pareto_frontier.{png,thumb.png}` and
`plots_summary/pareto_frontier.csv` listing the frontier points.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
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


def _is_dominated(point, others) -> bool:
    """Point (kw, coh) is dominated if any other has kw' >= kw and coh' >= coh
    with at least one strictly greater."""
    pkw, pcoh = point
    for okw, ocoh in others:
        if okw >= pkw and ocoh >= pcoh and (okw > pkw or ocoh > pcoh):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-suffix", default="summary")
    parser.add_argument("--include-clamp", action="store_true")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / f"plots_{args.out_suffix}"
    csv_path = out_dir / "all_variants.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path}; run summarise_variants.py first")

    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            try:
                kw = float(r["mean_keyword_fraction"])
                coh_raw = r.get("mean_coherence", "")
                if coh_raw in ("", None):
                    continue
                coh = float(coh_raw)
            except (ValueError, KeyError):
                continue
            if not args.include_clamp and r["mode"] == "sae_clamp":
                continue
            r["kw"] = kw
            r["coh"] = coh
            rows.append(r)

    points = [(r["kw"], r["coh"]) for r in rows]
    frontier = [r for r, p in zip(rows, points) if not _is_dominated(p, points)]
    frontier.sort(key=lambda r: r["kw"])

    print(f"[frontier] {len(frontier)} Pareto-optimal points out of {len(rows)} cells")
    print(f"{'variant':<22} {'target':<14} {'mode':<14} {'α':>6} {'kw%':>7} {'coh':>5}")
    for r in frontier:
        print(f"{r['variant']:<22} {r['target']:<14} {r['mode']:<14} "
              f"{float(r['magnitude']):>6.1f} {r['kw']*100:>6.2f}% {r['coh']:>5.2f}")

    front_csv = out_dir / "pareto_frontier.csv"
    with front_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "mode", "target", "magnitude", "kw", "coh"], extrasaction="ignore")
        w.writeheader()
        for r in frontier:
            w.writerow(r)
    print(f"[frontier] wrote {front_csv}")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fxs, fys = [r["kw"] for r in rows], [r["coh"] for r in rows]
    ax.scatter(fxs, fys, color="#cccccc", s=18, zorder=1, label="all cells (n={})".format(len(rows)))
    fxs_f = [r["kw"] for r in frontier]
    fys_f = [r["coh"] for r in frontier]
    ax.plot(fxs_f, fys_f, color="black", linestyle="--", linewidth=1.0, zorder=2, alpha=0.6)
    ax.scatter(fxs_f, fys_f, color="tab:red", s=70, zorder=3, label=f"Pareto frontier (n={len(frontier)})")
    for r in frontier:
        label = f"{r['variant']}:{r['target']}@α={float(r['magnitude']):g}"
        ax.annotate(label, xy=(r["kw"], r["coh"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=7)
    ax.set_xlabel("backtracking rate (mean keyword fraction)")
    ax.set_ylabel("Sonnet coherence (0 incoherent ↔ 3 fully coherent)")
    ax.set_ylim(-0.1, 3.1)
    ax.set_xlim(left=-0.005)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Backtracking case study — Pareto frontier across all Llama-Scope variants")
    fig.tight_layout()
    save_figure(fig, str(out_dir / "pareto_frontier.png"))
    plt.close(fig)
    print(f"[frontier] wrote {out_dir / 'pareto_frontier.png'}")


if __name__ == "__main__":
    main()
