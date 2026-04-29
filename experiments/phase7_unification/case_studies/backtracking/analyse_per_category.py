#!/usr/bin/env python3
"""Per-category breakdown of every variant's per-generation results.

The N=300 prompt set spans 10 categories (logic / geometry / probability /
number_theory / combinatorics / algebra / sequences / optimisation / set_theory
/ invariant). Some categories had zero backtracking events in the un-steered
distilled model (algebra, sequences) — interesting in itself, but it also means
those categories can't drive D_+ for the DoM step, so the steering signal
might transfer differently between categories.

This script walks every available `intervene<_suffix>/per_generation.csv` and
emits, per (variant, mode, target, category):
    n_prompts in cell
    mean keyword fraction at the variant's *peak* magnitude
    mean coherence at that magnitude

Output: `plots_summary/per_category.csv` and `plots_summary/per_category.txt`
(human-readable table).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    RESULTS_DIR,
)


def discover_per_gen() -> list[tuple[str, Path]]:
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
        pgen = child / "per_generation.csv"
        if pgen.exists():
            out.append((label, child))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-suffix", default="summary")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / f"plots_{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # cell: (variant, mode, target, category, magnitude) -> [(kw_frac, coh_grade)]
    cells: dict[tuple, list[tuple[float, float | None]]] = defaultdict(list)
    for variant, idir in discover_per_gen():
        with (idir / "per_generation.csv").open() as f:
            for r in csv.DictReader(f):
                cat = r.get("category", "") or "?"
                key = (variant, r["mode"], r["target"], cat, float(r["magnitude"]))
                kw = float(r["keyword_fraction"])
                coh_raw = r.get("coherence_grade", "")
                coh = float(coh_raw) if coh_raw not in ("", None) else None
                cells[key].append((kw, coh))

    # Reduce to means per cell.
    means: dict[tuple, dict[str, float]] = {}
    for key, vals in cells.items():
        n = len(vals)
        kw = sum(v[0] for v in vals) / n
        cohs = [v[1] for v in vals if v[1] is not None]
        coh = (sum(cohs) / len(cohs)) if cohs else float("nan")
        means[key] = {"n": n, "mean_kw": kw, "mean_coh": coh}

    # For each (variant, mode, target, category), pick the magnitude with the
    # highest kw fraction at coherence ≥ 1.5 (otherwise pick highest unfiltered).
    by_curve: dict[tuple, list[tuple]] = defaultdict(list)
    for (variant, mode, target, cat, mag), m in means.items():
        by_curve[(variant, mode, target, cat)].append((mag, m))
    rows = []
    for key, points in by_curve.items():
        points.sort(key=lambda x: -x[1]["mean_kw"])
        # Try to find the highest-kw point at acceptable coherence.
        best = None
        for mag, m in points:
            if not (m["mean_coh"] != m["mean_coh"]) and m["mean_coh"] >= 1.5:  # not-NaN check
                best = (mag, m)
                break
        if best is None:
            best = points[0]
        mag, m = best
        rows.append({
            "variant": key[0], "mode": key[1], "target": key[2], "category": key[3],
            "best_magnitude": mag, "n_prompts": m["n"],
            "mean_kw_fraction": m["mean_kw"],
            "mean_coherence": m["mean_coh"],
        })

    rows.sort(key=lambda r: (r["category"], r["variant"], r["mode"], r["target"]))

    csv_path = out_dir / "per_category.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            if r2["mean_coherence"] != r2["mean_coherence"]:  # NaN
                r2["mean_coherence"] = ""
            w.writerow(r2)
    print(f"[per_cat] wrote {csv_path} ({len(rows)} rows)")

    # Human-readable: per category, list each variant's best.
    txt_path = out_dir / "per_category.txt"
    with txt_path.open("w") as f:
        cur_cat = None
        for r in rows:
            if r["category"] != cur_cat:
                cur_cat = r["category"]
                f.write(f"\n=== category: {cur_cat} ===\n")
                f.write(f"{'variant':<22} {'mode':<14} {'target':<14} {'kw%':>7} {'coh':>5} {'mag':>6} {'n':>4}\n")
            coh_s = f"{r['mean_coherence']:.2f}" if r['mean_coherence'] == r['mean_coherence'] else "  ?"
            f.write(f"{r['variant']:<22} {r['mode']:<14} {r['target']:<14} "
                    f"{r['mean_kw_fraction']*100:>6.2f}% {coh_s:>5} "
                    f"{r['best_magnitude']:>6.1f} {r['n_prompts']:>4}\n")
    print(f"[per_cat] wrote {txt_path}")


if __name__ == "__main__":
    main()
