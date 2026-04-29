#!/usr/bin/env python3
"""Per-cell selectivity analysis: fraction of prompts with successful
induction (coherence ≥ COH_MIN and keyword fraction ≥ KW_MIN at the
given magnitude), as a function of magnitude.

Motivation: averaged metrics (mean kw, mean coherence) collapse bimodal
distributions into single numbers. The multipos variant in particular
showed bimodal behaviour at α=12 — half the prompts cleanly preserved
their reasoning, half collapsed into repetition loops. The mean
metrics looked Pareto-dominant only because of how the bimodality
shifted the centroid.

This script computes, for each (variant, mode, target, magnitude):
    n_prompts                      = total in cell
    n_coherent                     = coh >= COH_MIN
    n_induced                      = kw >= KW_MIN  AND coh >= COH_MIN
                                     (successful coherent induction)
    successful_induction_rate      = n_induced / n_prompts
    n_collapsed                    = coh < 1.0 (fully degenerate)

Output: `plots_summary/selectivity.csv` plus a sorted human-readable
table to stdout, and a Pareto-style bar plot.
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
        if (child / "per_generation.csv").exists():
            out.append((label, child))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coh-min", type=float, default=2.0,
                        help="minimum coherence for 'successful induction' (default 2.0 = mostly coherent)")
    parser.add_argument("--kw-min", type=float, default=0.05,
                        help="minimum keyword fraction (default 0.05 = 5%)")
    parser.add_argument("--out-suffix", default="summary")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / f"plots_{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cell: dict[tuple, list[tuple[float, float | None]]] = defaultdict(list)
    for variant, idir in discover_per_gen():
        with (idir / "per_generation.csv").open() as f:
            for r in csv.DictReader(f):
                key = (variant, r["mode"], r["target"], float(r["magnitude"]))
                kw = float(r["keyword_fraction"])
                cg = r.get("coherence_grade", "")
                coh = float(cg) if cg not in ("", None) else None
                cell[key].append((kw, coh))

    rows = []
    for key, vals in cell.items():
        n = len(vals)
        n_coh = sum(1 for kw, c in vals if c is not None and c >= args.coh_min)
        n_ind = sum(1 for kw, c in vals if c is not None and c >= args.coh_min and kw >= args.kw_min)
        n_col = sum(1 for kw, c in vals if c is not None and c < 1.0)
        rows.append({
            "variant": key[0], "mode": key[1], "target": key[2], "magnitude": key[3],
            "n_prompts": n, "n_coherent": n_coh, "n_induced": n_ind, "n_collapsed": n_col,
            "successful_induction_rate": n_ind / n if n else 0.0,
        })
    rows.sort(key=lambda r: -r["successful_induction_rate"])

    csv_path = out_dir / "selectivity.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[selectivity] wrote {csv_path}")

    # Print top 20 by successful induction rate.
    print(f"\n=== Successful induction rate (coh ≥ {args.coh_min}, kw ≥ {args.kw_min*100:.0f}%) ===")
    print(f"{'variant':<22} {'mode':<14} {'target':<14} {'mag':>6} {'rate':>8} {'n_ind':>6} {'n_col':>6}")
    for r in rows[:25]:
        print(f"{r['variant']:<22} {r['mode']:<14} {r['target']:<14} "
              f"{r['magnitude']:>6.1f} {r['successful_induction_rate']*100:>6.1f}%  "
              f"{r['n_induced']:>5} / {r['n_collapsed']:>5}")


if __name__ == "__main__":
    main()
