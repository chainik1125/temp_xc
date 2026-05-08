#!/usr/bin/env python3
"""Fill the phase3_benchmark.md results table from a comparison JSON.

Reads:
  - <comparison>.json from scripts/plot_phase3_comparison.py
    Schema: {"coh_floor": 70, "methods": {label: [{seed, delta, peak, n70}, ...], ...}}

Writes:
  - phase3_benchmark.md  — replaces the per-seed Round table for the seeds passed in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        if x != x:
            return "NaN"
        return f"{x:.2f}"
    return str(x)


METHOD_ORDER = [
    ("FRA QK→OV (Nura)", "L24 ln1"),
    ("SAE-resid pre",    "L24 resid_pre"),
    ("SAE-resid mid",    "L24 resid_mid"),
    ("SAE-resid post",   "L24 resid_post"),
    ("SAE-ln1 next",     "L25 ln1"),
]

# Map summary-JSON labels → (display name, hookpoint string)
LABEL_LOOKUP = {
    "FRA QK→OV\n(L24 ln1, Nura)": ("FRA QK→OV (Nura)", "L24 ln1"),
    "SAE\nresid_pre_L24": ("SAE-resid pre",   "L24 resid_pre"),
    "SAE\nresid_mid_L24": ("SAE-resid mid",   "L24 resid_mid"),
    "SAE\nresid_post_L24": ("SAE-resid post", "L24 resid_post"),
    "SAE\nln1_normalised_L25": ("SAE-ln1 next", "L25 ln1"),
}


def build_round(summary, seed):
    rows = ["| # | Method | Hookpoint | Δalign|coh≥70 | peak alignment |",
            "|---|--------|-----------|---------------:|---------------:|"]
    by_display = {}
    for label, entries in summary["methods"].items():
        display = LABEL_LOOKUP.get(label, (label.replace("\n", " "), ""))
        for e in entries:
            if e["seed"] == seed:
                by_display[display[0]] = (display[1], e["delta"], e["peak"])
    for i, (name, hook) in enumerate(METHOD_ORDER, 1):
        d, p = (None, None)
        if name in by_display:
            hook = by_display[name][0]
            d = by_display[name][1]
            p = by_display[name][2]
        rows.append(f"| {i} | {name} | {hook} | {fmt(d)} | {fmt(p)} |")
    return "\n".join(rows)


def splice(doc: str, seeds: list[int], summary: dict) -> str:
    """For each seed in `seeds`, replace the corresponding Round N table."""
    seeds = sorted(set(seeds))
    out = doc

    # Round 1 = first seed; Round 2 = first+second seeds combined; etc.
    rounds = []
    for i, seed in enumerate(seeds, 1):
        title = f"### Round {i} — seed{'s' if i > 1 else ''} = " + ", ".join(str(s) for s in seeds[:i])
        rounds.append((seed, title, build_round(summary, seed)))

    for seed, title, table in rounds:
        # Replace the corresponding "### Round N — seed = TBD" or filled section
        pattern = re.compile(rf"^### Round \d+ —.*?(?=^### |^## |\Z)", re.S | re.M)
        # Naive: only replace the first un-filled Round; for v1 just append.
        if "TBD" in out:
            # Replace first TBD round
            out = re.sub(rf"^### Round \d+ — seed.*?(?=^### |^## |\Z)",
                         title + "\n\n" + table + "\n\n",
                         out, count=1, flags=re.S | re.M)
        else:
            out += "\n" + title + "\n\n" + table + "\n"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison-json", required=True)
    p.add_argument("--doc", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    args = p.parse_args()

    summary = json.loads(Path(args.comparison_json).read_text())
    text = Path(args.doc).read_text()
    new_text = splice(text, args.seeds, summary)
    if new_text != text:
        Path(args.doc).write_text(new_text)
        print(f"[fill] updated {args.doc} with seeds={args.seeds}")
    else:
        print("[fill] no changes")


if __name__ == "__main__":
    sys.exit(main())
