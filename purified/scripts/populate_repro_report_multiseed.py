"""Multi-seed version of populate_repro_report — averages metrics across seeds.

Reads results/leaderboard.jsonl and writes mean ± std for each (ds, arch, k)
cell across all seeds present.

Usage:
    .venv/bin/python scripts/populate_repro_report_multiseed.py \
        results/leaderboard.jsonl REPRODUCTION_REPORT.md
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


DS_TO_TAG = {
    "toy_coupled_K10_M20_d256": "coupling",
    "toy_markov_n20_d40_noisy": "denoising",
}


DEPRECATED_ARCHS = {"txc_pro", "tfa", "tfa_pos"}

# The synthetic d_sae was changed 40 → 20 on 2026-06-01. The leaderboard
# schema doesn't yet carry resolved arch_hparams, so we cut over by
# timestamp. Anything before this is the historical d_sae=40 over-
# dictionary regime; anything after is the d_sae=20 scarce-dictionary
# regime that the report describes.
D_SAE_CUTOVER_TS = "2026-05-31T22:30:00Z"


def aggregate(leaderboard: Path) -> dict:
    """{(ds, arch, k_pos): {metric: [vals across seeds]}}"""
    by_cell: dict = defaultdict(lambda: defaultdict(list))
    seen_seeds_per_cell: dict = defaultdict(set)
    rows = []
    for line in leaderboard.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("evaluator_protocol_version") != "1.1.0":
            continue
        if r.get("experiment") != "synthetic":
            continue
        if r.get("datasource") not in DS_TO_TAG:
            continue
        if r.get("arch") in DEPRECATED_ARCHS:
            continue
        if r.get("ts", "") < D_SAE_CUTOVER_TS:
            continue
        rows.append(r)
    rows.sort(key=lambda r: r.get("ts", ""))

    # Keep only the latest row per (ds, arch, k_pos, seed) to handle re-runs.
    latest = {}
    for r in rows:
        ds = r["datasource"]
        arch = r["arch"]
        seed = int(r.get("seed", -1))
        override = r.get("training_cfg", {}).get("arch_hparams_override") or {}
        kp = override.get("k_pos") or r.get("eval_cfg", {}).get("k_pos")
        if kp is None:
            continue
        latest[(ds, arch, int(kp), seed)] = r

    for (ds, arch, k, seed), r in latest.items():
        for metric, v in r.get("metrics", {}).items():
            if isinstance(v, (int, float)) and metric in ("eauc", "gauc", "nmse"):
                by_cell[(ds, arch, k)][metric].append(v)
                seen_seeds_per_cell[(ds, arch, k)].add(seed)
    return by_cell, seen_seeds_per_cell


def render_block(by_cell: dict, ds: str, n_seeds_max: int) -> str:
    archs = sorted({a for (d, a, k) in by_cell if d == ds})
    ks = sorted({k for (d, a, k) in by_cell if d == ds})
    if not archs:
        return "_(no cells)_"
    out = []
    for metric in ["eauc", "gauc", "nmse"]:
        label = {"eauc": "eAUC", "gauc": "gAUC", "nmse": "NMSE"}[metric]
        out.append(f"**{label}**  (mean ± std across {n_seeds_max} seed{'s' if n_seeds_max > 1 else ''})\n")
        header = "| arch | " + " | ".join(f"k={k}" for k in ks) + " |"
        sep = "|---|" + "---|" * len(ks)
        out.append(header)
        out.append(sep)
        for arch in archs:
            row = [f"`{arch}`"]
            for k in ks:
                vals = by_cell.get((ds, arch, k), {}).get(metric, [])
                if not vals:
                    row.append("—")
                elif len(vals) == 1:
                    row.append(f"{vals[0]:.3f}")
                else:
                    m = statistics.mean(vals)
                    s = statistics.stdev(vals)
                    row.append(f"{m:.3f}±{s:.3f}")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    by_cell, seeds_per_cell = aggregate(Path(sys.argv[1]))
    report_path = Path(sys.argv[2])
    text = report_path.read_text()

    all_seeds = set()
    for s in seeds_per_cell.values():
        all_seeds |= s
    n_seeds_max = len(all_seeds)

    for ds, tag in DS_TO_TAG.items():
        block = render_block(by_cell, ds, n_seeds_max)
        marker_re = re.compile(
            rf"<!-- BEGIN AUTO-RESULTS {tag} -->.*?<!-- END AUTO-RESULTS {tag} -->",
            re.DOTALL,
        )
        repl = (
            f"<!-- BEGIN AUTO-RESULTS {tag} -->\n"
            f"{block}\n"
            f"<!-- END AUTO-RESULTS {tag} -->"
        )
        if marker_re.search(text):
            text = marker_re.sub(repl, text)
        else:
            print(f"[warn] markers for {tag} not found")

    text = re.sub(r"\n*## Failed cells \(raw\).*$", "", text, flags=re.DOTALL).rstrip() + "\n"
    report_path.write_text(text)
    print(f"[populate] wrote {report_path} (cells: {len(by_cell)}, seeds: {sorted(all_seeds)})")


if __name__ == "__main__":
    main()
