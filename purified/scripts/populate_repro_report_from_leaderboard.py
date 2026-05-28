"""Inject results from the canonical leaderboard into REPRODUCTION_REPORT.md.

Reads results/leaderboard.jsonl directly (instead of parsing the sweep log),
so retry-rerun cells are included automatically.

Usage:
    .venv/bin/python scripts/populate_repro_report_from_leaderboard.py \
        results/leaderboard.jsonl REPRODUCTION_REPORT.md
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


DS_TO_TAG = {
    "toy_coupled_K10_M20_d256": "coupling",
    "toy_markov_n20_d40_noisy": "denoising",
}


def aggregate(leaderboard: Path) -> dict:
    """{(ds, arch, k_pos): {eauc, gauc, nmse}} — latest row per cell."""
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
        rows.append(r)
    # Sort by timestamp so .get() picks the most recent.
    rows.sort(key=lambda r: r.get("ts", ""))
    cells: dict = {}
    for r in rows:
        ds = r["datasource"]
        arch = r["arch"]
        override = r.get("training_cfg", {}).get("arch_hparams_override") or {}
        kp = override.get("k_pos")
        if kp is None:
            kp = r.get("eval_cfg", {}).get("k_pos")
        if kp is None:
            continue
        cells[(ds, arch, int(kp))] = r.get("metrics", {})
    return cells


def render_block(cells: dict, ds: str) -> str:
    archs = sorted({a for (d, a, k) in cells if d == ds})
    ks = sorted({k for (d, a, k) in cells if d == ds})
    if not archs:
        return "_(no cells)_"
    out = []
    for metric in ["eauc", "gauc", "nmse"]:
        label = {"eauc": "eAUC", "gauc": "gAUC", "nmse": "NMSE"}[metric]
        out.append(f"**{label}**\n")
        header = "| arch | " + " | ".join(f"k={k}" for k in ks) + " |"
        sep = "|---|" + "---|" * len(ks)
        out.append(header)
        out.append(sep)
        for arch in archs:
            row = [f"`{arch}`"]
            for k in ks:
                m = cells.get((ds, arch, k), {})
                val = m.get(metric)
                if val is None:
                    row.append("—")
                else:
                    row.append(f"{val:.3f}")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    cells = aggregate(Path(sys.argv[1]))
    report_path = Path(sys.argv[2])
    text = report_path.read_text()

    for ds, tag in DS_TO_TAG.items():
        block = render_block(cells, ds)
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

    # Drop stale "Failed cells" section.
    text = re.sub(r"\n*## Failed cells \(raw\).*$", "", text, flags=re.DOTALL).rstrip() + "\n"
    report_path.write_text(text)
    n_cells = len(cells)
    print(f"[populate] wrote {report_path} (cells: {n_cells})")


if __name__ == "__main__":
    main()
