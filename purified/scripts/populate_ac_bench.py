"""Populate the AC-only signed-motion section of REPRODUCTION_REPORT.md.

Reads results/leaderboard.jsonl, selects the signed_motion rows, and writes
mean ± std (across seeds) tables for the AC metrics, one block per distinct
d_sae present (so the capacity dependence is visible). Renders between the
``<!-- BEGIN AUTO-RESULTS ac_signed_motion -->`` / ``<!-- END ... -->`` markers.

Usage:
    .venv/bin/python scripts/populate_ac_bench.py \
        results/leaderboard.jsonl REPRODUCTION_REPORT.md
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DATASOURCE = "toy_signed_motion_M19_d40"
TAG = "ac_signed_motion"
DEPRECATED_ARCHS = {"txc_pro", "tfa", "tfa_pos"}

# Report only the canonical 10K-step grid (excludes exploratory 3K smoke
# and 30K convergence-check rows that share (d_sae, arch, k_pos, seed) keys).
N_STEPS = 10000

# Synthetic per-section default when an arch_hparams_override omits d_sae.
DEFAULT_SYNTH_D_SAE = 20

# Headline metric first; atom_dc_fraction is sparse (txc only).
METRICS = [
    ("s_temp", "s_temp (headline: 0=chance, 1=oracle)"),
    ("sign_probe_acc", "sign_probe_acc (raw linear-probe accuracy)"),
    ("atom_dc_fraction", "atom_dc_fraction (DC energy share; window decoders only)"),
    ("eauc", "eAUC (alphabet-direction recovery)"),
    ("nmse", "NMSE (window reconstruction)"),
]


def _row_d_sae(r: dict) -> int:
    override = r.get("training_cfg", {}).get("arch_hparams_override") or {}
    return int(override.get("d_sae") or DEFAULT_SYNTH_D_SAE)


def _row_k_pos(r: dict):
    override = r.get("training_cfg", {}).get("arch_hparams_override") or {}
    kp = override.get("k_pos") or r.get("eval_cfg", {}).get("k_pos")
    return None if kp is None else int(kp)


def aggregate(leaderboard: Path):
    """{(d_sae, arch, k_pos): {metric: [vals]}} keeping latest row per seed."""
    rows = []
    for line in leaderboard.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("evaluator_protocol_version") != "1.1.0":
            continue
        if r.get("experiment") != "synthetic":
            continue
        if r.get("datasource") != DATASOURCE:
            continue
        if r.get("arch") in DEPRECATED_ARCHS:
            continue
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        if (r.get("training_cfg", {}) or {}).get("n_steps") != N_STEPS:
            continue
        rows.append(r)
    rows.sort(key=lambda r: r.get("ts", ""))

    latest = {}
    for r in rows:
        kp = _row_k_pos(r)
        if kp is None:
            continue
        seed = int(r.get("seed", -1))
        latest[(_row_d_sae(r), r["arch"], kp, seed)] = r

    by_cell = defaultdict(lambda: defaultdict(list))
    seeds = set()
    for (dsae, arch, kp, seed), r in latest.items():
        seeds.add(seed)
        for metric, _ in METRICS:
            v = r.get("metrics", {}).get(metric)
            if isinstance(v, (int, float)):
                by_cell[(dsae, arch, kp)][metric].append(v)
    return by_cell, seeds


def _fmt(vals: list) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f}±{statistics.stdev(vals):.3f}"


def render(by_cell: dict, n_seeds: int) -> str:
    dsaes = sorted({d for (d, a, k) in by_cell})
    if not dsaes:
        return "_(no signed_motion cells yet)_"
    out = []
    for dsae in dsaes:
        archs = sorted({a for (d, a, k) in by_cell if d == dsae})
        ks = sorted({k for (d, a, k) in by_cell if d == dsae})
        out.append(f"### d_sae = {dsae}\n")
        for metric, label in METRICS:
            # Skip a metric entirely if no cell at this d_sae reports it.
            if not any(by_cell.get((dsae, a, k), {}).get(metric)
                       for a in archs for k in ks):
                continue
            out.append(f"**{label}**  (mean ± std across ≤{n_seeds} seeds)\n")
            out.append("| arch | " + " | ".join(f"k_pos={k}" for k in ks) + " |")
            out.append("|---|" + "---|" * len(ks))
            for arch in archs:
                cells = [_fmt(by_cell.get((dsae, arch, k), {}).get(metric, [])) for k in ks]
                out.append(f"| `{arch}` | " + " | ".join(cells) + " |")
            out.append("")
    return "\n".join(out)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    by_cell, seeds = aggregate(Path(sys.argv[1]))
    report_path = Path(sys.argv[2])
    text = report_path.read_text()
    block = render(by_cell, len(seeds))

    marker_re = re.compile(
        rf"<!-- BEGIN AUTO-RESULTS {TAG} -->.*?<!-- END AUTO-RESULTS {TAG} -->",
        re.DOTALL,
    )
    repl = f"<!-- BEGIN AUTO-RESULTS {TAG} -->\n{block}\n<!-- END AUTO-RESULTS {TAG} -->"
    if marker_re.search(text):
        text = marker_re.sub(repl, text)
        report_path.write_text(text)
        print(f"[populate-ac] wrote {report_path} "
              f"(cells: {len(by_cell)}, seeds: {sorted(seeds)})")
    else:
        print(f"[populate-ac] markers for {TAG} not found in {report_path}; "
              "add a <!-- BEGIN/END AUTO-RESULTS ac_signed_motion --> block.")


if __name__ == "__main__":
    main()
