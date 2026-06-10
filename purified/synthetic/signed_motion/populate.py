"""Populate the AUTO-RESULTS tables in synthetic/signed_motion/bench.md.

Reads results/leaderboard.jsonl, selects the signed_motion rows (protocol
1.2.0, 10K-step grid), and writes mean ± std tables keyed by (arch, T) × d_sae
for each metric. F = 19 (the feature count) is the d_sae anchor; the per-tile
sign probe is memorization-free only for d_sae < 2F = 38.

Usage:
    .venv/bin/python synthetic/signed_motion/populate.py \
        results/leaderboard.jsonl synthetic/signed_motion/bench.md
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
N_STEPS = 10000
DEFAULT_SYNTH_D_SAE = 20

METRICS = [
    ("s_temp", "s_temp (sign recovery: 0=chance, 1=oracle)"),
    ("sign_probe_acc", "sign_probe_acc (raw linear-probe accuracy)"),
    ("eauc", "eAUC (local: alphabet-direction recovery)"),
    ("nmse", "NMSE (windowed reconstruction; lower=better)"),
]
# Display order for the (arch, T) rows.
ARCH_ORDER = {"txc_base": 0, "stacked_sae": 1, "topk_sae": 2, "tsae": 3}


def _override(r: dict) -> dict:
    return r.get("training_cfg", {}).get("arch_hparams_override") or {}


def _row_key(r: dict):
    ov = _override(r)
    d = int(ov.get("d_sae") or DEFAULT_SYNTH_D_SAE)
    T = int(ov.get("T") or 1)
    return r["arch"], T, d


def aggregate(leaderboard: Path):
    rows = []
    for line in leaderboard.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("evaluator_protocol_version") != "1.2.0":   # matches SyntheticRecovery
            continue
        if r.get("experiment") != "synthetic" or r.get("datasource") != DATASOURCE:
            continue
        if r.get("arch") in DEPRECATED_ARCHS or r.get("eval_cfg", {}).get("smoke"):
            continue
        if (r.get("training_cfg", {}) or {}).get("n_steps") != N_STEPS:
            continue
        rows.append(r)
    rows.sort(key=lambda r: r.get("ts", ""))

    latest = {}
    for r in rows:
        arch, T, d = _row_key(r)
        latest[(arch, T, d, int(r.get("seed", -1)))] = r

    by_cell = defaultdict(lambda: defaultdict(list))
    seeds = set()
    for (arch, T, d, seed), r in latest.items():
        seeds.add(seed)
        for metric, _ in METRICS:
            v = r.get("metrics", {}).get(metric)
            if isinstance(v, (int, float)):
                by_cell[(arch, T, d)][metric].append(v)
    return by_cell, seeds


def _fmt(vals: list) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f}±{statistics.stdev(vals):.3f}"


def render(by_cell: dict, n_seeds: int) -> str:
    rows = sorted({(a, T) for (a, T, d) in by_cell},
                  key=lambda x: (ARCH_ORDER.get(x[0], 9), x[1]))
    dsaes = sorted({d for (a, T, d) in by_cell})
    if not rows:
        return "_(no signed_motion cells yet)_"
    out = [
        f"F = 19 feature directions. Probe is memorization-free for d_sae < 2F = 38 "
        f"(d_sae=38 is the over-complete reference; its s_temp is confounded by "
        f"tabulation). Mean ± std across ≤{n_seeds} seeds.\n"
    ]
    for metric, label in METRICS:
        out.append(f"**{label}**\n")
        out.append("| arch (T) | " + " | ".join(f"d_sae={d}" for d in dsaes) + " |")
        out.append("|---|" + "---|" * len(dsaes))
        for (arch, T) in rows:
            cells = [_fmt(by_cell.get((arch, T, d), {}).get(metric, [])) for d in dsaes]
            out.append(f"| `{arch}` (T={T}) | " + " | ".join(cells) + " |")
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
        report_path.write_text(marker_re.sub(repl, text))
        print(f"[populate-ac] wrote {report_path} "
              f"(cells: {len(by_cell)}, seeds: {sorted(seeds)})")
    else:
        print(f"[populate-ac] markers for {TAG} not found in {report_path}")


if __name__ == "__main__":
    main()
