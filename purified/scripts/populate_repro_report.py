"""Inject minisweep results into REPRODUCTION_REPORT.md.

Reads the parsed sweep log and replaces the AUTO-RESULTS marker blocks
with per-bench gAUC/eAUC/NMSE tables. Idempotent.

Usage:
    .venv/bin/python scripts/populate_repro_report.py \
        logs/synth_minisweep_seed1.log docs/reproduction_report.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from parse_minisweep_log import parse


BENCH_SHORT_TO_TAG = {
    "coupled K10 M20 d256": "coupling",
    "markov n20 d40 noisy": "denoising",
}


def _render_block(archs: dict) -> str:
    if not archs:
        return "_(no cells in this bench yet)_"
    all_ks = sorted({k for d in archs.values() for k in d})
    lines = []
    for metric_idx, metric_name in enumerate(["eAUC", "gAUC", "NMSE"]):
        lines.append(f"**{metric_name}**\n")
        header = "| arch | " + " | ".join(f"k={k}" for k in all_ks) + " |"
        sep = "|---|" + "---|" * len(all_ks)
        lines.append(header)
        lines.append(sep)
        for arch in sorted(archs.keys()):
            row = [f"`{arch}`"]
            for k in all_ks:
                if k in archs[arch]:
                    val = archs[arch][k][metric_idx]
                    row.append(f"{val:.3f}" if val is not None else "—")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    log = Path(sys.argv[1])
    report = Path(sys.argv[2])
    parsed = parse(log)

    text = report.read_text()

    for short, tag in BENCH_SHORT_TO_TAG.items():
        archs = parsed["by_bench"].get(short, {})
        block = _render_block(archs)
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
            print(f"[warn] markers for {tag} not found in {report}")

    # Replace (not append) the failure section so re-runs stay idempotent.
    fail_re = re.compile(r"\n*## Failed cells \(raw\).*$", re.DOTALL)
    text = fail_re.sub("", text).rstrip() + "\n"
    if parsed["failures"]:
        text += "\n## Failed cells (raw)\n\n"
        for f in parsed["failures"]:
            text += f"- {f}\n"

    report.write_text(text)
    print(f"[populate] wrote {report} (failures: {len(parsed['failures'])})")


if __name__ == "__main__":
    main()
