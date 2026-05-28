"""Parse the minisweep log into a clean markdown summary table.

Usage:
    .venv/bin/python scripts/parse_minisweep_log.py logs/synth_minisweep_seed1.log
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


LINE = re.compile(
    r"\[\s*\d+/\d+\]\s+(\S+)\s+(\S+)\s+k=(\d+)\s+\.\.\.\s+"
    r"(?:eAUC=([\d.]+)\s+gAUC=([\d.]*)\s+NMSE=([\d.]+)|FAIL)"
)


def parse(path: Path) -> dict:
    """Return {bench: {arch: {k_pos: (eauc, gauc, nmse)}}}."""
    by_bench: dict[str, dict[str, dict[int, tuple]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    failures: list[str] = []
    for line in path.read_text().splitlines():
        m = LINE.search(line)
        if not m:
            continue
        arch, bench, k, eauc, gauc, nmse = m.groups()
        if eauc is None:
            failures.append(f"{arch} {bench} k={k}")
            continue
        bench_short = bench.replace("toy_", "").replace("_", " ")
        # gAUC is empty for benches without hidden_features (e.g. markov);
        # store NaN-as-None so render can show "—".
        gauc_val = float(gauc) if gauc else None
        by_bench[bench_short][arch][int(k)] = (
            float(eauc), gauc_val, float(nmse),
        )
    return {"by_bench": dict(by_bench), "failures": failures}


def render_md(parsed: dict) -> str:
    out = []
    out.append("# Minisweep results summary\n")
    if parsed["failures"]:
        out.append(f"⚠️  **{len(parsed['failures'])} failed cells**:")
        for f in parsed["failures"][:10]:
            out.append(f"  - {f}")
        out.append("")
    for bench, archs in parsed["by_bench"].items():
        out.append(f"## {bench}\n")
        all_ks = sorted({k for arch_data in archs.values() for k in arch_data})
        for metric_idx, metric_name in enumerate(["eAUC", "gAUC", "NMSE"]):
            out.append(f"### {metric_name}")
            header = "| arch | " + " | ".join(f"k={k}" for k in all_ks) + " |"
            sep = "|---|" + "---|" * len(all_ks)
            out.append(header)
            out.append(sep)
            for arch in sorted(archs.keys()):
                row = [f"`{arch}`"]
                for k in all_ks:
                    if k in archs[arch]:
                        val = archs[arch][k][metric_idx]
                        row.append(f"{val:.3f}" if val is not None else "—")
                    else:
                        row.append("—")
                out.append("| " + " | ".join(row) + " |")
            out.append("")
    return "\n".join(out)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    parsed = parse(Path(sys.argv[1]))
    print(render_md(parsed))


if __name__ == "__main__":
    main()
