"""C2 analysis — gAUC vs k-sweep + T-modulation.

Reads c2 leaderboard rows (filtering smoke + non-canonical), computes
mean ± std across seeds per (arch, T-label, k_pos), writes
AUTO-RESULTS to ``docs/components/c2.md``.

Headline:
- ``gAUC vs k`` line plot per (arch, T-label).
- ``eAUC vs k`` line plot per (arch, T-label).
- T-modulation table: TXC-pro at T ∈ {2, 5, 12}, gAUC vs k.

Invoked via:

    .venv/bin/python -m experiments.c2_synthetic_coupled.analysis
    # or
    .venv/bin/python -c "from temp_bench import report; report.render(component='c2')"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from temp_bench.cache import _read_jsonl, leaderboard_path


COMPONENT = "c2"

# Order in which arch+T appear in tables / plots. Includes only canonical
# C2 sweep cells (skipping smoke); the analysis filter drops anything
# else.
CANONICAL_ARCH_TS: list[tuple[str, str]] = [
    ("topk_sae",    "default"),
    ("stacked_sae", "T=2"),
    ("stacked_sae", "default"),  # T=5
    ("txc_base",    "default"),  # T=5
    ("txc_pro",     "T=2"),
    ("txc_pro",     "T=5"),
    ("txc_pro",     "T=12"),
]


@dataclass
class AnalysisResult:
    markdown: str
    results: dict[str, Any]


def run_analysis() -> AnalysisResult:
    rows = [
        r for r in _read_jsonl(leaderboard_path())
        if r.get("component") == COMPONENT
        and not r.get("eval_cfg", {}).get("smoke", False)
    ]
    if not rows:
        return AnalysisResult(
            markdown="_No canonical c2 cells in leaderboard yet._",
            results={"n_rows": 0},
        )

    # Group by (arch, t_label, k_pos).
    grouped: dict[tuple[str, str, int], dict[str, list[float]]] = {}
    for r in rows:
        arch = r["arch"]
        cfg = r.get("eval_cfg", {})
        t_label = cfg.get("t_label", "default")
        k_pos = int(cfg.get("k_pos", -1))
        if k_pos < 0:
            continue
        key = (arch, t_label, k_pos)
        grouped.setdefault(key, {"eauc": [], "gauc": []})
        grouped[key]["eauc"].append(float(r["metrics"].get("eauc", float("nan"))))
        grouped[key]["gauc"].append(float(r["metrics"].get("gauc", float("nan"))))

    # Aggregate (mean, std, n).
    agg: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, vals in grouped.items():
        agg[key] = {
            "eauc_mean": float(np.nanmean(vals["eauc"])),
            "eauc_std":  float(np.nanstd(vals["eauc"], ddof=1)) if len(vals["eauc"]) > 1 else 0.0,
            "gauc_mean": float(np.nanmean(vals["gauc"])),
            "gauc_std":  float(np.nanstd(vals["gauc"], ddof=1)) if len(vals["gauc"]) > 1 else 0.0,
            "n": len(vals["gauc"]),
        }

    # k_pos values present.
    ks = sorted({k for (_, _, k) in agg.keys()})

    # ── Build markdown tables ──
    out_lines: list[str] = []

    out_lines.append("**Headline: gAUC (global feature recovery) vs k_pos**\n")
    out_lines.append(_render_table(agg, ks, metric="gauc"))
    out_lines.append("")
    out_lines.append("**eAUC (emission feature recovery) vs k_pos**\n")
    out_lines.append(_render_table(agg, ks, metric="eauc"))
    out_lines.append("")
    out_lines.append(
        f"_Cells aggregated over seeds; n shown per cell. Filter: "
        f"`component='{COMPONENT}'`, `smoke=False`._"
    )

    return AnalysisResult(
        markdown="\n".join(out_lines),
        results={
            "n_rows": len(rows),
            "n_cells": len(grouped),
            "ks": ks,
            "agg_keys": [list(k) for k in agg.keys()],
        },
    )


def _render_table(
    agg: dict[tuple[str, str, int], dict[str, float]],
    ks: list[int],
    *,
    metric: str,
) -> str:
    header = "| arch | T | " + " | ".join(f"k={k}" for k in ks) + " |"
    sep    = "|---|---|" + "|".join(["---:" for _ in ks]) + "|"
    rows = [header, sep]
    for arch, t_label in CANONICAL_ARCH_TS:
        cells = []
        for k in ks:
            stat = agg.get((arch, t_label, k))
            if stat is None:
                cells.append("—")
                continue
            mean = stat[f"{metric}_mean"]
            std = stat[f"{metric}_std"]
            cells.append(f"{mean:.3f}±{std:.3f}")
        rows.append(
            f"| `{arch}` | {t_label} | " + " | ".join(cells) + " |"
        )
    return "\n".join(rows)


def main():
    result = run_analysis()
    print(result.markdown)


if __name__ == "__main__":
    main()
