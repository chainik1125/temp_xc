"""C1 analysis — feature recovery AUC vs k_pos sweep.

Reads c1 leaderboard rows (filtering smoke + non-canonical), computes
mean ± std across seeds per (arch, T-label, k_pos), writes
AUTO-RESULTS to ``docs/components/c1.md``.

Headline: AUC vs k_pos line per arch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from temp_bench.cache import _read_jsonl, leaderboard_path


COMPONENT = "c1"

CANONICAL_ARCH_TS: list[tuple[str, str]] = [
    ("topk_sae",    "default"),
    ("tsae_paper",  "default"),
    ("tfa",         "default"),
    ("tfa_pos",     "default"),
    ("stacked_sae", "T=2"),
    ("stacked_sae", "default"),  # T=5
    ("txc_base",    "default"),  # T=5
    ("txc_pro",     "default"),  # T_max=10
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
            markdown="_No canonical c1 cells in leaderboard yet._",
            results={"n_rows": 0},
        )

    grouped: dict[tuple[str, str, int], list[float]] = {}
    for r in rows:
        arch = r["arch"]
        cfg = r.get("eval_cfg", {})
        t_label = cfg.get("t_label", "default")
        k_pos = int(cfg.get("k_pos", -1))
        if k_pos < 0:
            continue
        key = (arch, t_label, k_pos)
        grouped.setdefault(key, []).append(
            float(r["metrics"].get("auc", float("nan")))
        )

    agg: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, vals in grouped.items():
        agg[key] = {
            "auc_mean": float(np.nanmean(vals)),
            "auc_std":  float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    ks = sorted({k for (_, _, k) in agg.keys()})

    out_lines: list[str] = []
    out_lines.append("**Headline: feature recovery AUC vs k_pos**\n")
    header = "| arch | T | " + " | ".join(f"k={k}" for k in ks) + " |"
    sep    = "|---|---|" + "|".join(["---:" for _ in ks]) + "|"
    out_lines.extend([header, sep])
    for arch, t_label in CANONICAL_ARCH_TS:
        cells = []
        for k in ks:
            stat = agg.get((arch, t_label, k))
            if stat is None:
                cells.append("—")
                continue
            mean = stat["auc_mean"]
            std = stat["auc_std"]
            cells.append(f"{mean:.3f}±{std:.3f}")
        out_lines.append(
            f"| `{arch}` | {t_label} | " + " | ".join(cells) + " |"
        )

    out_lines.append("")
    out_lines.append(
        f"_Cells aggregated over seeds; n shown per cell. Filter: "
        f"`component='{COMPONENT}'`, `smoke=False`. Skipped cells "
        f"(k_train > arch budget at toy d_sae=40) appear as `—`._"
    )

    return AnalysisResult(
        markdown="\n".join(out_lines),
        results={
            "n_rows": len(rows),
            "n_cells": len(grouped),
            "ks": ks,
        },
    )


def main():
    result = run_analysis()
    print(result.markdown)


if __name__ == "__main__":
    main()
